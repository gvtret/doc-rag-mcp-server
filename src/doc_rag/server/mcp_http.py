from __future__ import annotations

"""MCP server over Streamable HTTP transport.

Implements a single MCP endpoint (default: /mcp) that supports:
- HTTP POST: JSON-RPC messages (application/json response)
- HTTP GET : optional SSE stream (text/event-stream) for server -> client notifications/requests

This follows the MCP Streamable HTTP transport spec (2025-03-26+).
"""

import asyncio
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)

from doc_rag.raglib.pipeline import (
    clean_orphans as _pipeline_clean_orphans,
)
from doc_rag.raglib.pipeline import (
    clear_incoming as _pipeline_clear_incoming,
)
from doc_rag.raglib.pipeline import (
    delete_documents as _pipeline_delete_documents,
)
from doc_rag.raglib.pipeline import (
    wipe_index as _pipeline_wipe_index,
)
from doc_rag.server import metrics as _metrics
from doc_rag.server.logging_setup import (
    configure_logging,
    get_logger,
    new_request_id,
    set_request_id,
)
from doc_rag.server.retrieval import document_preview, indexed_catalog, load_manifest_json

# Configure structured logging on import. Idempotent; honours
# DOC_RAG_LOG_LEVEL and DOC_RAG_LOG_FORMAT env vars.
configure_logging()
log = get_logger("doc_rag.server.mcp_http")

Json = dict[str, Any] | list[Any]

_TOOL_SEM = asyncio.Semaphore(int(os.environ.get("DOC_RAG_MAX_CONCURRENCY", "4")))

_RL_LOCK = asyncio.Lock()
_RL_STATE: dict[str, tuple[float, float]] = {}


def _rate_limit_params() -> tuple[float, float]:
    """Return (rps, burst). rps<=0 disables rate limiting."""
    try:
        rps = float(os.environ.get("DOC_RAG_RATE_LIMIT_RPS", "0"))
    except Exception:
        rps = 0.0
    try:
        burst = float(os.environ.get("DOC_RAG_RATE_LIMIT_BURST", "5"))
    except Exception:
        burst = 5.0
    return max(0.0, rps), max(1.0, burst)


async def _rate_limit_allow(key: str) -> bool:
    """Simple token bucket per key (usually client IP)."""
    rps, burst = _rate_limit_params()
    if rps <= 0:
        return True

    now = time.time()
    async with _RL_LOCK:
        tokens, last = _RL_STATE.get(key, (burst, now))
        # Refill
        tokens = min(burst, tokens + (now - last) * rps)
        if tokens < 1.0:
            _RL_STATE[key] = (tokens, now)
            return False
        tokens -= 1.0
        _RL_STATE[key] = (tokens, now)
        return True


def _api_key_required() -> str | None:
    # Open-by-default (LAN): if DOC_RAG_API_KEY is not set, allow all requests.
    key = (os.environ.get("DOC_RAG_API_KEY") or "").strip()
    return key or None


def _check_api_key(request: Request) -> bool:
    required = _api_key_required()
    if not required:
        return True
    hdr = (request.headers.get("authorization") or "").strip()
    if hdr.lower().startswith("bearer "):
        return hdr.split(" ", 1)[1].strip() == required
    xk = (request.headers.get("x-api-key") or "").strip()
    return xk == required


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_path() -> Path:
    # Supports DOC_RAG_ROOT override used elsewhere.
    root = Path((os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir()))
    return root / "config" / "config.yaml"


def _ui_restart_allowed() -> bool:
    v = (os.environ.get("DOC_RAG_UI_RESTART_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes")


def _ui_restart_cmd() -> str:
    return (os.environ.get("DOC_RAG_UI_RESTART_CMD") or "").strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _validate_root_yaml(raw: str) -> tuple[bool, str]:
    try:
        import yaml

        data = yaml.safe_load(raw)
    except Exception as e:
        return False, str(e)
    if data is None:
        return False, "YAML пустой или null в корне"
    if not isinstance(data, dict):
        return False, "Корень конфигурации должен быть объектом (mapping), не списком"
    return True, "ok"


def _sources_incoming_dir() -> Path:
    root = Path((os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir()))
    return root / "sources" / "incoming"


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip()
    if not base:
        return "upload.bin"
    # keep it simple: replace path separators and control chars
    base = base.replace("/", "_").replace("\\", "_")
    base = "".join(ch if ch.isprintable() and ch not in ("\n", "\r", "\t") else "_" for ch in base)
    return base[:180]


def _ui_max_upload_files() -> int:
    try:
        n = int(os.environ.get("DOC_RAG_UI_MAX_UPLOAD_FILES", "48"))
    except Exception:
        n = 48
    return max(1, min(200, n))


def _incoming_unique_path(incoming: Path, filename: str) -> Path:
    name = _sanitize_filename(filename)
    dst = incoming / name
    n = 0
    while dst.exists():
        stem = Path(name).stem
        suf = Path(name).suffix
        n += 1
        dst = incoming / f"{stem}__{int(time.time())}_{n}{suf}"
    return dst


def _find_dup_in_manifest(sha256_hex: str) -> str | None:
    """Return basename of the matching indexed source file, or None."""
    try:
        manifest = load_manifest_json()
        if not manifest:
            return None
        for doc in manifest.get("documents", []):
            if isinstance(doc, dict) and doc.get("sha256") == sha256_hex:
                src = doc.get("source_file") or doc.get("title_hint") or doc.get("doc_id") or "?"
                return os.path.basename(src)
    except Exception:
        pass
    return None


def _find_dup_in_incoming(sha256_hex: str, incoming: Path) -> str | None:
    """Return filename of an existing file in incoming/ with the same sha256, or None."""
    try:
        for p in incoming.iterdir():
            if not p.is_file():
                continue
            h = hashlib.sha256()
            try:
                with open(p, "rb") as f:
                    for blk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(blk)
                if h.hexdigest() == sha256_hex:
                    return p.name
            except Exception:
                continue
    except Exception:
        pass
    return None


async def _save_upload_to_incoming(file: UploadFile, incoming: Path) -> tuple[bool, str | None]:
    """Save upload to incoming dir.

    Returns (is_dup, message):
      (False, None) → saved OK
      (True,  msg)  → skipped: duplicate (msg names the conflicting file)
      (False, msg)  → skipped: format/size error
    """
    name = _sanitize_filename(file.filename or "")
    ext = os.path.splitext(name.lower())[1]
    if ext not in (".pdf", ".docx", ".doc", ".md", ".txt"):
        lab = name or (file.filename or "unnamed")
        return False, f"{lab}: поддерживаются .pdf, .docx, .doc, .md, .txt"

    max_mb = int(os.environ.get("DOC_RAG_UI_MAX_UPLOAD_MB", "100"))
    limit = max(1, max_mb) * 1024 * 1024

    buf: list[bytes] = []
    total = 0
    h = hashlib.sha256()
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                return False, f"{name}: файл больше лимита ({max_mb} MiB)"
            buf.append(chunk)
            h.update(chunk)
    except Exception as e:
        return False, f"{name}: {e}"
    sha256_hex = h.hexdigest()

    dup = _find_dup_in_manifest(sha256_hex)
    if dup:
        return True, f"{name}: уже проиндексирован — совпадает с «{dup}»"

    dup = _find_dup_in_incoming(sha256_hex, incoming)
    if dup:
        return True, f"{name}: уже в очереди — совпадает с «{dup}»"

    dst = _incoming_unique_path(incoming, name)
    try:
        with open(dst, "wb") as out:
            for blk in buf:
                out.write(blk)
    except Exception as e:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"{name}: {e}"
    return False, None


def _ui_key_ok(request: Request, key: str) -> bool:
    """Allow UI auth via query/form key when DOC_RAG_API_KEY is set.

    Rationale: plain HTML forms cannot set Authorization headers.
    """
    required = _api_key_required()
    if not required:
        return True
    k = (key or "").strip()
    if k and k == required:
        return True
    # Also allow header-based auth if caller can set it.
    return _check_api_key(request)


def _public_base_url(request: Request) -> str:
    # Prefer X-Forwarded-* when behind a reverse proxy.
    proto = (request.headers.get("x-forwarded-proto") or "").strip()
    host = (request.headers.get("x-forwarded-host") or "").strip()
    if not proto:
        proto = request.url.scheme
    if not host:
        host = request.headers.get("host", "").strip() or request.url.netloc
    return f"{proto}://{host}"


def _mcp_config_payload(name: str, url: str, api_key: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "mcpServers": {
            name: {
                "transport": "streamableHttp",
                "url": url,
            }
        }
    }
    if api_key:
        # Not all clients support headers in config, but include it for VSCode/others when possible.
        cfg["mcpServers"][name]["headers"] = {"Authorization": f"Bearer {api_key}"}
    return cfg


def _download_json(payload: dict[str, Any], filename: str) -> Response:
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_INGEST_LOCK = asyncio.Lock()


def _ingest_ui_log_max_lines() -> int:
    try:
        n = int(os.environ.get("DOC_RAG_UI_INGEST_LOG_MAX_LINES", "500"))
    except Exception:
        n = 500
    return max(50, min(5000, n))


def _ingest_ui_log_line_max_chars() -> int:
    try:
        n = int(os.environ.get("DOC_RAG_UI_INGEST_LOG_LINE_MAX", "8000"))
    except Exception:
        n = 8000
    return max(200, min(100_000, n))


def _ingest_ui_log_scratch_max_chars() -> int:
    try:
        n = int(os.environ.get("DOC_RAG_UI_INGEST_LOG_BUF_MAX", str(128 * 1024)))
    except Exception:
        n = 128 * 1024
    return max(4096, min(8 * 1024 * 1024, n))


try:
    _INGEST_UI_LOG_POLL_MS = max(
        250, min(600_000, int(os.environ.get("DOC_RAG_UI_INGEST_POLL_MS", "2000")))
    )
except Exception:
    _INGEST_UI_LOG_POLL_MS = 2000


def _server_http_ui_log_max_lines() -> int:
    try:
        n = int(os.environ.get("DOC_RAG_UI_HTTP_LOG_MAX_LINES", "500"))
    except Exception:
        n = 500
    return max(50, min(5000, n))


_INGEST_LOG_LOCK = threading.Lock()
_INGEST_UI_LOG_LINES: deque[str] = deque(maxlen=_ingest_ui_log_max_lines())
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_INGEST_STATE: dict[str, Any] = {
    "running": False,
    "job": None,
    "last_started": None,
    "last_finished": None,
    "last_ok": None,
    "last_error": None,
    # Per-document progress derived from pipeline log lines. `current_doc`
    # is the basename being parsed right now; `docs_done` counts files
    # finished in this run (ok + skip + failed); `docs_total` is the size
    # of the input queue once known. `current_doc_started_at` is used by
    # the ETA calculator in `_ui_status_payload`.
    "current_doc": None,
    "docs_done": 0,
    "docs_total": None,
    "current_doc_started_at": None,
}


def _reset_progress_state() -> None:
    _INGEST_STATE["current_doc"] = None
    _INGEST_STATE["docs_done"] = 0
    _INGEST_STATE["docs_total"] = None
    _INGEST_STATE["current_doc_started_at"] = None


# Markers from pipeline._log lines we use to derive per-doc progress.
# Keep these in sync with src/doc_rag/raglib/pipeline.py.
_RX_FOUND = re.compile(r"\bfound\s+(\d+)\s+file\(s\)")
_RX_REBUILD_PASS = re.compile(
    r"\brebuild:\s+(?:archived|incoming(?:\s+ingest)?)\s+pass\s+\((\d+)\s+file\(s\)\)"
)
_RX_PARSE = re.compile(r"\bparse:\s+(.+?)\s*$")
_RX_DONE = re.compile(r"\b(?:ok|skip|failed):\s+(.+?)(?::|\s|$)")


def _apply_progress_from_line(line: str) -> None:
    """Update `_INGEST_STATE` progress fields from a single log line.

    Best-effort: a malformed or unfamiliar line is silently ignored. The
    parser only ever updates `current_doc` / `docs_done` / `docs_total`
    / `current_doc_started_at` — never the lifecycle fields owned by
    `_run_index_job_background`."""
    if not line:
        return

    m = _RX_FOUND.search(line)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            n = None
        if n is not None:
            cur = _INGEST_STATE.get("docs_total")
            # ingest() emits one "found N" line. rebuild() emits one per
            # pass (archived, incoming); the totals accumulate.
            if isinstance(cur, int):
                _INGEST_STATE["docs_total"] = cur + n
            else:
                _INGEST_STATE["docs_total"] = n
        return

    m = _RX_PARSE.search(line)
    if m:
        path = m.group(1).strip()
        _INGEST_STATE["current_doc"] = os.path.basename(path) if path else None
        _INGEST_STATE["current_doc_started_at"] = time.time()
        return

    m = _RX_DONE.search(line)
    if m:
        try:
            cur = int(_INGEST_STATE.get("docs_done") or 0)
        except (TypeError, ValueError):
            cur = 0
        _INGEST_STATE["docs_done"] = cur + 1
        _INGEST_STATE["current_doc"] = None
        _INGEST_STATE["current_doc_started_at"] = None


def _reset_ingest_ui_log_lines() -> None:
    global _INGEST_UI_LOG_LINES
    with _INGEST_LOG_LOCK:
        _INGEST_UI_LOG_LINES = deque(maxlen=_ingest_ui_log_max_lines())


def _append_ingest_ui_line(line: str, *, ansi_strip: bool = True) -> None:
    lm = _ingest_ui_log_line_max_chars()
    raw = line if not ansi_strip else _ANSI_RE.sub("", line)
    raw = raw.rstrip("\n\r\t ")
    # Do not strip leading whitespace: keeps tracebacks/indented diagnostics readable.
    if not raw.strip():
        return
    raw_len = len(raw)
    if raw_len > lm:
        raw = raw[: lm - 35] + f" … (trimmed from {raw_len} chars)"
    with _INGEST_LOG_LOCK:
        _INGEST_UI_LOG_LINES.append(raw)
    # Update per-doc progress on the way through. Kept outside the log
    # lock so a slow regex (it isn't) cannot block log writers.
    _apply_progress_from_line(raw)


def _snapshot_ingest_ui_log_tail() -> list[str]:
    with _INGEST_LOG_LOCK:
        return list(_INGEST_UI_LOG_LINES)


_SERVER_HTTP_LOG_LOCK = threading.Lock()
_SERVER_HTTP_UI_LOG_LINES: deque[str] = deque(maxlen=_server_http_ui_log_max_lines())


def _append_server_http_ui_line(msg: str) -> None:
    """Buffered copy of formatted HTTP access log lines for /ui (avoids reliance on journald)."""
    lm = _ingest_ui_log_line_max_chars()
    raw = _ANSI_RE.sub("", msg).rstrip("\n\r\t ")
    if not raw.strip():
        return
    raw_len = len(raw)
    if raw_len > lm:
        raw = raw[: lm - 35] + f" … (trimmed from {raw_len} chars)"
    with _SERVER_HTTP_LOG_LOCK:
        _SERVER_HTTP_UI_LOG_LINES.append(raw)


def _snapshot_server_http_ui_log_tail() -> list[str]:
    with _SERVER_HTTP_LOG_LOCK:
        return list(_SERVER_HTTP_UI_LOG_LINES)


class _TeeStderrToIngestLog:
    """Tee stderr to the real stderr and split into ingest UI log lines.

    Mimics carriage return semantics for progress bars (\r rewinds display).
    """

    __slots__ = ("_underlying", "_scratch", "_emit_line", "_enc", "_err")

    def __init__(self, underlying: Any, emit_line: Callable[[str], None]) -> None:
        self._underlying = underlying
        self._scratch = ""
        self._emit_line = emit_line
        self._enc = getattr(underlying, "encoding", None) or "utf-8"
        self._err = getattr(underlying, "errors", None) or "replace"

    def writable(self) -> bool:
        return True

    def write(self, s: Any) -> int:
        if s is None:
            return 0
        enc = getattr(self, "_enc", "utf-8") or "utf-8"
        err = getattr(self, "_err", "replace") or "replace"
        if isinstance(s, bytes):
            s_dec = s.decode(enc, errors=err)
        elif isinstance(s, str):
            s_dec = s
        else:
            s_dec = str(s)
        try:
            self._underlying.write(s_dec)
        except Exception:
            pass
        mx = _ingest_ui_log_scratch_max_chars()
        scr = self._scratch + s_dec.replace("\r\n", "\n")
        # Terminal-style \r: drop everything before final segment (overwrite same line).
        while "\r" in scr:
            scr = scr.split("\r", 1)[1]
        while True:
            pos = scr.find("\n")
            if pos == -1:
                break
            line = scr[:pos]
            scr = scr[pos + 1 :]
            self._emit_line(line)
        if len(scr) > mx:
            self._emit_line(
                f"[doc-rag][ui-log] … промежуточный вывод stderr обрезан ({len(scr)} симв.) …"
            )
            scr = scr[len(scr) - mx // 4 :]
        self._scratch = scr
        return len(s_dec)

    def flush(self) -> None:
        try:
            self._underlying.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        fn = getattr(self._underlying, "isatty", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return False

    def finish(self) -> None:
        if self._scratch.strip():
            self._emit_line(self._scratch)
            self._scratch = ""


def _sync_run_ingest_with_ui_logging() -> None:
    from doc_rag.raglib.pipeline import ingest

    stderr = getattr(sys, "stderr", None) or sys.__stderr__
    tee = _TeeStderrToIngestLog(stderr, _append_ingest_ui_line)
    old_stderr = sys.stderr
    sys.stderr = tee
    try:
        ingest(str(_config_path()))
    finally:
        sys.stderr = old_stderr
        tee.finish()


def _sync_run_rebuild_with_ui_logging() -> None:
    from doc_rag.raglib.pipeline import rebuild

    stderr = getattr(sys, "stderr", None) or sys.__stderr__
    tee = _TeeStderrToIngestLog(stderr, _append_ingest_ui_line)
    old_stderr = sys.stderr
    sys.stderr = tee
    try:
        rebuild(str(_config_path()))
    finally:
        sys.stderr = old_stderr
        tee.finish()


async def _run_index_job_background(job: str, sync_runner: Callable[[], None]) -> None:
    """Run ingest or rebuild in a worker thread; mutual exclusion via _INGEST_LOCK."""
    async with _INGEST_LOCK:
        if _INGEST_STATE.get("running"):
            return
        _reset_ingest_ui_log_lines()
        _reset_progress_state()
        _INGEST_STATE["running"] = True
        _INGEST_STATE["job"] = job
        _INGEST_STATE["last_started"] = time.time()
        _INGEST_STATE["last_error"] = None
        _INGEST_STATE["last_ok"] = None

    _append_ingest_ui_line(f"[doc-rag][INFO] --- старт задачи: {job} ---", ansi_strip=False)

    ok = False
    err = None
    try:
        await asyncio.to_thread(sync_runner)
        ok = True
    except Exception as exc:
        ok = False
        err = str(exc) or repr(exc)
        for ln in traceback.format_exc().strip().split("\n"):
            if ln.strip():
                _append_ingest_ui_line(ln.rstrip("\r"), ansi_strip=False)

    async with _INGEST_LOCK:
        _INGEST_STATE["running"] = False
        _INGEST_STATE["job"] = None
        _INGEST_STATE["last_finished"] = time.time()
        _INGEST_STATE["last_ok"] = ok
        _INGEST_STATE["last_error"] = err
        _reset_progress_state()
        if ok:
            _append_ingest_ui_line(f"[doc-rag][INFO] {job} завершился успешно.", ansi_strip=False)
        elif err:
            _append_ingest_ui_line(
                f"[doc-rag][ERROR] {job} завершился с ошибкой: {err}", ansi_strip=False
            )


async def _run_ingest_background() -> None:
    await _run_index_job_background("ingest", _sync_run_ingest_with_ui_logging)


async def _run_rebuild_background() -> None:
    await _run_index_job_background("rebuild", _sync_run_rebuild_with_ui_logging)


def _origin_allowed(origin: str | None) -> bool:
    """Basic DNS-rebinding mitigation (configurable).

    By default:
    - If Origin header is absent: allow (native clients often omit it)
    - If Origin is present: allow only if it matches DOC_RAG_ALLOWED_ORIGINS (comma-separated)
    """
    if not origin:
        return True
    allowed = [
        o.strip() for o in os.environ.get("DOC_RAG_ALLOWED_ORIGINS", "").split(",") if o.strip()
    ]
    if not allowed:
        # Be conservative: if Origin is present but allow-list is empty, deny.
        return False
    return origin in allowed


def _accepts_sse(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/event-stream" in accept.lower()


def _is_notification(req: dict[str, Any]) -> bool:
    # JSON-RPC: notifications are requests without an id, or with id=null
    return ("id" not in req) or (req.get("id") is None)


def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
    # For errors to requests, include id (can be null)
    payload["id"] = req_id
    return payload


def _log_line(line: str) -> None:
    """Emit one structured log record and mirror it into UI/file sinks.

    The structured logger handles stderr (and JSON when configured); the
    UI ring buffer and the optional DOC_RAG_HTTP_LOG file are kept on the
    legacy text format for backwards compatibility with the UI widget and
    the existing operator habits.
    """
    log.info(line)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    legacy = f"[doc-rag][http] {ts} {line}"
    _append_server_http_ui_line(legacy)
    log_path = os.environ.get("DOC_RAG_HTTP_LOG", "").strip()
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(legacy + "\n")
        except Exception:
            # Don't crash on logging issues
            pass


@dataclass
class _SseClient:
    queue: asyncio.Queue[str]
    created_ts: float


class _SseHub:
    """Very small SSE hub for server->client notifications."""

    def __init__(self) -> None:
        self._clients: list[_SseClient] = []
        self._lock = asyncio.Lock()

    async def add(self) -> _SseClient:
        client = _SseClient(queue=asyncio.Queue(), created_ts=time.time())
        async with self._lock:
            self._clients.append(client)
        return client

    async def remove(self, client: _SseClient) -> None:
        async with self._lock:
            self._clients = [c for c in self._clients if c is not client]

    async def broadcast_jsonrpc(self, msg: dict[str, Any]) -> None:
        payload = json.dumps(msg, ensure_ascii=False)
        # SSE "data:" lines must not contain raw newlines. JSON dumps doesn't, unless strings contain them.
        payload = payload.replace("\n", "\\n")
        frame = f"data: {payload}\n\n"
        async with self._lock:
            for client in list(self._clients):
                try:
                    client.queue.put_nowait(frame)
                except Exception:
                    # ignore slow/broken clients
                    pass


_sse_hub = _SseHub()


def _sse_frame(msg: dict[str, Any]) -> str:
    payload = json.dumps(msg, ensure_ascii=False).replace("\n", "\\n")
    return f"data: {payload}\n\n"


def _handle_one(req: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
    """Handle a single JSON-RPC message.

    Returns:
    - http_status
    - jsonrpc response object or None (for notifications/accepted responses)
    """
    method = req.get("method", "")
    req_id = req.get("id", None)

    # MCP handshake
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "doc-rag", "version": "2.4.0"},
            "capabilities": {"tools": {"listChanged": True}},
        }
        return 200, _ok(req_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "doc_search",
                "description": "Search the document knowledge base (semantic if FAISS+embeddings are available).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 6},
                    },
                    "required": ["query"],
                },
            }
        ]
        return 200, _ok(req_id, {"tools": tools})

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if name != "doc_search":
            _metrics.record_mcp_request(
                tool=name or "unknown", status="unknown_tool", duration_seconds=0.0
            )
            return 200, _ok(
                req_id,
                {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True},
            )

        # Lazy import to keep base install lightweight
        from doc_rag.server.search_tool import doc_search_tool

        started = time.time()
        try:
            content = doc_search_tool(arguments)
            _metrics.record_mcp_request("doc_search", "ok", time.time() - started)
            return 200, _ok(req_id, {"content": content, "isError": False})
        except Exception as exc:
            _metrics.record_mcp_request("doc_search", "error", time.time() - started)
            return 200, _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"doc_search failed: {exc}"}],
                    "isError": True,
                },
            )

    # Default: method not found
    if _is_notification(req):
        return 202, None
    return 200, _err(req_id, -32601, f"Method not found: {method}")


def _handle_jsonrpc(payload: Json) -> tuple[int, Json | None]:
    """Handle a JSON-RPC object or batch."""
    if isinstance(payload, list):
        responses: list[dict[str, Any]] = []
        status = 200
        for item in payload:
            if not isinstance(item, dict):
                continue
            st, resp = _handle_one(item)
            status = max(status, st)
            if resp is not None:
                responses.append(resp)
        if not responses:
            # Only notifications/responses -> 202
            return 202, None
        return 200, responses

    if not isinstance(payload, dict):
        return 400, _err(None, -32700, "Parse error: expected JSON object or array")

    status, resp = _handle_one(payload)
    if resp is None:
        return status, None
    return 200, resp


app = FastAPI(title="doc-rag MCP HTTP", version="2.4.0")


def _mount_ui_next(app: FastAPI) -> None:
    """Serve the Svelte UI bundle at `/ui-next/` when it's been built.

    v2.2.0 introduces a Svelte + Vite frontend under `ui/`. The build
    output lives at `ui/dist/`. If that directory exists we mount it;
    if not (e.g. minimal dev install that skipped `npm run build`),
    `/ui-next/` returns 404 and the legacy inline `/ui` keeps serving
    everyone.
    """
    from fastapi.staticfiles import StaticFiles

    candidates: list[str] = []
    # Editable install / dev: walk up from this file to find a sibling `ui/dist`.
    here = os.path.dirname(os.path.abspath(__file__))
    for n in range(1, 5):
        cand = os.path.join(os.path.abspath(os.path.join(here, *([".."] * n))), "ui", "dist")
        candidates.append(cand)
    # Production override.
    env_dir = (os.environ.get("DOC_RAG_UI_DIST") or "").strip()
    if env_dir:
        candidates.insert(0, env_dir)
    for d in candidates:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "index.html")):
            # `html=True` makes StaticFiles fall back to index.html for
            # unknown paths, which is what SPAs want for client-side routing.
            app.mount("/ui-next", StaticFiles(directory=d, html=True), name="ui_next")
            return


_mount_ui_next(app)


@app.middleware("http")
async def _http_log_mw(request: Request, call_next):
    # Honour an inbound request id if the client supplies one; otherwise
    # mint a fresh one. The id is propagated into every log record emitted
    # during this request via a ContextVar.
    inbound_rid = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
    rid = (inbound_rid or "").strip() or new_request_id()
    set_request_id(rid)

    start = time.time()
    response: Response | None = None
    try:
        client_host = getattr(getattr(request, "client", None), "host", None) or "unknown"
        # Don't rate-limit health checks or metrics scrapes (high-frequency probes).
        if request.url.path not in ("/health", "/health/live", "/health/ready", "/metrics"):
            if not await _rate_limit_allow(client_host):
                response = PlainTextResponse("Too Many Requests", status_code=429)
                response.headers["X-Request-ID"] = rid
                return response
        response = await call_next(request)
        if response is not None:
            response.headers["X-Request-ID"] = rid
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        code = getattr(response, "status_code", "ERR")
        path = request.url.path or ""
        # Avoid drowning the UI buffer with health checks and UI JSON polling.
        if path not in (
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/ui/status",
            "/ui/document-preview",
            "/api/v1/manifest",
            "/ui/config/raw",
            "/ui/config/save",
            "/ui/config/parsed",
            "/ui/config/patch",
            "/ui/env",
            "/ui/env/save",
            "/ui/restart",
        ):
            _log_line(f"{request.method} {request.url.path} -> {code} ({dur_ms}ms)")
        set_request_id(None)


def _readiness_state() -> dict[str, Any]:
    """Compute current readiness — manifest present, no background job running."""
    root = (os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir())
    manifest_abs = os.path.join(root, "build", "manifest.json")
    has_manifest = os.path.isfile(manifest_abs)
    job_running = bool(_INGEST_STATE.get("running"))
    job_name = _INGEST_STATE.get("job") if job_running else None
    ready = has_manifest and not job_running
    reasons: list[str] = []
    if not has_manifest:
        reasons.append("manifest_missing")
    if job_running:
        reasons.append(f"job_in_flight:{job_name}")
    return {
        "ready": ready,
        "has_manifest": has_manifest,
        "job_running": job_running,
        "job": job_name,
        "reasons": reasons,
    }


@app.get("/health/live")
async def health_live() -> JSONResponse:
    """Liveness probe — the process is up. Always 200 unless the event loop is wedged."""
    return JSONResponse({"status": "ok", "name": "doc-rag"})


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    """Readiness probe — 200 if the service can answer real queries, 503 otherwise.

    Ready when:
      - build/manifest.json exists (something has been ingested), and
      - no ingest or rebuild job is currently in flight.
    """
    state = _readiness_state()
    code = 200 if state["ready"] else 503
    return JSONResponse(
        {"status": "ready" if state["ready"] else "not_ready", **state}, status_code=code
    )


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus text exposition; 503 if the [metrics] extra is not installed."""
    if not _metrics.metrics_available():
        return PlainTextResponse(
            _metrics.render_text().decode("utf-8"),
            status_code=503,
            media_type="text/plain; charset=utf-8",
        )
    # Refresh the index-size gauge opportunistically each scrape.
    try:
        cat = indexed_catalog()
        n = int(cat.get("indexed_chunks_total", 0) or 0)
        _metrics.set_faiss_index_size(n)
    except Exception:
        pass
    body = _metrics.render_text()
    return PlainTextResponse(body.decode("utf-8"), media_type=_metrics.CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> JSONResponse:
    """Legacy combined health probe — kept for backwards compatibility.

    Returns 200 + ready/not_ready in the body so existing scrapers keep
    working but can be migrated to /health/live or /health/ready on their
    own schedule.
    """
    state = _readiness_state()
    return JSONResponse(
        {
            "status": "ok",
            "name": "doc-rag",
            "version": "2.4.0",
            "ready": state["ready"],
            "reasons": state["reasons"],
        }
    )


@app.get("/api/v1/manifest")
async def api_v1_manifest(request: Request) -> JSONResponse:
    """Machine-readable manifest for CI / reproducibility (same auth as MCP when DOC_RAG_API_KEY is set)."""
    if not _check_api_key(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    data = load_manifest_json()
    if not data:
        return JSONResponse({"error": "manifest_not_found"}, status_code=404)
    return JSONResponse(data)


@app.get("/ui/config/raw")
async def ui_config_raw(request: Request, key: str = "") -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    p = _config_path()
    try:
        yaml_text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(p), "yaml": yaml_text})


@app.post("/ui/config/save")
async def ui_config_save(
    request: Request, key: str = Form(""), content: str = Form("")
) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    ok, msg = _validate_root_yaml(content)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    try:
        _atomic_write_text(_config_path(), content)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(_config_path())})


@app.get("/ui/config/parsed")
async def ui_config_parsed(request: Request, key: str = "") -> JSONResponse:
    """Parsed view of config.yaml for the structured form editor.

    The form populates from this; comment-preserving writes go through
    `/ui/config/patch`. The raw text path (`/ui/config/raw`) still backs
    the Advanced tab."""
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    p = _config_path()
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    if not isinstance(data, dict):
        return JSONResponse(
            {"ok": False, "error": "Корень конфигурации не является объектом."},
            status_code=500,
        )
    return JSONResponse({"ok": True, "path": str(p), "config": data})


@app.post("/ui/config/validate")
async def ui_config_validate(
    request: Request, key: str = Form(""), updates: str = Form("")
) -> JSONResponse:
    """Validate config field updates without writing. Returns errors if any."""
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        parsed = json.loads(updates)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"updates не JSON: {exc}"}, status_code=400)
    if not isinstance(parsed, dict):
        return JSONResponse(
            {"ok": False, "error": "updates должен быть объектом."},
            status_code=400,
        )

    p = _config_path()
    try:
        import yaml as yaml_mod

        data = yaml_mod.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    errors: list[dict[str, str]] = []
    for dotted, value in parsed.items():
        parts = str(dotted).split(".")
        node = data
        try:
            for seg in parts[:-1]:
                node = node[seg]
            _ = node[parts[-1]]
        except (KeyError, TypeError):
            errors.append({"path": dotted, "error": "Неизвестный ключ конфигурации"})
            continue
        try:
            coerced = _coerce_like(node[parts[-1]], value)
        except (ValueError, TypeError) as exc:
            errors.append({"path": dotted, "error": str(exc)})
            continue
        val_err = _validate_field_value(dotted, coerced)
        if val_err:
            errors.append({"path": dotted, "error": val_err})

    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return JSONResponse({"ok": True})


def _coerce_like(existing: Any, value: Any) -> Any:
    """Coerce `value` to the type of `existing` so a round-trip patch keeps
    scalar types stable (the form sends typed JSON, but be defensive).

    bool is checked before int because `bool` is a subclass of `int`."""
    if isinstance(existing, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    if existing is None:
        return value
    return str(value)


_FIELD_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "parsing.pdf_backend": {"options": ["docling", "auto", "cascade"]},
    "parsing.docx_backend": {"options": ["python-docx", "docling"]},
    "parsing.min_chars_per_page": {"type": "int", "min": 0},
    "sectioning.min_heading_len": {"type": "int", "min": 1},
    "sectioning.max_heading_len": {"type": "int", "min": 1},
    "chunking.target_tokens": {"type": "int", "min": 1},
    "chunking.overlap_tokens": {"type": "int", "min": 0},
    "chunking.dedup_similarity_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "embeddings.batch_size": {"type": "int", "min": 1},
    "embeddings.device": {"options": ["cpu", "cuda"]},
    "index.backend": {"options": ["faiss"]},
    "index.metric": {"options": ["ip", "l2"]},
    "index.top_k": {"type": "int", "min": 1},
    "quality.fail_on_severity": {"options": ["never", "warn", "error"]},
}


def _validate_field_value(dotted: str, value: Any) -> str | None:
    """Return an error message if the value violates constraints, else None."""
    c = _FIELD_CONSTRAINTS.get(dotted)
    if c is None:
        return None
    if "options" in c and str(value) not in c["options"]:
        return f"Допустимые значения: {', '.join(c['options'])}"
    if c.get("type") == "int":
        try:
            n = int(value)
        except (ValueError, TypeError):
            return "Значение должно быть целым числом"
        if "min" in c and n < c["min"]:
            return f"Минимум: {c['min']}"
        if "max" in c and n > c["max"]:
            return f"Максимум: {c['max']}"
    if c.get("type") == "float":
        try:
            n = float(value)
        except (ValueError, TypeError):
            return "Значение должно быть числом"
        if "min" in c and n < c["min"]:
            return f"Минимум: {c['min']}"
        if "max" in c and n > c["max"]:
            return f"Максимум: {c['max']}"
    return None


@app.post("/ui/config/patch")
async def ui_config_patch(
    request: Request, key: str = Form(""), updates: str = Form("")
) -> JSONResponse:
    """Apply field-level updates to config.yaml while preserving comments.

    `updates` is a JSON object of dotted paths to values, e.g.
    `{"chunking.target_tokens": 512, "parsing.pdf_backend": "docling"}`.
    Only paths that already exist in the file are written; unknown paths
    are rejected so the form cannot silently invent keys."""
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    try:
        parsed = json.loads(updates)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"updates не JSON: {exc}"}, status_code=400)
    if not isinstance(parsed, dict) or not parsed:
        return JSONResponse(
            {"ok": False, "error": "updates должен быть непустым объектом."},
            status_code=400,
        )

    from ruamel.yaml import YAML

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    # ruamel renders Python None as an empty scalar by default, which would
    # silently rewrite untouched `key: null` lines to `key:`. Keep the
    # explicit `null` so a field patch only changes the field it touched.
    yaml_rt.representer.add_representer(
        type(None),
        lambda r, _d: r.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    p = _config_path()
    try:
        with p.open("r", encoding="utf-8") as fh:
            doc = yaml_rt.load(fh)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    for dotted, value in parsed.items():
        parts = str(dotted).split(".")
        node = doc
        try:
            for seg in parts[:-1]:
                node = node[seg]
            leaf = parts[-1]
            existing = node[leaf]
        except (KeyError, TypeError):
            return JSONResponse(
                {"ok": False, "error": f"Неизвестный ключ конфигурации: {dotted}"},
                status_code=400,
            )
        try:
            node[leaf] = _coerce_like(existing, value)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                {"ok": False, "error": f"Неверное значение для {dotted}: {exc}"},
                status_code=400,
            )
        val_err = _validate_field_value(dotted, node[leaf])
        if val_err:
            return JSONResponse(
                {"ok": False, "error": f"{dotted}: {val_err}"},
                status_code=400,
            )

    import io

    buf = io.StringIO()
    yaml_rt.dump(doc, buf)
    out = buf.getvalue()
    ok, msg = _validate_root_yaml(out)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    try:
        _atomic_write_text(p, out)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(p)})


# --- service runtime env editor (Stage 2) ---------------------------------
#
# Production runs under systemd, which loads /etc/default/doc-rag (root-owned,
# not writable by the `docrag` service user). So the UI manages a separate
# <root>/.env that `scripts/run_mcp_http.sh` sources at startup (overriding
# /etc/default). Changes apply on the next service restart.
#
# Each editable key carries a type for validation. DOC_RAG_API_KEY is a
# secret: surfaced as set/not-set only, never echoed and never written here.

_EDITABLE_ENV: dict[str, dict[str, Any]] = {
    "DOC_RAG_HTTP_HOST": {"type": "text"},
    "DOC_RAG_HTTP_PORT": {"type": "int"},
    "DOC_RAG_ALLOWED_ORIGINS": {"type": "text"},
    "DOC_RAG_HTTP_LOG": {"type": "text"},
    "DOC_RAG_UI_RESTART_ENABLED": {"type": "bool"},
    "DOC_RAG_UI_RESTART_CMD": {"type": "text"},
    "DOC_RAG_UI_MAX_UPLOAD_MB": {"type": "int"},
    "DOC_RAG_MAX_CONCURRENCY": {"type": "int"},
    "DOC_RAG_RATE_LIMIT_RPS": {"type": "float"},
    "DOC_RAG_RATE_LIMIT_BURST": {"type": "float"},
    "DOC_RAG_LOG_LEVEL": {"type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    "DOC_RAG_LOG_FORMAT": {"type": "select", "options": ["text", "json"]},
}
_SECRET_ENV = ("DOC_RAG_API_KEY",)


def _env_file_path() -> Path:
    override = (os.environ.get("DOC_RAG_ENV_FILE") or "").strip()
    if override:
        return Path(override)
    root = Path((os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir()))
    return root / ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs from a .env file, stripping surrounding quotes.
    Comments and blank lines are ignored. Tolerant of a missing file."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def _coerce_env_value(key: str, raw: str) -> tuple[bool, str, str]:
    """Validate+normalize a string value for an editable env key.
    Returns (ok, normalized, error)."""
    spec = _EDITABLE_ENV[key]
    t = spec["type"]
    s = str(raw).strip()
    if t == "int":
        try:
            return True, str(int(s)), ""
        except ValueError:
            return False, "", f"{key}: ожидается целое число"
    if t == "float":
        try:
            return True, repr(float(s)), ""
        except ValueError:
            return False, "", f"{key}: ожидается число"
    if t == "bool":
        truthy = s.lower() in ("1", "true", "yes", "on")
        return True, "1" if truthy else "0", ""
    if t == "select":
        if s not in spec["options"]:
            return False, "", f"{key}: допустимо одно из {spec['options']}"
        return True, s, ""
    return True, s, ""


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    """Merge managed keys into an existing .env, preserving unrelated lines
    and comments. Values are single-quoted so the file is safe to `source`
    from run_mcp_http.sh (DOC_RAG_UI_RESTART_CMD etc. contain spaces)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key in remaining:
            val = remaining.pop(key).replace("'", "'\\''")
            out.append(f"{key}='{val}'")
        else:
            out.append(line)
    for key, val in remaining.items():
        esc = val.replace("'", "'\\''")
        out.append(f"{key}='{esc}'")
    _atomic_write_text(path, "\n".join(out) + "\n")


@app.get("/ui/env")
async def ui_env_get(request: Request, key: str = "") -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    env_file = _env_file_path()
    persisted = _parse_env_file(env_file)
    fields = []
    for k, spec in _EDITABLE_ENV.items():
        # Prefer the value persisted in the UI-managed .env; fall back to the
        # effective process env (which includes /etc/default/doc-rag).
        if k in persisted:
            value, source = persisted[k], "file"
        elif os.environ.get(k) is not None:
            value, source = os.environ[k], "env"
        else:
            value, source = "", "default"
        fields.append(
            {
                "key": k,
                "type": spec["type"],
                "options": spec.get("options"),
                "value": value,
                "source": source,
            }
        )
    secrets = [
        {"key": k, "set": bool((os.environ.get(k) or "").strip() or persisted.get(k))}
        for k in _SECRET_ENV
    ]
    return JSONResponse({"ok": True, "path": str(env_file), "fields": fields, "secrets": secrets})


@app.post("/ui/env/save")
async def ui_env_save(
    request: Request, key: str = Form(""), updates: str = Form("")
) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    try:
        parsed = json.loads(updates)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"updates не JSON: {exc}"}, status_code=400)
    if not isinstance(parsed, dict) or not parsed:
        return JSONResponse(
            {"ok": False, "error": "updates должен быть непустым объектом."}, status_code=400
        )
    normalized: dict[str, str] = {}
    for k, v in parsed.items():
        if k in _SECRET_ENV:
            return JSONResponse(
                {"ok": False, "error": f"{k} нельзя задавать из UI."}, status_code=400
            )
        if k not in _EDITABLE_ENV:
            return JSONResponse(
                {"ok": False, "error": f"Неизвестный ключ env: {k}"}, status_code=400
            )
        ok, norm, err = _coerce_env_value(k, v)
        if not ok:
            return JSONResponse({"ok": False, "error": err}, status_code=400)
        normalized[k] = norm
    try:
        _write_env_file(_env_file_path(), normalized)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(_env_file_path())})


@app.post("/ui/restart")
async def ui_restart_service(
    request: Request, background_tasks: BackgroundTasks, key: str = Form("")
) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if not _ui_restart_allowed():
        return JSONResponse(
            {
                "ok": False,
                "error": "Задайте DOC_RAG_UI_RESTART_ENABLED=1 и DOC_RAG_UI_RESTART_CMD в окружении сервиса.",
            },
            status_code=403,
        )
    cmd = _ui_restart_cmd()
    if not cmd:
        return JSONResponse(
            {"ok": False, "error": "Пустой DOC_RAG_UI_RESTART_CMD."},
            status_code=400,
        )

    def _run() -> None:
        import subprocess

        subprocess.Popen(
            cmd,
            shell=True,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    background_tasks.add_task(_run)
    return JSONResponse({"ok": True, "message": "Команда перезапуска запущена в фоне."})


def _format_eta_seconds(seconds: float) -> str:
    """Compact human reading of an ETA. Round to keep noise out of the UI."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} мин {s:02d} с" if s else f"{m} мин"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин" if m else f"{h} ч"


def _compute_eta() -> tuple[float | None, str | None]:
    """Return (eta_seconds, eta_human) for the running job, if estimable."""
    if not _INGEST_STATE.get("running"):
        return None, None
    total = _INGEST_STATE.get("docs_total")
    done = int(_INGEST_STATE.get("docs_done") or 0)
    started = _INGEST_STATE.get("last_started")
    if not isinstance(total, int) or total <= 0 or not isinstance(started, (int, float)):
        return None, None
    remaining = total - done
    if remaining <= 0:
        return 0.0, _format_eta_seconds(0)
    # Need at least one completed doc to extrapolate. While the first
    # doc is parsing we leave ETA unknown rather than guess.
    if done <= 0:
        return None, None
    elapsed = max(0.0, time.time() - float(started))
    avg = elapsed / float(done)
    eta = avg * float(remaining)
    return eta, _format_eta_seconds(eta)


def _ui_status_payload() -> dict[str, Any]:
    out = dict(_INGEST_STATE)
    out["log_tail"] = _snapshot_ingest_ui_log_tail()
    out["http_log_tail"] = _snapshot_server_http_ui_log_tail()
    log_path = (os.environ.get("DOC_RAG_HTTP_LOG") or "").strip()
    out["http_log_file"] = log_path if log_path else None
    eta_s, eta_h = _compute_eta()
    out["eta_seconds"] = eta_s
    out["eta_human"] = eta_h
    try:
        out["indexed"] = indexed_catalog()
    except Exception as exc:
        out["indexed"] = {"error": str(exc), "documents": [], "document_count": 0}
    return out


def _ocr_badge_html(coverage: Any) -> str:
    """Inline OCR badge for the document table.

    Returns empty string when OCR did not fire on the document. When it
    did, returns a small `<span>` with a tooltip listing pages and the
    mean RapidOCR confidence."""
    if not isinstance(coverage, dict):
        return ""
    ocr = coverage.get("ocr")
    if not isinstance(ocr, dict) or not ocr.get("applied"):
        return ""
    pages_recognized = ocr.get("pages_recognized")
    confidence = ocr.get("confidence")
    title_parts: list[str] = ["OCR применился (RapidOCR через Docling)"]
    if isinstance(pages_recognized, int) and pages_recognized > 0:
        title_parts.append(f"страниц: {pages_recognized}")
    if isinstance(confidence, (int, float)):
        title_parts.append(f"средняя уверенность: {confidence:.2f}")
    title = " · ".join(title_parts)
    return f'<span class="ocr-badge" title="{html.escape(title, quote=True)}">OCR</span>'


def _indexed_documents_table_rows_html(
    catalog: dict[str, Any], max_rows: int = 300
) -> tuple[str, str]:
    """Return (tbody inner HTML, optional note HTML)."""
    docs = catalog.get("documents") if isinstance(catalog.get("documents"), list) else []
    note = ""
    shown = docs[:max_rows]
    if len(docs) > max_rows:
        note = f'<p class="muted">Показаны первые {max_rows} из {len(docs)} документов.</p>'
    rows: list[str] = []
    for i, d in enumerate(shown, 1):
        if not isinstance(d, dict):
            continue
        did_raw = str(d.get("doc_id") or "")
        did_attr = html.escape(did_raw, quote=True)
        bn_disp = html.escape(str(d.get("basename") or "—"), quote=False)
        did_cell = html.escape(did_raw or "—", quote=False)
        sf = html.escape(str(d.get("source_file") or ""), quote=True)
        cc = d.get("chunk_count")
        cc_s = html.escape(str(cc) if cc is not None else "—", quote=False)
        ey = d.get("edition_year")
        ey_s = html.escape(str(ey) if ey is not None else "—", quote=False)
        sh = d.get("sha256")
        sh_s = html.escape(
            str(sh)[:16] + "…" if sh and len(str(sh)) > 16 else (str(sh) if sh else "—"),
            quote=False,
        )
        ocr_badge = _ocr_badge_html(d.get("coverage"))
        rows.append(
            f'<tr data-doc-id="{did_attr}">'
            f'<td><input type="checkbox" class="row-check" data-doc-id="{did_attr}" /></td>'
            f"<td>{i}</td>"
            f'<td title="{sf}">{ocr_badge}<button type="button" class="doc-preview-btn" data-doc-id="{did_attr}">{bn_disp}</button></td>'
            f'<td class="doc-id">{did_cell}</td><td>{cc_s}</td><td>{ey_s}</td><td class="muted">{sh_s}</td>'
            f'<td><button type="button" class="row-delete-btn" data-doc-id="{did_attr}" data-name="{html.escape(str(d.get("basename") or ""), quote=True)}" title="Удалить документ">✕</button></td>'
            f"</tr>"
        )
    return ("\n".join(rows), note)


def _indexed_documents_summary_html(catalog: dict[str, Any]) -> str:
    if catalog.get("error"):
        return f'<p class="muted">Не удалось прочитать индекс: <code>{html.escape(str(catalog.get("error")), quote=False)}</code></p>'
    n = int(catalog.get("document_count") or 0)
    mg = catalog.get("manifest_generated_at_utc")
    mg_s = html.escape(str(mg), quote=False) if mg else "—"
    corp = catalog.get("corpus_content_sha256")
    corp_s = (
        html.escape(str(corp)[:20] + "…", quote=False)
        if corp and len(str(corp)) > 20
        else html.escape(str(corp), quote=False)
        if corp
        else "—"
    )
    pv = catalog.get("pipeline_version")
    pv_s = html.escape(str(pv), quote=False) if pv else "—"
    mp = catalog.get("manifest_present")
    lex = catalog.get("lexical_search_ready")
    sem = catalog.get("semantic_search_ready")
    chunks = catalog.get("chunks_jsonl_present")
    fidx = catalog.get("semantic_index_present")
    bits = [
        f"Записей в <code>manifest</code>: <strong>{n}</strong>.",
        f"Файл manifest: <strong>{'есть' if mp else 'нет'}</strong>.",
        f"<code>chunks.jsonl</code>: <strong>{'есть' if chunks else 'нет'}</strong>.",
        f"Векторный индекс (FAISS): <strong>{'есть' if fidx else 'нет'}</strong>.",
        f"Лексический поиск (doc_search): <strong>{'готов' if lex else 'не готов'}</strong>.",
        f"Семантический поиск: <strong>{'готов' if sem else 'не готов'}</strong>.",
        f"Время генерации manifest (UTC): <code>{mg_s}</code>.",
        f"Версия пайплайна: <code>{pv_s}</code>. Fingerprint корпуса (SHA): <code>{corp_s}</code>. API: <code>/api/v1/manifest</code>.",
    ]
    return '<p class="muted">' + " ".join(bits) + "</p>"


@app.get("/ui")
async def ui(request: Request, key: str = "") -> HTMLResponse:
    if not _ui_key_ok(request, key):
        return HTMLResponse(
            "<h3>Unauthorized</h3><p>Append <code>?key=...</code> to URL or send Authorization header.</p>",
            status_code=401,
        )

    qp = request.query_params
    upload_banner = ""
    try:
        us = int(qp.get("up_saved") or "0")
    except Exception:
        us = 0
    try:
        ue = int(qp.get("up_err") or "0")
    except Exception:
        ue = 0
    try:
        ud = int(qp.get("up_dup") or "0")
    except Exception:
        ud = 0
    um = (qp.get("up_msg") or "").strip()
    udm = (qp.get("up_dup_msg") or "").strip()
    if us > 0 or ue > 0 or ud > 0:
        parts = []
        if us > 0:
            parts.append(f"Загружено файлов: {us}.")
        if ud > 0:
            dup_detail = f" «{html.escape(udm)}»" if udm else ""
            parts.append(f"Пропущено дубликатов: {ud}.{dup_detail}")
        if ue > 0:
            err_detail = f" «{html.escape(um)}»" if um else ""
            parts.append(f"Ошибок: {ue}.{err_detail}")
        banner_cls = (
            "upload-banner has-errors"
            if ue > 0
            else ("upload-banner has-dups" if ud > 0 else "upload-banner")
        )
        upload_banner = f'<div class="{banner_cls}">{" ".join(parts)}</div>'

    state = dict(_INGEST_STATE)
    incoming = _sources_incoming_dir()
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        files = sorted([p.name for p in incoming.iterdir() if p.is_file()])[:200]
    except Exception:
        files = []

    key_q = f"?key={key}" if key else ""
    poll_ms_js = int(_INGEST_UI_LOG_POLL_MS)
    status_query_json = json.dumps(f"?key={key}" if key else "")
    log_pre_esc = html.escape("\n".join(_snapshot_ingest_ui_log_tail()), quote=False)
    http_log_pre_esc = html.escape("\n".join(_snapshot_server_http_ui_log_tail()), quote=False)
    http_log_file_note = ""
    hl = (os.environ.get("DOC_RAG_HTTP_LOG") or "").strip()
    if hl:
        escaped_hl = html.escape(hl, quote=True)
        http_log_file_note = f'<p class="muted">Тот же текст дублируется в файл: <code>{escaped_hl}</code> (<code>DOC_RAG_HTTP_LOG</code>).</p>'
    base = _public_base_url(request)
    mcp_url = f"{base}/mcp"
    try:
        ic0 = indexed_catalog()
    except Exception as exc:
        ic0 = {"error": str(exc), "documents": [], "document_count": 0}
    try:
        cfg_yaml_body = _config_path().read_text(encoding="utf-8")
    except Exception as exc:
        cfg_yaml_body = f"# не удалось прочитать конфиг: {exc}\n"
    cfg_path_esc = html.escape(str(_config_path()), quote=False)
    cfg_yaml_esc = html.escape(cfg_yaml_body, quote=False)
    restart_btn_on = _ui_restart_allowed() and bool(_ui_restart_cmd())
    restart_note_html = ""
    if not restart_btn_on:
        restart_note_html = (
            '<p class="muted">Перезапуск из UI по умолчанию выключен. Для кнопки задайте в окружении процесса '
            "<code>DOC_RAG_UI_RESTART_ENABLED=1</code> и <code>DOC_RAG_UI_RESTART_CMD</code> "
            "(например <code>sudo /bin/systemctl restart doc-rag-mcp</code> — см. sudoers).</p>"
        )
    restart_disabled_attr = " disabled" if not restart_btn_on else ""
    idx_summary_html = _indexed_documents_summary_html(ic0)
    idx_tbody_html, idx_cap_html = _indexed_documents_table_rows_html(ic0)
    body = f"""
  <html>
    <head>
      <meta charset="utf-8" />
      <title>doc-rag</title>
      <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
        code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
        .row {{ display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; min-width: 320px; }}
        #ingest-log-wrap {{ flex: 1 1 100%; min-width: min(920px, 100%); }}
        .mono-log-pre {{
          white-space: pre-wrap; word-break: break-word;
          max-height: min(60vh, 520px); overflow: auto;
          font-size: 12px; line-height: 1.35;
          background: #f9fafb; padding: 12px; border-radius: 10px;
          margin: 0; border: 1px solid #e5e7eb;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }}
        .muted {{ color:#6b7280; }}
        button {{ padding: 8px 12px; border-radius: 10px; border: 1px solid #111827; background:#111827; color:white; cursor:pointer; }}
        button.secondary {{ background:#374151; border-color:#374151; }}
        button:disabled {{ opacity:0.5; cursor:not-allowed; }}
        button.doc-preview-btn {{
          background: transparent; border: none; color: #2563eb; padding: 0; cursor: pointer;
          text-decoration: underline; font: inherit; text-align: left;
        }}
        button.doc-preview-btn:hover {{ color: #1d4ed8; }}
        .ocr-badge {{
          display: inline-block; margin-right: 6px; padding: 1px 6px;
          font-size: 11px; font-weight: 600; line-height: 1.3;
          color: #92400e; background: #fef3c7; border: 1px solid #fcd34d;
          border-radius: 6px; vertical-align: middle; cursor: help;
        }}
        .modal-root {{ display: none; position: fixed; inset: 0; z-index: 50; align-items: center; justify-content: center; }}
        .modal-root.is-open {{ display: flex; }}
        .modal-backdrop {{ position: absolute; inset: 0; background: rgba(15, 23, 42, 0.45); }}
        .modal-panel {{
          position: relative; z-index: 1; background: #fff; border-radius: 12px; padding: 20px 24px;
          max-width: min(560px, 92vw); max-height: min(72vh, 560px); overflow: auto;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }}
        .modal-panel button.modal-close {{
          position: absolute; top: 10px; right: 14px; padding: 4px 8px;
          border: none; background: transparent; font-size: 22px; line-height: 1;
          cursor: pointer; color: #64748b; border-radius: 8px;
        }}
        #doc-preview-heading {{ margin: 0 36px 8px 0; font-size: 1.1rem; }}
        #doc-preview-text {{ margin: 12px 0 0; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }}
        .upload-banner {{ margin:16px 0; padding:10px 12px; background:#ecfdf5; border:1px solid #6ee7b7; border-radius:10px; color:#065f46; }}
        .upload-banner.has-dups {{ background:#fefce8; border-color:#fde047; color:#713f12; }}
        .upload-banner.has-errors {{ background:#fef2f2; border-color:#fca5a5; color:#991b1b; }}
        .semantic-banner {{
          margin:16px 0; padding:12px 14px;
          background:#fff7ed; border:1px solid #fdba74; border-radius:10px; color:#7c2d12;
          display:flex; align-items:center; gap:8px; flex-wrap:wrap;
        }}
        .semantic-banner button {{ background:#c2410c; border-color:#c2410c; }}
        .semantic-banner button:hover {{ background:#9a3412; border-color:#9a3412; }}
        table.idx {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        table.idx th, table.idx td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
        table.idx th {{ background: #f9fafb; font-weight: 600; }}
        table.idx td.doc-id {{ font-size: 11px; color: #6b7280; word-break: break-all; max-width: min(280px, 28vw); }}
        .idx-toolbar {{ margin: 8px 0; display:flex; gap:10px; align-items:center; }}
        button.row-delete-btn {{
          background: transparent; border: 1px solid #fecaca; color: #b91c1c;
          padding: 2px 8px; border-radius: 6px; font-size: 14px; line-height: 1;
          cursor: pointer;
        }}
        button.row-delete-btn:hover {{ background:#fee2e2; }}
        button.danger {{ background:#b91c1c; border-color:#b91c1c; }}
        button.danger:hover {{ background:#991b1b; border-color:#991b1b; }}
        .danger-zone {{ border-color:#fecaca; background:#fff7f7; }}
        .danger-actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }}
        textarea.mono-cfg {{
          width: 100%; min-height: 420px; font-size: 12px; line-height: 1.4;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          padding: 10px 12px; border-radius: 10px; border: 1px solid #e5e7eb; box-sizing: border-box;
        }}
      </style>
    </head>
    <body>
      <h2>doc-rag — управление индексом</h2>
      <p class="muted">LAN UI. MCP endpoint: <code>/mcp</code></p>

      <div id="semantic-banner" class="semantic-banner" style="display:{("none" if ic0.get("semantic_search_ready") else "flex")};">
        <strong>Семантический поиск недоступен.</strong>
        FAISS-индекс отсутствует или повреждён — клиенты получают только лексические результаты.
        <form action="/ui/rebuild{key_q}" method="post" style="display:inline-block; margin-left:8px;">
          <input type="hidden" name="key" value="{key}"/>
          <button type="submit" id="semantic-banner-rebuild-btn"{" disabled" if state.get("running") else ""}>Запустить rebuild</button>
        </form>
      </div>

      {upload_banner}

      <div class="row">
        <div class="card">
          <h3>Загрузка документов</h3>
          <form action="/ui/upload{key_q}" method="post" enctype="multipart/form-data">
            <input type="hidden" name="key" value="{key}"/>
            <input type="file" name="files" accept=".pdf,.docx,.doc,.md,.txt,.PDF,.DOCX,.DOC,.MD,.TXT" multiple required />
            <div style="height: 12px;"></div>
            <button type="submit" id="ui-btn-upload" {"disabled" if state.get("running") else ""}>Загрузить в sources/incoming</button>
          </form>
          <p class="muted">PDF, DOCX, DOC, MD, TXT; можно выбрать несколько файлов сразу (лимит: env <code>DOC_RAG_UI_MAX_UPLOAD_FILES</code>, по умолчанию 48).</p>
        </div>

        <div class="card">
          <h3>Ingest / rebuild</h3>
          <form action="/ui/ingest{key_q}" method="post">
            <input type="hidden" name="key" value="{key}"/>
            <button type="submit" id="ui-btn-ingest" {"disabled" if state.get("running") else ""}>Запустить ingest</button>
          </form>
          <div style="height:12px;"></div>
          <form action="/ui/rebuild{key_q}" method="post">
            <input type="hidden" name="key" value="{key}"/>
            <button type="submit" id="ui-btn-rebuild" class="secondary" onclick="return confirm('Полный rebuild очистит build/docs markdown и chunks, затем пересканирует archived и incoming. Продолжить?');" {"disabled" if state.get("running") else ""}>Rebuild индекса</button>
          </form>
          <p class="muted">Фоновая задача: <code id="ingest-running-badge">{(state.get("job") or ("busy")) if state.get("running") else "idle"}</code></p>
          <p class="muted" id="ingest-progress-line" style="display:none">
            Сейчас: <code id="ingest-current-doc">—</code>
            <span id="ingest-progress-counts" class="muted"></span>
            <span id="ingest-eta" class="muted"></span>
          </p>
          <p class="muted">Последний результат: <code id="ingest-last-ok">{state.get("last_ok")}</code></p>
          <p class="muted">Ошибка: <code id="ingest-last-error">{(state.get("last_error") or "-")}</code></p>
          <p><a href="/ui/status{key_q}">/ui/status</a></p>
          <p><a href="/ui/mcp/cursor.json{key_q}">Скачать MCP config для Cursor</a></p>
          <p><a href="/ui/mcp/vscode.json{key_q}">Скачать MCP config для VSCode</a></p>
          <p class="muted">MCP URL: <code>{mcp_url}</code></p>
        </div>

        <div class="card">
          <h3>Incoming ({len(files)})</h3>
          <div class="muted" style="max-height: 240px; overflow:auto;">
            {"<br/>".join(files) if files else "<span class='muted'>empty</span>"}
          </div>
        </div>
      </div>

      <div class="row">
        <div class="card" style="flex: 1 1 100%; min-width: min(960px, 100%);">
          <h3>Конфигурация</h3>
          <p class="muted">Файл: <code>{cfg_path_esc}</code>. После правок часть параметров подхватывается при следующем запросе; для смены порта или переменных systemd может понадобиться перезапуск.</p>
          <textarea id="cfg-yaml" class="mono-cfg" name="content" spellcheck="false">{cfg_yaml_esc}</textarea>
          <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
            <button type="button" id="cfg-save-btn">Сохранить</button>
            <button type="button" id="srv-restart-btn" class="secondary"{restart_disabled_attr}>Перезапустить сервис</button>
          </div>
          <p id="cfg-save-msg" class="muted" style="margin-top:8px;"></p>
          {restart_note_html}
        </div>
      </div>

      <div class="row">
        <div class="card" style="flex: 1 1 100%; min-width: min(960px, 100%);">
          <h3>Проиндексированные документы</h3>
          <div id="indexed-summary">{idx_summary_html}</div>
          <div id="indexed-cap-note">{idx_cap_html}</div>
          <div class="idx-toolbar">
            <button type="button" id="idx-bulk-delete-btn" class="secondary" disabled>Удалить выбранные (<span id="idx-bulk-count">0</span>)</button>
            <span class="muted" id="idx-bulk-msg"></span>
          </div>
          <div style="overflow:auto; max-height: min(55vh, 520px); border: 1px solid #e5e7eb; border-radius: 10px;">
            <table class="idx" id="indexed-table">
              <thead>
                <tr>
                  <th style="width:32px;"><input type="checkbox" id="idx-select-all" title="Выбрать все" /></th>
                  <th style="width:36px;">#</th>
                  <th>Файл</th>
                  <th>doc_id</th>
                  <th style="width:88px;">Чанков</th>
                  <th style="width:72px;">Год</th>
                  <th style="width:120px;">SHA256</th>
                  <th style="width:40px;"></th>
                </tr>
              </thead>
              <tbody id="indexed-tbody">{idx_tbody_html}</tbody>
            </table>
          </div>
          <p class="muted" style="margin-top:10px;">Список берётся из <code>build/manifest.json</code> после ingest/rebuild. После завершения задачи таблица обновится автоматически.</p>
        </div>
      </div>

      <div class="row">
        <div class="card danger-zone" style="flex: 1 1 100%; min-width: min(960px, 100%);">
          <h3>Опасная зона</h3>
          <p class="muted">Эти операции необратимы. Дождитесь завершения текущего ingest/rebuild.</p>
          <div class="danger-actions">
            <button type="button" id="dz-wipe-btn" class="danger">Удалить всё (sources/archived + build/)</button>
            <button type="button" id="dz-orphans-btn" class="secondary">Удалить осиротевшие артефакты</button>
            <button type="button" id="dz-incoming-btn" class="secondary">Очистить sources/incoming</button>
          </div>
          <p id="dz-msg" class="muted" style="margin-top:10px;"></p>
        </div>
      </div>

      <div class="card" id="http-log-wrap" style="margin-top:24px;">
        <h3>Журнал HTTP-сервера (запросы)</h3>
        <p class="muted">
          Кольцевой буфер в памяти — без <code>journalctl</code>. Запросы к <code>/health</code>, <code>/ui/status</code> и <code>/ui/document-preview</code> не пишутся (чтобы не забивать лог опросами).
        </p>
        {http_log_file_note}
        <pre id="http-log-pre" class="mono-log-pre">{http_log_pre_esc}</pre>
      </div>

      <div class="card" id="ingest-log-wrap" style="margin-top:24px;">
        <h3>Лог ingest / rebuild (stderr + ошибки пайплайна)</h3>
        <p class="muted">
          Авто‑обновление каждые {poll_ms_js} ms через <code>/ui/status</code>.
          При длинном прогрессе промежуточные строки с <code>\r</code> схлопываются.
        </p>
        <pre id="ingest-log-pre" class="mono-log-pre">{log_pre_esc}</pre>
      </div>

      <div id="doc-preview-modal" class="modal-root" aria-hidden="true">
        <div class="modal-backdrop" id="doc-preview-backdrop"></div>
        <div class="modal-panel" role="dialog" aria-labelledby="doc-preview-heading">
          <button type="button" class="modal-close" id="doc-preview-close" aria-label="Закрыть">&times;</button>
          <h4 id="doc-preview-heading"></h4>
          <p class="muted" id="doc-preview-meta"></p>
          <p id="doc-preview-text"></p>
        </div>
      </div>

      <script>
      (function () {{
        var pollMs = {poll_ms_js};
        var statusPath = "/ui/status" + {status_query_json};
        var uiKey = {json.dumps(key)};
        var elIng = document.getElementById("ingest-log-pre");
        var elHttp = document.getElementById("http-log-pre");
        function esc(s) {{
          if (s === null || s === undefined) return "";
          return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
        }}
        function renderIndexed(idx) {{
          var sumEl = document.getElementById("indexed-summary");
          var capEl = document.getElementById("indexed-cap-note");
          var tbEl = document.getElementById("indexed-tbody");
          if (!sumEl || !tbEl) return;
          if (!idx || idx.error) {{
            sumEl.innerHTML = idx && idx.error ? '<p class="muted">Не удалось прочитать индекс: <code>' + esc(idx.error) + '</code></p>' : '<p class="muted">Нет данных.</p>';
            tbEl.innerHTML = "";
            if (capEl) capEl.innerHTML = "";
            return;
          }}
          var n = idx.document_count || 0;
          var mg = idx.manifest_generated_at_utc ? '<code>' + esc(String(idx.manifest_generated_at_utc)) + '</code>' : '<code>—</code>';
          var pv = idx.pipeline_version ? '<code>' + esc(String(idx.pipeline_version)) + '</code>' : '<code>—</code>';
          var corp = idx.corpus_content_sha256 ? String(idx.corpus_content_sha256) : '';
          var corpDisp = corp ? '<code>' + esc(corp.length > 20 ? corp.slice(0, 20) + '…' : corp) + '</code>' : '<code>—</code>';
          var bits = [
            'Записей в <code>manifest</code>: <strong>' + n + '</strong>.',
            'Файл manifest: <strong>' + (idx.manifest_present ? 'есть' : 'нет') + '</strong>.',
            '<code>chunks.jsonl</code>: <strong>' + (idx.chunks_jsonl_present ? 'есть' : 'нет') + '</strong>.',
            'Векторный индекс (FAISS): <strong>' + (idx.semantic_index_present ? 'есть' : 'нет') + '</strong>.',
            'Лексический поиск (doc_search): <strong>' + (idx.lexical_search_ready ? 'готов' : 'не готов') + '</strong>.',
            'Семантический поиск: <strong>' + (idx.semantic_search_ready ? 'готов' : 'не готов') + '</strong>.',
            'Время генерации manifest (UTC): ' + mg + '.',
            'Версия пайплайна: ' + pv + '. Fingerprint корпуса (SHA): ' + corpDisp + '. API: <code>/api/v1/manifest</code>.'
          ];
          sumEl.innerHTML = '<p class="muted">' + bits.join(' ') + '</p>';
          var docs = idx.documents || [];
          var maxR = 300;
          var slice = docs.slice(0, maxR);
          if (capEl) {{
            capEl.innerHTML = docs.length > maxR ? '<p class="muted">Показаны первые ' + maxR + ' из ' + docs.length + ' документов.</p>' : '';
          }}
          function ocrBadge(cov) {{
            if (!cov || !cov.ocr || !cov.ocr.applied) return '';
            var parts = ['OCR применился (RapidOCR через Docling)'];
            var pr = cov.ocr.pages_recognized;
            if (typeof pr === 'number' && pr > 0) parts.push('страниц: ' + pr);
            var conf = cov.ocr.confidence;
            if (typeof conf === 'number') parts.push('средняя уверенность: ' + conf.toFixed(2));
            return '<span class="ocr-badge" title="' + esc(parts.join(' · ')) + '">OCR</span>';
          }}
          tbEl.innerHTML = slice.map(function (d, i) {{
            var bn = esc(d.basename || '—');
            var sf = esc(d.source_file || '');
            var didRaw = d.doc_id != null ? String(d.doc_id) : '';
            var didAttr = esc(didRaw);
            var didCell = esc(didRaw || '—');
            var nameCell = didRaw
              ? '<button type="button" class="doc-preview-btn" data-doc-id="' + didAttr + '">' + bn + '</button>'
              : bn;
            var cc = d.chunk_count != null ? esc(String(d.chunk_count)) : '—';
            var yr = d.edition_year != null && d.edition_year !== undefined ? esc(String(d.edition_year)) : '—';
            var sh = d.sha256 ? String(d.sha256) : '';
            var shDisp = sh.length > 16 ? esc(sh.slice(0, 16)) + '…' : esc(sh || '—');
            var chk = didRaw ? '<input type="checkbox" class="row-check" data-doc-id="' + didAttr + '" />' : '';
            var del = didRaw ? '<button type="button" class="row-delete-btn" data-doc-id="' + didAttr + '" data-name="' + esc(d.basename || '') + '" title="Удалить документ">✕</button>' : '';
            var badge = ocrBadge(d.coverage);
            return '<tr data-doc-id="' + didAttr + '"><td>' + chk + '</td><td>' + (i + 1) + '</td><td title="' + sf + '">' + badge + nameCell + '</td><td class="doc-id">' + didCell + '</td><td>' + cc + '</td><td>' + yr + '</td><td class="muted">' + shDisp + '</td><td>' + del + '</td></tr>';
          }}).join('');
          updateBulkDeleteState();
        }}
        function poll() {{
          fetch(statusPath, {{credentials: "same-origin"}})
            .then(function (r) {{ return r.json(); }})
            .then(function (j) {{
              if (!j || j.error === "unauthorized") return;
              if (elIng) {{
                var lines = j.log_tail || [];
                elIng.textContent = lines.join("\\n");
              }}
              if (elHttp) {{
                var hlines = j.http_log_tail || [];
                elHttp.textContent = hlines.join("\\n");
              }}
              var b = document.getElementById("ingest-running-badge");
              var job = j.job || "";
              if (b) b.textContent = j.running ? (job || "busy") : "idle";
              var busy = !!j.running;
              var prog = document.getElementById("ingest-progress-line");
              var progCur = document.getElementById("ingest-current-doc");
              var progCnt = document.getElementById("ingest-progress-counts");
              var progEta = document.getElementById("ingest-eta");
              if (prog) {{
                if (busy) {{
                  prog.style.display = "";
                  if (progCur) progCur.textContent = j.current_doc || "(ожидание…)";
                  if (progCnt) {{
                    var d = (j.docs_done != null) ? j.docs_done : 0;
                    var t = (j.docs_total != null) ? j.docs_total : "?";
                    progCnt.textContent = " · " + d + "/" + t;
                  }}
                  if (progEta) {{
                    progEta.textContent = j.eta_human ? (" · осталось ~" + j.eta_human) : "";
                  }}
                }} else {{
                  prog.style.display = "none";
                }}
              }}
              var bu = document.getElementById("ui-btn-upload");
              var bi = document.getElementById("ui-btn-ingest");
              var br = document.getElementById("ui-btn-rebuild");
              if (bu) bu.disabled = busy;
              if (bi) bi.disabled = busy;
              if (br) br.disabled = busy;
              var lk = document.getElementById("ingest-last-ok");
              if (lk) lk.textContent = (j.last_ok === null || j.last_ok === undefined) ? "-" : String(j.last_ok);
              var le = document.getElementById("ingest-last-error");
              if (le) le.textContent = (j.last_error != null && String(j.last_error).length) ? String(j.last_error) : "-";
              renderIndexed(j.indexed);
              var sb = document.getElementById("semantic-banner");
              if (sb) {{
                var ready = j.indexed && j.indexed.semantic_search_ready;
                sb.style.display = ready ? "none" : "flex";
              }}
              var sbBtn = document.getElementById("semantic-banner-rebuild-btn");
              if (sbBtn) sbBtn.disabled = busy;
              if (j.running && elIng) {{
                elIng.scrollTop = elIng.scrollHeight;
              }}
            }})
            .catch(function () {{}});
        }}
        var modal = document.getElementById("doc-preview-modal");
        var elPrevHead = document.getElementById("doc-preview-heading");
        var elPrevMeta = document.getElementById("doc-preview-meta");
        var elPrevText = document.getElementById("doc-preview-text");
        function closeDocPreview() {{
          if (!modal) return;
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
        }}
        function openDocPreview(did) {{
          if (!modal || !elPrevHead || !elPrevText) return;
          elPrevHead.textContent = "Загрузка…";
          if (elPrevMeta) elPrevMeta.textContent = "";
          elPrevText.textContent = "";
          modal.classList.add("is-open");
          modal.setAttribute("aria-hidden", "false");
          var u = "/ui/document-preview?doc_id=" + encodeURIComponent(did);
          if (typeof uiKey === "string" && uiKey.length) u += "&key=" + encodeURIComponent(uiKey);
          fetch(u, {{ credentials: "same-origin" }})
            .then(function (r) {{ return r.json(); }})
            .then(function (j) {{
              if (j && j.error === "unauthorized") {{
                elPrevHead.textContent = "Доступ запрещён";
                elPrevText.textContent = "";
                return;
              }}
              if (!j || !j.ok) {{
                elPrevHead.textContent = "Документ";
                elPrevText.textContent = j && j.error ? String(j.error) : "Не удалось загрузить аннотацию.";
                return;
              }}
              var t = j.title && String(j.title).trim() ? String(j.title).trim() : (j.basename || did);
              elPrevHead.textContent = t;
              if (elPrevMeta) {{
                var bits = [];
                if (j.source_file) bits.push(esc(String(j.source_file)));
                bits.push("<code>" + esc(String(j.doc_id || did)) + "</code>");
                if (j.edition_year != null && j.edition_year !== undefined) bits.push("год редакции: " + esc(String(j.edition_year)));
                elPrevMeta.innerHTML = bits.join(" · ");
              }}
              elPrevText.textContent = j.preview || "";
            }})
            .catch(function () {{
              elPrevHead.textContent = "Ошибка сети";
              elPrevText.textContent = "";
            }});
        }}
        document.addEventListener("click", function (ev) {{
          var btn = ev.target && ev.target.closest ? ev.target.closest(".doc-preview-btn") : null;
          if (!btn) return;
          var did = btn.getAttribute("data-doc-id");
          if (!did) return;
          ev.preventDefault();
          openDocPreview(did);
        }});
        var bd = document.getElementById("doc-preview-backdrop");
        var bx = document.getElementById("doc-preview-close");
        if (bd) bd.addEventListener("click", closeDocPreview);
        if (bx) bx.addEventListener("click", closeDocPreview);
        var keyQ = {json.dumps(key_q)};
        (function () {{
          var saveBtn = document.getElementById("cfg-save-btn");
          var restartBtn = document.getElementById("srv-restart-btn");
          var msgEl = document.getElementById("cfg-save-msg");
          var ta = document.getElementById("cfg-yaml");
          if (saveBtn && ta) saveBtn.addEventListener("click", function () {{
            if (msgEl) msgEl.textContent = "";
            var fd = new FormData();
            if (uiKey) fd.append("key", uiKey);
            fd.append("content", ta.value);
            fetch("/ui/config/save" + keyQ, {{ method: "POST", body: fd, credentials: "same-origin" }})
              .then(function (r) {{ return r.json(); }})
              .then(function (j) {{
                if (msgEl) msgEl.textContent = j && j.ok ? "Сохранено." : ("Ошибка: " + (j && j.error ? j.error : "?"));
              }})
              .catch(function () {{ if (msgEl) msgEl.textContent = "Ошибка сети."; }});
          }});
          if (restartBtn) restartBtn.addEventListener("click", function () {{
            if (!confirm("Запустить настроенную команду перезапуска сервиса?")) return;
            if (msgEl) msgEl.textContent = "";
            var fd = new FormData();
            if (uiKey) fd.append("key", uiKey);
            fetch("/ui/restart" + keyQ, {{ method: "POST", body: fd, credentials: "same-origin" }})
              .then(function (r) {{ return r.json(); }})
              .then(function (j) {{
                if (msgEl) msgEl.textContent = j && j.ok ? (j.message || "OK") : ("Ошибка: " + (j && j.error ? j.error : "?"));
              }})
              .catch(function () {{ if (msgEl) msgEl.textContent = "Ошибка сети."; }});
          }});
        }})();

        function getCheckedDocIds() {{
          var nodes = document.querySelectorAll(".row-check:checked");
          var ids = [];
          for (var i = 0; i < nodes.length; i++) {{
            var v = nodes[i].getAttribute("data-doc-id");
            if (v) ids.push(v);
          }}
          return ids;
        }}
        function updateBulkDeleteState() {{
          var ids = getCheckedDocIds();
          var btn = document.getElementById("idx-bulk-delete-btn");
          var cnt = document.getElementById("idx-bulk-count");
          var sa = document.getElementById("idx-select-all");
          if (cnt) cnt.textContent = String(ids.length);
          if (btn) btn.disabled = ids.length === 0;
          if (sa) {{
            var rows = document.querySelectorAll(".row-check");
            sa.checked = rows.length > 0 && ids.length === rows.length;
            sa.indeterminate = ids.length > 0 && ids.length < rows.length;
          }}
        }}
        document.addEventListener("change", function (ev) {{
          var t = ev.target;
          if (!t) return;
          if (t.id === "idx-select-all") {{
            var rows = document.querySelectorAll(".row-check");
            for (var i = 0; i < rows.length; i++) rows[i].checked = t.checked;
            updateBulkDeleteState();
            return;
          }}
          if (t.classList && t.classList.contains("row-check")) {{
            updateBulkDeleteState();
          }}
        }});

        function postForm(path, fields) {{
          var fd = new FormData();
          if (uiKey) fd.append("key", uiKey);
          for (var k in fields) {{ if (Object.prototype.hasOwnProperty.call(fields, k)) fd.append(k, fields[k]); }}
          return fetch(path + keyQ, {{ method: "POST", body: fd, credentials: "same-origin" }})
            .then(function (r) {{ return r.json().then(function (j) {{ return {{ status: r.status, body: j }}; }}); }});
        }}
        function flashBulkMsg(text) {{
          var el = document.getElementById("idx-bulk-msg");
          if (el) el.textContent = text;
        }}
        function flashDangerMsg(text) {{
          var el = document.getElementById("dz-msg");
          if (el) el.textContent = text;
        }}

        document.addEventListener("click", function (ev) {{
          var btn = ev.target && ev.target.closest ? ev.target.closest(".row-delete-btn") : null;
          if (!btn) return;
          var did = btn.getAttribute("data-doc-id");
          var nm = btn.getAttribute("data-name") || did;
          if (!did) return;
          ev.preventDefault();
          if (!confirm("Удалить документ «" + nm + "» из базы знаний?\\nЭта операция необратима.")) return;
          flashBulkMsg("Удаляем «" + nm + "»…");
          postForm("/ui/delete", {{ doc_ids: did }}).then(function (resp) {{
            if (resp.body && resp.body.ok) {{
              flashBulkMsg("Удалено: " + (resp.body.deleted || 0) + " док., чанков: " + (resp.body.removed_chunks || 0));
              poll();
            }} else {{
              flashBulkMsg("Ошибка: " + (resp.body && resp.body.error ? resp.body.error : ("HTTP " + resp.status)));
            }}
          }}).catch(function () {{ flashBulkMsg("Ошибка сети."); }});
        }});

        var bulkBtn = document.getElementById("idx-bulk-delete-btn");
        if (bulkBtn) bulkBtn.addEventListener("click", function () {{
          var ids = getCheckedDocIds();
          if (!ids.length) return;
          if (!confirm("Удалить " + ids.length + " документ(ов) из базы знаний?\\nЭта операция необратима.")) return;
          flashBulkMsg("Удаляем " + ids.length + " документ(ов)…");
          postForm("/ui/delete", {{ doc_ids: ids.join(",") }}).then(function (resp) {{
            if (resp.body && resp.body.ok) {{
              flashBulkMsg("Удалено: " + (resp.body.deleted || 0) + " док., чанков: " + (resp.body.removed_chunks || 0) + ", векторов: " + ((resp.body.index && resp.body.index.removed_vectors) || 0));
              poll();
            }} else {{
              flashBulkMsg("Ошибка: " + (resp.body && resp.body.error ? resp.body.error : ("HTTP " + resp.status)));
            }}
          }}).catch(function () {{ flashBulkMsg("Ошибка сети."); }});
        }});

        var wipeBtn = document.getElementById("dz-wipe-btn");
        if (wipeBtn) wipeBtn.addEventListener("click", function () {{
          var ans = prompt("Это удалит ВСЕ документы, чанки, индекс и архив. Введите DELETE для подтверждения:");
          if (ans !== "DELETE") {{
            flashDangerMsg(ans === null ? "" : "Отменено (нужно ввести строго DELETE).");
            return;
          }}
          flashDangerMsg("Удаление всего…");
          postForm("/ui/wipe", {{ confirm: "DELETE" }}).then(function (resp) {{
            if (resp.body && resp.body.ok) {{
              flashDangerMsg("Готово. Удалено записей: " + (resp.body.removed_entries || 0));
              poll();
            }} else {{
              flashDangerMsg("Ошибка: " + (resp.body && resp.body.error ? resp.body.error : ("HTTP " + resp.status)));
            }}
          }}).catch(function () {{ flashDangerMsg("Ошибка сети."); }});
        }});

        var orphBtn = document.getElementById("dz-orphans-btn");
        if (orphBtn) orphBtn.addEventListener("click", function () {{
          if (!confirm("Удалить осиротевшие файлы (md/чанки/векторы без записи в manifest)?")) return;
          flashDangerMsg("Удаление осиротевших…");
          postForm("/ui/clean-orphans", {{}}).then(function (resp) {{
            if (resp.body && resp.body.ok) {{
              flashDangerMsg("Удалено md: " + (resp.body.orphan_md_removed || 0) + ", чанков: " + (resp.body.orphan_chunks_removed || 0) + ", векторов: " + ((resp.body.index && resp.body.index.removed_vectors) || 0));
              poll();
            }} else {{
              flashDangerMsg("Ошибка: " + (resp.body && resp.body.error ? resp.body.error : ("HTTP " + resp.status)));
            }}
          }}).catch(function () {{ flashDangerMsg("Ошибка сети."); }});
        }});

        var incBtn = document.getElementById("dz-incoming-btn");
        if (incBtn) incBtn.addEventListener("click", function () {{
          if (!confirm("Удалить все файлы из sources/incoming/?")) return;
          flashDangerMsg("Очистка incoming…");
          postForm("/ui/clear-incoming", {{}}).then(function (resp) {{
            if (resp.body && resp.body.ok) {{
              flashDangerMsg("Удалено файлов: " + (resp.body.removed || 0));
              poll();
            }} else {{
              flashDangerMsg("Ошибка: " + (resp.body && resp.body.error ? resp.body.error : ("HTTP " + resp.status)));
            }}
          }}).catch(function () {{ flashDangerMsg("Ошибка сети."); }});
        }});

        updateBulkDeleteState();
        poll();
        setInterval(poll, pollMs);
      }})();
      </script>

    </body>
  </html>
  """
    return HTMLResponse(body)


@app.get("/ui/mcp/cursor.json")
async def ui_mcp_cursor(request: Request, key: str = "") -> Response:
    if not _ui_key_ok(request, key):
        return PlainTextResponse("Unauthorized", status_code=401)
    base = _public_base_url(request)
    payload = _mcp_config_payload("doc-rag", f"{base}/mcp", key)
    return _download_json(payload, "cursor-mcp-doc-rag.json")


@app.get("/ui/mcp/vscode.json")
async def ui_mcp_vscode(request: Request, key: str = "") -> Response:
    if not _ui_key_ok(request, key):
        return PlainTextResponse("Unauthorized", status_code=401)
    base = _public_base_url(request)
    payload = _mcp_config_payload("doc-rag", f"{base}/mcp", key)
    return _download_json(payload, "vscode-mcp-doc-rag.json")


@app.get("/ui/status")
async def ui_status(request: Request, key: str = "") -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse(_ui_status_payload())


@app.get("/ui/document-preview")
async def ui_document_preview(request: Request, doc_id: str = "", key: str = "") -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse(document_preview(doc_id))


@app.post("/ui/upload")
async def ui_upload(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    key: str = Form(""),
) -> Response:
    if not _ui_key_ok(request, key):
        return PlainTextResponse("Unauthorized", status_code=401)

    incoming = _sources_incoming_dir()
    incoming.mkdir(parents=True, exist_ok=True)

    lim = _ui_max_upload_files()
    if len(files) > lim:
        return PlainTextResponse(
            f"Слишком много файлов за один запрос: {len(files)} > лимит {lim} (DOC_RAG_UI_MAX_UPLOAD_FILES).",
            status_code=400,
        )

    errors: list[str] = []
    dups: list[str] = []
    saved = 0
    for uf in files:
        is_dup, msg = await _save_upload_to_incoming(uf, incoming)
        if msg is None:
            saved += 1
        elif is_dup:
            dups.append(msg)
        else:
            errors.append(msg)

    qdict: dict[str, str] = {}
    k = (key or "").strip()
    if k:
        qdict["key"] = k
    if saved > 0:
        qdict["up_saved"] = str(saved)
    if errors:
        qdict["up_err"] = str(len(errors))
        qdict["up_msg"] = errors[0][:280]
    if dups:
        qdict["up_dup"] = str(len(dups))
        qdict["up_dup_msg"] = dups[0][:280]
    loc = "/ui?" + urlencode(qdict)
    return RedirectResponse(url=loc, status_code=303)


@app.post("/ui/rebuild")
async def ui_rebuild(request: Request, key: str = Form("")) -> Response:
    if not _ui_key_ok(request, key):
        return PlainTextResponse("Unauthorized", status_code=401)
    asyncio.create_task(_run_rebuild_background())
    kq = f"?key={key}" if key else ""
    return RedirectResponse(url=f"/ui{kq}", status_code=303)


@app.post("/ui/ingest")
async def ui_ingest(request: Request, key: str = Form("")) -> Response:
    if not _ui_key_ok(request, key):
        return PlainTextResponse("Unauthorized", status_code=401)
    # Fire-and-forget
    asyncio.create_task(_run_ingest_background())
    return RedirectResponse(url=f"/ui?key={key}", status_code=303)


def _busy_response() -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": "Идёт ingest/rebuild. Дождитесь завершения и повторите."},
        status_code=409,
    )


@app.post("/ui/delete")
async def ui_delete(request: Request, key: str = Form(""), doc_ids: str = Form("")) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if _INGEST_STATE.get("running"):
        return _busy_response()
    ids = [s.strip() for s in (doc_ids or "").split(",") if s.strip()]
    if not ids:
        return JSONResponse({"ok": False, "error": "no doc_ids provided"}, status_code=400)
    try:
        result = await asyncio.to_thread(_pipeline_delete_documents, str(_config_path()), ids)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/wipe")
async def ui_wipe(request: Request, key: str = Form(""), confirm: str = Form("")) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if _INGEST_STATE.get("running"):
        return _busy_response()
    if (confirm or "").strip() != "DELETE":
        return JSONResponse(
            {"ok": False, "error": "Подтверждение: введите слово DELETE."},
            status_code=400,
        )
    try:
        result = await asyncio.to_thread(_pipeline_wipe_index, str(_config_path()))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/clean-orphans")
async def ui_clean_orphans(request: Request, key: str = Form("")) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if _INGEST_STATE.get("running"):
        return _busy_response()
    try:
        result = await asyncio.to_thread(_pipeline_clean_orphans, str(_config_path()))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/clear-incoming")
async def ui_clear_incoming(request: Request, key: str = Form("")) -> JSONResponse:
    if not _ui_key_ok(request, key):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    try:
        result = await asyncio.to_thread(_pipeline_clear_incoming, str(_config_path()))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.get("/mcp")
async def mcp_get(request: Request) -> Response:
    # SSE stream endpoint (optional per spec)
    origin = request.headers.get("origin")
    if not _origin_allowed(origin):
        return PlainTextResponse("Origin not allowed", status_code=403)

    if not _check_api_key(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    if not _accepts_sse(request):
        # Spec allows 405 if SSE is not offered; here we *do* offer SSE, but require Accept.
        return PlainTextResponse("Client must accept text/event-stream", status_code=406)

    client = await _sse_hub.add()

    async def event_stream():
        # Send a 'ready' notification to THIS client (no id -> notification)
        yield _sse_frame({"jsonrpc": "2.0", "method": "doc_rag/ready", "params": {"status": "ok"}})
        keepalive_sec = int(os.environ.get("DOC_RAG_SSE_KEEPALIVE_SEC", "25"))
        last_keepalive = time.time()

        try:
            while True:
                # Wait for next message or keepalive
                timeout = max(1, keepalive_sec - int(time.time() - last_keepalive))
                try:
                    frame = await asyncio.wait_for(client.queue.get(), timeout=timeout)
                    yield frame
                except asyncio.TimeoutError:
                    # SSE comment keepalive (ignored by clients)
                    last_keepalive = time.time()
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            await _sse_hub.remove(client)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.post("/mcp")
async def mcp_post(request: Request) -> Response:
    origin = request.headers.get("origin")
    if not _origin_allowed(origin):
        return PlainTextResponse("Origin not allowed", status_code=403)

    if not _check_api_key(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    try:
        payload: Json = await request.json()
    except Exception:
        return JSONResponse(_err(None, -32700, "Parse error: invalid JSON"), status_code=400)

    timeout_sec = float(os.environ.get("DOC_RAG_TOOL_TIMEOUT_SEC", "30"))
    async with _TOOL_SEM:
        try:
            status, out = await asyncio.wait_for(
                asyncio.to_thread(_handle_jsonrpc, payload), timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            return JSONResponse(_err(None, -32001, "Request timed out"), status_code=504)
    if out is None:
        return Response(status_code=status)

    # For now we return JSON. SSE streaming for POST is optional; GET provides notifications streaming.
    return JSONResponse(out, status_code=status)
