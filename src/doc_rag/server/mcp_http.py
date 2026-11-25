from __future__ import annotations

"""MCP server over Streamable HTTP transport.

Implements a single MCP endpoint (default: /mcp) that supports:
- HTTP POST: JSON-RPC messages (application/json response)
- HTTP GET : optional SSE stream (text/event-stream) for server -> client notifications/requests

This follows the MCP Streamable HTTP transport spec (2025-03-26+).
"""

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import threading


def _get_version() -> str:
    """Read version from package metadata (single source: pyproject.toml)."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("doc-rag")
    except Exception:
        return "0.0.0"


_VERSION = _get_version()
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote as _url_quote
from urllib.parse import urlencode

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import (
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
    rebuild_index as _pipeline_rebuild_index,
)
from doc_rag.raglib.pipeline import (
    wipe_index as _pipeline_wipe_index,
)
from doc_rag.raglib.utils import ensure_dir
from doc_rag.server import metrics as _metrics
from doc_rag.server.logging_setup import (
    configure_logging,
    get_logger,
    new_request_id,
    set_request_id,
)
from doc_rag.server.retrieval import (
    document_preview,
    indexed_catalog,
    load_manifest_json,
    project_root,
)
from doc_rag.server.stats_helpers import (
    build_sources_index,
    count_chunks_per_doc,
    make_size_resolver,
    stat_size,
)

# Configure structured logging on import. Idempotent; honours
# DOC_RAG_LOG_LEVEL and DOC_RAG_LOG_FORMAT env vars.
configure_logging()
log = get_logger("doc_rag.server.mcp_http")

Json = dict[str, Any] | list[Any]

_TOOL_SEM = asyncio.Semaphore(int(os.environ.get("DOC_RAG_MAX_CONCURRENCY", "4")))

_RL_LOCK = asyncio.Lock()
_RL_STATE: dict[str, tuple[float, float]] = {}
_RL_LAST_CLEANUP: float = 0.0
_RL_TTL: float = 300.0  # Evict entries older than 5 minutes


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
    """Simple token bucket per key (usually client IP) with TTL eviction."""
    global _RL_LAST_CLEANUP
    rps, burst = _rate_limit_params()
    if rps <= 0:
        return True

    now = time.time()
    async with _RL_LOCK:
        # Periodic cleanup of stale entries
        if now - _RL_LAST_CLEANUP > _RL_TTL:
            stale = [k for k, (_, last) in _RL_STATE.items() if now - last > _RL_TTL]
            for k in stale:
                del _RL_STATE[k]
            _RL_LAST_CLEANUP = now

        tokens, last = _RL_STATE.get(key, (burst, now))
        # Refill
        tokens = min(burst, tokens + (now - last) * rps)
        if tokens < 1.0:
            _RL_STATE[key] = (tokens, now)
            return False
        tokens -= 1.0
        _RL_STATE[key] = (tokens, now)
        return True


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


def _public_base_url(request: Request) -> str:
    # Prefer X-Forwarded-* when behind a reverse proxy.
    proto = (request.headers.get("x-forwarded-proto") or "").strip()
    host = (request.headers.get("x-forwarded-host") or "").strip()
    if not proto:
        proto = request.url.scheme
    if not host:
        host = request.headers.get("host", "").strip() or request.url.netloc
    return f"{proto}://{host}"


def _mcp_config_payload(name: str, url: str) -> dict[str, Any]:
    return {
        "mcpServers": {
            name: {
                "transport": "streamableHttp",
                "url": url,
            }
        }
    }


def _download_json(payload: dict[str, Any], filename: str) -> Response:
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_INGEST_LOCK = asyncio.Lock()


class _JobBusyError(Exception):
    """Raised when an exclusive job is requested while another is running."""


@contextlib.asynccontextmanager
async def _exclusive_job(job: str):
    """Mark `_INGEST_STATE` busy for the duration of a one-off destructive op.

    Same `_INGEST_STATE["running"]` flag `_run_index_job_background` uses, so
    delete/wipe/clean-orphans/clear-incoming can't race each other or a
    concurrent ingest/rebuild — the check-and-set happens under `_INGEST_LOCK`
    instead of being a separate unlocked read before the operation starts.
    """
    async with _INGEST_LOCK:
        if _INGEST_STATE.get("running"):
            raise _JobBusyError()
        _INGEST_STATE["running"] = True
        _INGEST_STATE["job"] = job
    try:
        yield
    finally:
        async with _INGEST_LOCK:
            _INGEST_STATE["running"] = False
            _INGEST_STATE["job"] = None


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
    # Indexing phase progress (set via the pipeline progress hook):
    # phase is one of parse|chunk|embed|index|done; idx_* count chunks.
    "phase": None,
    "idx_done": 0,
    "idx_total": None,
    "phase_started_at": None,
    # Doc_ids processed during the current job (for live /ui/documents updates)
    "processed_doc_ids": set(),
}


def _reset_progress_state() -> None:
    _INGEST_STATE["current_doc"] = None
    _INGEST_STATE["docs_done"] = 0
    _INGEST_STATE["docs_total"] = None
    _INGEST_STATE["current_doc_started_at"] = None
    _INGEST_STATE["phase"] = None
    _INGEST_STATE["idx_done"] = 0
    _INGEST_STATE["idx_total"] = None
    _INGEST_STATE["phase_started_at"] = None
    _INGEST_STATE["processed_doc_ids"] = set()


def _on_pipeline_progress(phase: str, done: int, total: int) -> None:
    """Progress hook wired into pipeline.set_progress_callback.

    Called from the index-build worker thread during `ingest`/`rebuild`.
    """
    if _INGEST_STATE.get("phase") != phase:
        _INGEST_STATE["phase_started_at"] = time.time()
    _INGEST_STATE["phase"] = phase
    _INGEST_STATE["idx_done"] = int(done)
    _INGEST_STATE["idx_total"] = int(total) if total else None


# Markers from pipeline._log lines we use to derive per-doc progress.
# Keep these in sync with src/doc_rag/raglib/pipeline.py.
_RX_FOUND = re.compile(r"\bfound\s+(\d+)\s+file\(s\)")
_RX_REBUILD_PASS = re.compile(
    r"\brebuild:\s+(?:archived|incoming(?:\s+ingest)?)\s+pass\s+\((\d+)\s+file\(s\)\)"
)
_RX_REBUILD_SUMMARY = re.compile(
    r"\brebuild:\s+(\d+)\s+source\(s\)\s+total,\s+(\d+)\s+skipped,\s+(\d+)\s+to\s+parse"
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

    m = _RX_REBUILD_SUMMARY.search(line)
    if m:
        try:
            total = int(m.group(1))
        except ValueError:
            return
        _INGEST_STATE["docs_total"] = total
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
        # Track processed filename for live /ui/documents updates
        done_path = m.group(1).strip()
        if done_path:
            _INGEST_STATE["processed_doc_ids"].add(os.path.basename(done_path))
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


def _sync_run_rebuild_with_ui_logging(force: bool = False, skip_index: bool = False) -> None:
    from doc_rag.raglib.pipeline import rebuild

    stderr = getattr(sys, "stderr", None) or sys.__stderr__
    tee = _TeeStderrToIngestLog(stderr, _append_ingest_ui_line)
    old_stderr = sys.stderr
    sys.stderr = tee
    try:
        rebuild(str(_config_path()), force=force, skip_index=skip_index)
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

    # Write lock file so interrupted rebuilds can be detected on restart
    lock_path = os.path.join(str(_root_dir()), "build", ".rebuild.lock")
    lock_ts = time.time()
    if "rebuild" in job:
        try:
            ensure_dir(os.path.dirname(lock_path))
            with open(lock_path, "w") as lf:
                lf.write(f"{job}\n{lock_ts}\n")
        except Exception:
            pass

    _append_ingest_ui_line(f"[doc-rag][INFO] --- старт задачи: {job} ---", ansi_strip=False)

    ok = False
    err = None
    try:
        from doc_rag.raglib import pipeline as _pipeline

        _pipeline.set_progress_callback(_on_pipeline_progress)
        try:
            await asyncio.to_thread(sync_runner)
        finally:
            _pipeline.set_progress_callback(None)
        ok = True
    except Exception as exc:
        ok = False
        err = str(exc) or repr(exc)
        for ln in traceback.format_exc().strip().split("\n"):
            if ln.strip():
                _append_ingest_ui_line(ln.rstrip("\r"), ansi_strip=False)
    finally:
        # The index job may have rewritten faiss.index / index_meta.json:
        # drop cached VectorIndex instances so search sees the fresh index
        # (belt-and-suspenders alongside the mtime validation in retrieval).
        try:
            from doc_rag.server.retrieval import invalidate_index_cache

            invalidate_index_cache()
        except Exception:
            pass

    async with _INGEST_LOCK:
        _INGEST_STATE["running"] = False
        _INGEST_STATE["job"] = None
        _INGEST_STATE["last_finished"] = time.time()
        _INGEST_STATE["last_ok"] = ok
        _INGEST_STATE["last_error"] = err
        _reset_progress_state()
        # Remove lock file on successful completion. Write a matching
        # "<lock_ts>.done" marker *first*: if the process is killed between
        # finishing sync_runner and removing the lock (OOM-killer, restart),
        # `_ui_status_payload` can still recognize this exact run as
        # completed after a restart by matching timestamps. Do not fall
        # back to comparing against manifest.json's mtime — delete/
        # clean-orphans/wipe touch it too, which previously caused the
        # "interrupted rebuild" banner to be cleared by an unrelated op.
        if ok and "rebuild" in job:
            done_path = lock_path + ".done"
            try:
                with open(done_path, "w") as df:
                    df.write(f"{lock_ts}\n")
            except OSError:
                pass
            try:
                os.remove(lock_path)
            except OSError:
                pass
            try:
                os.remove(done_path)
            except OSError:
                pass
        if ok:
            _append_ingest_ui_line(f"[doc-rag][INFO] {job} завершился успешно.", ansi_strip=False)
        elif err:
            _append_ingest_ui_line(
                f"[doc-rag][ERROR] {job} завершился с ошибкой: {err}", ansi_strip=False
            )


async def _run_ingest_background() -> None:
    await _run_index_job_background("ingest", _sync_run_ingest_with_ui_logging)


async def _run_rebuild_background(force: bool = False, skip_index: bool = False) -> None:
    await _run_index_job_background(
        "rebuild", lambda: _sync_run_rebuild_with_ui_logging(force=force, skip_index=skip_index)
    )


def _sync_run_rebuild_docs_with_ui_logging(doc_ids: list[str], filenames: list[str]) -> None:
    from doc_rag.raglib.pipeline import rebuild_docs

    stderr = getattr(sys, "stderr", None) or sys.__stderr__
    tee = _TeeStderrToIngestLog(stderr, _append_ingest_ui_line)
    old_stderr = sys.stderr
    sys.stderr = tee
    try:
        rebuild_docs(str(_config_path()), doc_ids, filenames)
    finally:
        sys.stderr = old_stderr
        tee.finish()


async def _run_rebuild_docs_background(doc_ids: list[str], filenames: list[str]) -> None:
    total = len(doc_ids) + len(filenames)
    await _run_index_job_background(
        f"rebuild-docs ({total} doc(s))",
        lambda: _sync_run_rebuild_docs_with_ui_logging(doc_ids, filenames),
    )


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


def _doc_download_tool(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Download a source document by filename."""
    import base64

    # Size limit on the FILE (checked via stat before any read): the
    # payload returned to the client is base64 (~4/3 of the file), so a
    # 25 MB file already produces a ~33 MB response.
    _MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024

    filename = arguments.get("filename", "")
    if not filename:
        return [{"type": "text", "text": "Error: filename is required"}]

    # Honor DOC_RAG_ROOT like the rest of the server: project_root() also
    # expands ~ in the override (unlike _root_dir(), which always points
    # at the package checkout).
    root = Path(project_root())
    cfg_path = _config_path()
    try:
        with cfg_path.open(encoding="utf-8") as f:
            import yaml

            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    paths = cfg.get("paths", {}) or {}

    def _serve(p: Path) -> list[dict[str, Any]]:
        try:
            if p.stat().st_size > _MAX_DOWNLOAD_BYTES:
                return [{"type": "text", "text": "Error: file exceeds 25 MB limit"}]
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return [
                {
                    "type": "file",
                    "file": {
                        "name": filename,
                        "mimeType": "application/octet-stream",
                        "data": b64,
                    },
                }
            ]
        except Exception as e:
            return [{"type": "text", "text": f"Error reading file: {e}"}]

    # Resolution order: (1) manifest source_file (exact path under
    # sources/, keeps nested layouts reachable); (2) rglob scan of the
    # sources directories so files absent from the manifest stay
    # downloadable.
    try:
        from doc_rag.server.retrieval import _load_manifest_cached

        manifest = _load_manifest_cached()
        docs = manifest.get("documents", []) if isinstance(manifest, dict) else []
        for d in docs:
            if not isinstance(d, dict):
                continue
            sf = d.get("source_file")
            if not (isinstance(sf, str) and sf and os.path.basename(sf) == filename):
                continue
            p = root / sf
            # The manifest entry may carry traversal (..\/); stay inside root.
            if p.is_file() and p.resolve().is_relative_to(root.resolve()):
                return _serve(p)
    except Exception as exc:
        # Manifest resolve is best-effort: fall through to the rglob scan,
        # but leave a trace for diagnosis.
        log.debug("doc_download manifest resolve failed: %s: %s", type(exc).__name__, exc)

    archived_rel = paths.get("sources_archived", "sources/archived")
    incoming_rel = paths.get("sources_incoming", "sources/incoming")

    # Search in archived and incoming directories
    for dir_rel in [archived_rel, incoming_rel]:
        dir_path = root / dir_rel
        if not dir_path.is_dir():
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and p.name == filename:
                return _serve(p)

    return [{"type": "text", "text": f"File not found: {filename}"}]


def _doc_search_docs_tool(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Search documents and return document-level results with download links."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        return [{"type": "text", "text": "Empty query. Provide `query` string."}]

    top_k_raw = arguments.get("top_k", 10)
    try:
        top_k = int(top_k_raw)
    except Exception:
        top_k = 10

    from doc_rag.server.retrieval import doc_search, indexed_catalog

    results = doc_search(query=query, top_k=max(top_k * 3, 30), namespace="default", filters=None)

    # After the search: a half-open probe in this request may have recovered
    # the breaker, and then the results are semantic (no notice).
    notice_items: list[dict[str, Any]] = []
    try:
        if indexed_catalog().get("embeddings_degraded"):
            from doc_rag.server.search_tool import DEGRADED_NOTICE

            notice_items = [{"type": "text", "text": DEGRADED_NOTICE}]
    except Exception:
        notice_items = []

    if not results:
        return [*notice_items, {"type": "text", "text": "No results."}]

    seen_docs: dict[str, dict[str, Any]] = {}
    for r in results:
        source_file = r.get("source_file", "")
        if not source_file:
            continue
        bn = os.path.basename(source_file)
        if bn in seen_docs:
            old = seen_docs[bn]
            if (r.get("score") or 0) > (old.get("best_score") or 0):
                old["best_score"] = r.get("score")
                old["best_section"] = r.get("section_path", "")
                old["best_preview"] = str(r.get("text", ""))[:200]
            old["match_count"] = old.get("match_count", 0) + 1
            continue
        seen_docs[bn] = {
            "filename": bn,
            "source_file": source_file,
            "doc_id": r.get("doc_id", ""),
            "best_score": r.get("score"),
            "best_section": r.get("section_path", ""),
            "best_preview": str(r.get("text", ""))[:200],
            "match_count": 1,
        }

    docs = sorted(seen_docs.values(), key=lambda d: d.get("best_score") or 0, reverse=True)[:top_k]

    if not docs:
        return [*notice_items, {"type": "text", "text": "No documents found."}]

    base_url = os.environ.get("DOC_RAG_HTTP_HOST", "http://127.0.0.1:3333")
    host = os.environ.get("DOC_RAG_HTTP_HOST", "")
    port = os.environ.get("DOC_RAG_HTTP_PORT", "3333")
    if host and not host.startswith("http"):
        base_url = f"http://{host}:{port}"

    lines = [f"Found {len(docs)} document(s):\n"]
    for i, d in enumerate(docs, 1):
        score = d.get("best_score")
        score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        section = d.get("best_section", "")
        preview = d.get("best_preview", "")
        download_url = f"{base_url}/api/v1/download?filename={_url_quote(d['filename'])}"

        header = f"[{i}] {d['filename']}"
        if d.get("doc_id"):
            header += f" (doc_id={d['doc_id']})"
        header += f"  score={score_s}  matches={d.get('match_count', 1)}"
        if section:
            header += f"  §{section}"

        block = header
        if preview:
            block += f"\n  {preview}"
        block += f"\n  Download: {download_url}"
        lines.append(block)

    return [*notice_items, {"type": "text", "text": "\n\n".join(lines)}]


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
            "serverInfo": {"name": "doc-rag", "version": _VERSION},
            "capabilities": {"tools": {"listChanged": True}},
        }
        return 200, _ok(req_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "doc_search",
                "description": "Search the document knowledge base. Returns relevant chunks with scores and structured citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "top_k": {
                            "type": "integer",
                            "default": 6,
                            "description": "Number of results to return",
                        },
                        "namespace": {
                            "type": "string",
                            "default": "default",
                            "description": "Collection namespace",
                        },
                        "doc_id": {"type": "string", "description": "Filter by document ID"},
                        "section_path": {
                            "type": "string",
                            "description": "Filter by section path prefix",
                        },
                        "tables_only": {
                            "type": "boolean",
                            "default": False,
                            "description": "Return only table chunks",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "doc_download",
                "description": "Download a source document by filename. Returns the file content as base64-encoded data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename of the document to download (e.g. 'СТО_34.01-5.1-006-2023v4.pdf')",
                        },
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "doc_search_docs",
                "description": "Search documents by query and return document-level results with download links. Returns unique documents (not chunks) with relevance scores and URLs to download source files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max number of documents to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]
        return 200, _ok(req_id, {"tools": tools})

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if name not in ("doc_search", "doc_download", "doc_search_docs"):
            _metrics.record_mcp_request(
                tool=name or "unknown", status="unknown_tool", duration_seconds=0.0
            )
            return 200, _ok(
                req_id,
                {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True},
            )

        started = time.time()
        try:
            if name == "doc_search":
                from doc_rag.server.search_tool import doc_search_tool

                content = doc_search_tool(arguments)
            elif name == "doc_download":
                content = _doc_download_tool(arguments)
            elif name == "doc_search_docs":
                content = _doc_search_docs_tool(arguments)
            _metrics.record_mcp_request(name, "ok", time.time() - started)
            return 200, _ok(req_id, {"content": content, "isError": False})
        except Exception as exc:
            _metrics.record_mcp_request(name, "error", time.time() - started)
            return 200, _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"{name} failed: {exc}"}],
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


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm up the embedding model before the service starts accepting work.

    Offline only: the loader resolves a local directory (or the local HF
    cache) and loads with local_files_only=True, so startup cannot hang on
    the network. A failed load is recorded in model_loader.status() and
    logged; /health/ready will report 503 in semantic mode, but the
    service itself keeps running (lexical fallback).
    """
    from doc_rag.raglib import model_loader as model_loader
    from doc_rag.server.retrieval import load_config

    # Initialized before the try: if load_config() raises, the watcher
    # block below must still see a dict (an empty cfg just disables the
    # watcher) instead of raising NameError.
    cfg: dict[str, Any] = {}
    try:
        cfg = load_config()
        model = await asyncio.to_thread(model_loader.load_embedder, cfg)
        st = model_loader.status()
        if model is None:
            log.error(
                "Модель эмбеддингов не загружена при старте (%s). "
                "Семантический поиск будет недоступен; /health/ready вернёт 503. "
                "Выполните scripts/fetch_model.py и перезапустите сервис, "
                "либо переключите mcp.retrieval_mode на lexical.",
                st.get("error") or "причина неизвестна",
            )
        else:
            log.info(
                "Модель эмбеддингов загружена при старте (source=%s, path=%s)",
                st.get("source"),
                st.get("path"),
            )
    except Exception as exc:  # config errors must not prevent serving
        log.error(
            "Прогрев модели при старте не выполнен (%s: %s); "
            "семантический поиск будет недоступен до перезапуска.",
            type(exc).__name__,
            exc,
        )

    # Background HF Hub weight check (read-only, no download): starts only
    # when the interval is positive, the model is resolvable locally and
    # huggingface_hub is installed. Disabled for explicit model_path.
    from doc_rag.server import model_watcher

    try:
        task = model_watcher.start(cfg)
        if task is not None:
            log.info(
                "Фоновая сверка весов с HF Hub запущена (интервал %s ч).",
                model_watcher._parse_interval(cfg),
            )
    except Exception as exc:  # watcher failures must not prevent serving
        log.warning("Не удалось запустить сверку весов с HF Hub (%s: %s).", type(exc).__name__, exc)

    yield

    model_watcher.stop()


app = FastAPI(title="doc-rag MCP HTTP", version=_VERSION, lifespan=_lifespan)


def _mount_ui_next(app: FastAPI) -> None:
    """Serve the Svelte UI bundle at `/ui-next/`.

    The build output lives at `ui/dist/`. There is no fallback anymore —
    if it hasn't been built, `/ui-next/` 404s. `bootstrap.sh` and
    `install_server_native.sh` fail the install if Node/npm isn't
    available, so a running server always has this built.

    Note: this can't also be mounted at bare `/ui` — `StaticFiles` does
    prefix matching, and `/ui/status`, `/ui/upload`, etc. are real JSON
    API routes registered later in this module; a mount at `/ui` would
    shadow all of them. `GET /ui` below is a plain redirect instead.
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
            path = request.url.path
            if path in ("/ui-next/", "/ui-next", "/ui/", "/ui") or path.endswith("/index.html"):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
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
            "/ui/quality",
        ):
            _log_line(f"{request.method} {request.url.path} -> {code} ({dur_ms}ms)")
        set_request_id(None)


def _readiness_state() -> dict[str, Any]:
    """Compute current readiness - manifest present, no background job running.

    Additionally checks subsystem health: model, semantic index files, disk
    space. In semantic/hybrid mode the service cannot answer its primary
    queries when the embedding model cannot load, the index no longer
    matches the model weights, or the local index files are missing, so
    readiness fails with `model_missing` / `index_stale` / `index_missing`
    reasons; in lexical mode all of these are informational only and not
    reported as failure reasons. The `index_missing` gate applies only to
    the faiss backend: qdrant/pgvector store vectors remotely and must not
    fail readiness for absent local faiss files.
    """
    root = (os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir())
    manifest_abs = os.path.join(root, "build", "manifest.json")
    has_manifest = os.path.isfile(manifest_abs)
    job_running = bool(_INGEST_STATE.get("running"))
    job_name = _INGEST_STATE.get("job") if job_running else None

    # Config: needed here for backend/index-dir resolution and the mode
    # gates below.
    try:
        from doc_rag.server.retrieval import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    mode = str((cfg.get("mcp", {}) or {}).get("retrieval_mode", "semantic")).lower()
    semantic_required = mode in ("semantic", "hybrid")
    index_backend = str((cfg.get("index") or {}).get("backend", "faiss")).lower()
    index_dir_rel = str((cfg.get("paths") or {}).get("index_dir", "build/index"))

    # Subsystem checks
    checks: dict[str, bool] = {}

    # Embedding model (from model_loader's last load attempt)
    try:
        from doc_rag.raglib import model_loader as model_loader

        model_status = model_loader.status()
    except Exception:
        model_status = {}
    checks["model_ready"] = bool(model_status.get("ready"))
    # Informational: embeddings are failing fast and searches degrade to
    # lexical. Deliberately NOT in `checks` (that dict is readiness gates)
    # and NOT a readiness failure: lexical results are still correct and fast.
    embeddings_degraded = bool(model_status.get("degraded"))

    # Where Ollama runs the embedding model (gpu / cpu / unloaded /
    # unreachable). A CPU-only backend is the 2026-11-25 incident signature;
    # it is invisible in /api/tags and only shows up as slow embeddings.
    ollama_backend: str | None = None
    try:
        embeddings_cfg = cfg.get("embeddings") or {}
        if str(embeddings_cfg.get("provider") or "").strip().lower() == "ollama":
            from doc_rag.raglib.ollama_embedder import ollama_backend_state

            ollama_backend = ollama_backend_state(
                str(embeddings_cfg.get("api_base") or "http://127.0.0.1:11434"),
                str(embeddings_cfg.get("model_name") or ""),
                timeout=1.0,
            )
    except Exception:
        ollama_backend = None

    # Semantic index files (faiss backend). Both files must exist — same
    # semantics as indexed_catalog's semantic_index_present.
    index_dir = os.path.join(root, index_dir_rel)
    index_faiss = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "index_meta.json")
    checks["faiss_index"] = os.path.isfile(index_faiss)
    checks["index_meta"] = os.path.isfile(meta_path)

    # Disk space (at least 1 GB free)
    try:
        import shutil

        usage = shutil.disk_usage(root)
        checks["disk_space"] = usage.free >= 1024**3
    except Exception:
        checks["disk_space"] = True  # don't block on check failure

    # Index staleness: read index_meta.json directly (cheap; no
    # indexed_catalog() so the probe stays lightweight). The parse is
    # mtime-cached in retrieval._INDEX_META_CACHE via the shared helper.
    try:
        from doc_rag.server.retrieval import read_index_weights_fingerprint

        index_fp = read_index_weights_fingerprint(meta_path)
    except Exception:
        index_fp = None

    model_fp = model_status.get("fingerprint")
    model_fp_s = str(model_fp) if isinstance(model_fp, str) and model_fp.strip() else None
    index_stale = bool(checks["model_ready"] and model_fp_s and index_fp and model_fp_s != index_fp)

    ready = has_manifest and not job_running
    reasons: list[str] = []
    if not has_manifest:
        reasons.append("manifest_missing")
    if job_running:
        reasons.append(f"job_in_flight:{job_name}")

    # Model gate applies only when the retrieval mode needs embeddings.
    if semantic_required and not checks["model_ready"]:
        ready = False
        reasons.append("model_missing")
    if semantic_required and index_stale:
        # Stale index means semantic search is unusable: gate readiness too.
        ready = False
        reasons.append("index_stale")
    if (
        semantic_required
        and index_backend == "faiss"
        and not (checks["faiss_index"] and checks["index_meta"])
    ):
        # Missing local index files is a separate failure from a stale one:
        # without them semantic search degrades to 6-8s per query. Only the
        # faiss backend is gated: qdrant/pgvector keep vectors remotely.
        ready = False
        reasons.append("index_missing")

    return {
        "ready": ready,
        "has_manifest": has_manifest,
        "job_running": job_running,
        "job": job_name,
        "retrieval_mode": mode,
        "checks": checks,
        "reasons": reasons,
        "index_stale": index_stale,
        "embeddings_degraded": embeddings_degraded,
        "ollama_backend": ollama_backend,
    }


@app.get("/health/live")
async def health_live() -> JSONResponse:
    """Liveness probe — the process is up. Always 200 unless the event loop is wedged."""
    return JSONResponse({"status": "ok", "name": "doc-rag"})


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    """Readiness probe - 200 if the service can answer real queries, 503 otherwise.

    Ready when:
      - build/manifest.json exists (something has been ingested), and
      - no ingest or rebuild job is currently in flight, and
      - in semantic/hybrid mode: the embedding model is loaded
        (`model_ready`, reason `model_missing`), the FAISS index was built
        with the current weights (`index_stale` otherwise), and — for the
        faiss backend — the local index files exist (`index_missing`
        otherwise; qdrant/pgvector are not gated on local files).

    In lexical mode the model/index checks stay informational: none of
    `model_missing`, `index_stale` or `index_missing` is reported.
    """
    state = await asyncio.to_thread(_readiness_state)
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
    state = await asyncio.to_thread(_readiness_state)
    return JSONResponse(
        {
            "status": "ok",
            "name": "doc-rag",
            "version": _VERSION,
            "ready": state["ready"],
            "reasons": state["reasons"],
            "embeddings_degraded": state.get("embeddings_degraded", False),
            "ollama_backend": state.get("ollama_backend"),
        }
    )


@app.get("/api/v1/manifest")
async def api_v1_manifest(request: Request) -> JSONResponse:
    """Machine-readable manifest for CI / reproducibility."""
    data = load_manifest_json()
    if not data:
        return JSONResponse({"error": "manifest_not_found"}, status_code=404)
    return JSONResponse(data)


@app.get("/ui/config/raw")
async def ui_config_raw() -> JSONResponse:
    p = _config_path()
    try:
        yaml_text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(p), "yaml": yaml_text})


@app.post("/ui/config/save")
async def ui_config_save(content: str = Form("")) -> JSONResponse:
    ok, msg = _validate_root_yaml(content)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    try:
        _atomic_write_text(_config_path(), content)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "path": str(_config_path())})


@app.get("/ui/config/parsed")
async def ui_config_parsed() -> JSONResponse:
    """Parsed view of config.yaml for the structured form editor.

    The form populates from this; comment-preserving writes go through
    `/ui/config/patch`. The raw text path (`/ui/config/raw`) still backs
    the Advanced tab."""
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
async def ui_config_validate(updates: str = Form("")) -> JSONResponse:
    """Validate config field updates without writing. Returns errors if any."""
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
    "parsing.quality_fallback_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "parsing.min_chars_per_page": {"type": "int", "min": 0},
    "sectioning.min_heading_len": {"type": "int", "min": 1},
    "sectioning.max_heading_len": {"type": "int", "min": 1},
    "chunking.strategy": {"options": ["fixed", "recursive"]},
    "chunking.target_tokens": {"type": "int", "min": 1},
    "chunking.overlap_tokens": {"type": "int", "min": 0},
    "chunking.dedup_similarity_threshold": {"type": "float", "min": 0.0, "max": 1.0},
    "embeddings.batch_size": {"type": "int", "min": 1},
    "embeddings.device": {"options": ["cpu", "cuda"]},
    "embeddings.update_check_interval_hours": {"type": "int", "min": 0},
    "index.backend": {"options": ["faiss"]},
    "index.metric": {"options": ["ip", "l2"]},
    "index.top_k": {"type": "int", "min": 1},
    "quality.fail_on_severity": {"options": ["never", "warn", "error"]},
    "mcp.retrieval_mode": {"options": ["semantic", "lexical", "hybrid"]},
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
async def ui_config_patch(updates: str = Form("")) -> JSONResponse:
    """Apply field-level updates to config.yaml while preserving comments.

    `updates` is a JSON object of dotted paths to values, e.g.
    `{"chunking.target_tokens": 512, "parsing.pdf_backend": "docling"}`.
    Only paths that already exist in the file are written; unknown paths
    are rejected so the form cannot silently invent keys."""
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
# Each editable key carries a type for validation.

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
async def ui_env_get() -> JSONResponse:
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
    return JSONResponse({"ok": True, "path": str(env_file), "fields": fields})


@app.post("/ui/env/save")
async def ui_env_save(updates: str = Form("")) -> JSONResponse:
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
async def ui_restart_service(background_tasks: BackgroundTasks) -> JSONResponse:
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
    # Convert set to list for JSON serialization
    if isinstance(out.get("processed_doc_ids"), set):
        out["processed_doc_ids"] = list(out["processed_doc_ids"])
    out["log_tail"] = _snapshot_ingest_ui_log_tail()
    out["http_log_tail"] = _snapshot_server_http_ui_log_tail()
    log_path = (os.environ.get("DOC_RAG_HTTP_LOG") or "").strip()
    out["http_log_file"] = log_path if log_path else None
    eta_s, eta_h = _compute_eta()
    out["eta_seconds"] = eta_s
    out["eta_human"] = eta_h
    # Indexing-phase elapsed/ETA (chunk-based, independent of doc count).
    out["phase_elapsed_s"] = None
    out["idx_eta_s"] = None
    out["idx_eta_human"] = None
    phase_started = _INGEST_STATE.get("phase_started_at")
    if _INGEST_STATE.get("running") and phase_started:
        elapsed = max(0.0, time.time() - phase_started)
        out["phase_elapsed_s"] = elapsed
        idx_done = int(_INGEST_STATE.get("idx_done") or 0)
        idx_total = _INGEST_STATE.get("idx_total")
        if idx_done > 0 and idx_total and idx_done < idx_total:
            remaining = (elapsed / idx_done) * (idx_total - idx_done)
            out["idx_eta_s"] = remaining
            out["idx_eta_human"] = _format_eta_seconds(remaining)
    # Detect interrupted rebuild via lock file.  The lock is written at
    # rebuild start and removed on success.  After a server restart the
    # in-memory state is gone, so on success `_run_index_job_background`
    # also drops a "<lock_ts>.done" marker before removing the lock — if
    # the process is killed in that narrow window, the marker survives and
    # we can match it against the lock's own timestamp to confirm *this*
    # run finished. We deliberately do NOT compare against manifest.json's
    # mtime: delete/clean-orphans/wipe touch it too, which used to make an
    # unrelated op silently clear a real "rebuild was interrupted" signal.
    lock_path = os.path.join(str(_root_dir()), "build", ".rebuild.lock")
    rebuild_interrupted = False
    if os.path.isfile(lock_path) and not _INGEST_STATE.get("running"):
        stale = False
        done_path = lock_path + ".done"
        try:
            with open(lock_path, encoding="utf-8") as lf:
                lines = lf.read().strip().split("\n")
                lock_ts = lines[1] if len(lines) >= 2 else ""
            if lock_ts and os.path.isfile(done_path):
                with open(done_path, encoding="utf-8") as df:
                    stale = df.read().strip() == lock_ts
        except OSError:
            pass
        if stale:
            try:
                os.remove(lock_path)
            except OSError:
                pass
            try:
                os.remove(done_path)
            except OSError:
                pass
        else:
            rebuild_interrupted = True
    out["rebuild_interrupted"] = rebuild_interrupted
    out["version"] = _VERSION
    try:
        out["indexed"] = indexed_catalog()
    except Exception as exc:
        out["indexed"] = {"error": str(exc), "documents": [], "document_count": 0}
    return out


@app.get("/ui")
async def ui_redirect() -> Response:
    """The Svelte SPA is mounted at `/ui-next/` (see `_mount_ui_next` — it
    can't be mounted at bare `/ui` without shadowing the `/ui/*` JSON API
    routes below). Bare `/ui` just redirects there."""
    return RedirectResponse(url="/ui-next/", status_code=302)


@app.get("/ui/mcp/cursor.json")
async def ui_mcp_cursor(request: Request) -> Response:
    base = _public_base_url(request)
    payload = _mcp_config_payload("doc-rag", f"{base}/mcp")
    return _download_json(payload, "cursor-mcp-doc-rag.json")


@app.get("/ui/mcp/vscode.json")
async def ui_mcp_vscode(request: Request) -> Response:
    base = _public_base_url(request)
    payload = _mcp_config_payload("doc-rag", f"{base}/mcp")
    return _download_json(payload, "vscode-mcp-doc-rag.json")


def _load_quality_summary() -> dict[str, Any]:
    """Load full quality summary dict for the /ui/quality endpoint."""
    try:
        root = os.environ.get("DOC_RAG_ROOT", "")
        if not root:
            root = str(_config_path().parent.parent)
        summary_path = os.path.join(root, "build", "quality", "summary.json")
        if not os.path.exists(summary_path):
            return {"documents": [], "message": "No quality reports yet"}
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"documents": [], "message": "No quality reports yet"}


@app.get("/ui/status")
async def ui_status() -> JSONResponse:
    return JSONResponse(_ui_status_payload())


@app.get("/ui/quality")
async def ui_quality() -> JSONResponse:
    """Return quality summary from build/quality/summary.json."""
    try:
        data = _load_quality_summary()
        return JSONResponse({"ok": True, **data})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/ui/doc-stats")
async def ui_doc_stats() -> JSONResponse:
    """Return detailed statistics for all documents."""
    try:
        # Respect DOC_RAG_ROOT the same way _config_path() does: tests and
        # alternative deployments point the server at another corpus root.
        root = Path((os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir()))
        cfg_path = _config_path()
        with cfg_path.open(encoding="utf-8") as f:
            import yaml

            cfg = yaml.safe_load(f) or {}
        paths = cfg.get("paths", {}) or {}
        manifest_path = root / paths.get("manifest_path", "build/manifest.json")
        quality_dir = root / "build" / "quality"

        # Shared size resolver (same as /ui/documents): exact manifest
        # path -> legacy roots -> basename index under sources/.
        sources_index = build_sources_index(root)
        resolve_size = make_size_resolver(root, paths, sources_index)

        # One pass over chunks.jsonl for ALL doc_ids: the old code
        # re-scanned the whole file per document (~12 s at 199 docs).
        chunk_counts = count_chunks_per_doc(
            root / paths.get("chunks_dir", "build/chunks_jsonl") / "chunks.jsonl"
        )

        stats = []
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            for d in manifest.get("documents", []):
                doc_id = d.get("doc_id", "")
                source_file = d.get("source_file", "")
                bn = os.path.basename(source_file)

                # Get quality data
                quality = {}
                q_path = quality_dir / f"{doc_id}.json"
                if q_path.is_file():
                    try:
                        with q_path.open(encoding="utf-8") as f:
                            quality = json.load(f)
                    except Exception:
                        pass

                sz = resolve_size(source_file)

                # Get blocks count from quality
                blocks = quality.get("blocks", {})
                total_blocks = sum(blocks.values())

                stats.append(
                    {
                        "doc_id": doc_id,
                        "filename": bn,
                        "source_file": source_file,
                        "size_bytes": sz,
                        "size_mb": round(sz / 1024 / 1024, 2) if sz is not None else None,
                        "chunk_count": chunk_counts.get(doc_id) or d.get("chunk_count", 0),
                        "quality_score": quality.get("score", d.get("quality_score", 0)),
                        # Per-doc quality reports have no "warning_count"
                        # key -- count the warnings list (the old lookup
                        # always fell through to 0, rendering "-" in UI).
                        "quality_warnings": len(quality.get("warnings") or [])
                        or d.get("quality_warning_count", 0),
                        "pages": quality.get("pages", 0),
                        "blocks": blocks,
                        "total_blocks": total_blocks,
                        "edition_year": d.get("edition_year"),
                        "coverage": d.get("coverage", {}),
                    }
                )

        return JSONResponse({"ok": True, "documents": stats, "count": len(stats)})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.get("/ui/documents")
async def ui_documents() -> JSONResponse:
    """List ALL source files (incoming + archived) cross-referenced with manifest.

    Returns indexed docs (with chunk_count, quality) and pending files
    (from sources/incoming and sources/archived not yet in manifest).
    """
    try:
        result = await asyncio.to_thread(_build_documents_list)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def _build_documents_list() -> dict[str, Any]:
    # Honor DOC_RAG_ROOT like the other helpers (it was the only caller
    # that read _root_dir() directly, so tests and overrides were ignored).
    root = Path((os.environ.get("DOC_RAG_ROOT") or "").strip() or str(_root_dir()))
    cfg_path = _config_path()
    with cfg_path.open(encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f) or {}
    paths = cfg.get("paths", {}) or {}
    incoming_rel = paths.get("sources_incoming", "sources/incoming")
    archived_rel = paths.get("sources_archived", "sources/archived")
    manifest_rel = paths.get("manifest_path", "build/manifest.json")

    incoming_dir = root / incoming_rel
    archived_dir = root / archived_rel
    manifest_path = root / manifest_rel

    SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".md", ".txt"}

    def _scan_dir(d: Path) -> list[dict[str, Any]]:
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                out.append(
                    {
                        "path": str(p.relative_to(root)),
                        "basename": p.name,
                        "size": stat_size(p),
                    }
                )
        return out

    incoming_files = _scan_dir(incoming_dir)
    archived_files = _scan_dir(archived_dir)

    # Same resolver as /ui/doc-stats: exact manifest path -> legacy roots
    # -> basename index. Before this, the Documents page kept the old
    # root-only lookup and rendered a dash for ~60 documents whose real
    # size Stats showed fine.
    sources_index = build_sources_index(root)
    resolve_size = make_size_resolver(root, paths, sources_index)

    indexed_docs: list[dict[str, Any]] = []
    indexed_basenames: set[str] = set()
    _seen_doc_ids: set[str] = set()
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            for d in manifest.get("documents", []):
                sf = str(d.get("source_file", ""))
                bn = os.path.basename(sf)
                did = d.get("doc_id", "")
                if did in _seen_doc_ids:
                    continue
                _seen_doc_ids.add(did)
                indexed_basenames.add(bn)
                # null (not 0) when the source file is gone from disk;
                # a real empty file keeps size 0 and is rendered as "0 B".
                sz = resolve_size(sf)
                indexed_docs.append(
                    {
                        "doc_id": did,
                        "basename": bn,
                        "source_file": sf,
                        "chunk_count": d.get("chunk_count"),
                        "edition_year": d.get("edition_year"),
                        "size": sz,
                        "indexed": True,
                        "coverage": d.get("coverage", {}),
                    }
                )
        except Exception:
            pass

    pending = []
    seen_pending: set[str] = set()
    for f in incoming_files + archived_files:
        bn = f["basename"]
        if bn not in indexed_basenames and bn not in seen_pending:
            seen_pending.add(bn)
            pending.append(
                {
                    "basename": bn,
                    "path": f["path"],
                    "size": f["size"],
                    "indexed": False,
                }
            )

    rebuild_progress = None
    job = _INGEST_STATE.get("job") or ""
    # Applies to any job that reports per-document progress, not only
    # "rebuild*". ingest() also emits "found N file(s)" and ok/skip/failed
    # lines, so `docs_total`/`processed_doc_ids` are populated for it too —
    # without this the UI never updated during an ingest run.
    if _INGEST_STATE.get("running") and (
        "rebuild" in job
        or _INGEST_STATE.get("docs_total") is not None
        or _INGEST_STATE.get("phase") is not None
    ):
        processed = _INGEST_STATE.get("processed_doc_ids", set())
        current = _INGEST_STATE.get("current_doc")
        rebuild_progress = {
            "current_doc": current,
            "docs_done": _INGEST_STATE.get("docs_done", 0),
            "docs_total": _INGEST_STATE.get("docs_total"),
            "processed_basenames": list(processed),
            "phase": _INGEST_STATE.get("phase"),
            "idx_done": _INGEST_STATE.get("idx_done", 0),
            "idx_total": _INGEST_STATE.get("idx_total"),
        }
        for d in indexed_docs:
            bn = d["basename"]
            if bn == current:
                d["status"] = "processing"
            elif bn in processed:
                d["status"] = "processed"
        for p in pending:
            if p["basename"] in processed:
                p["status"] = "processed"
            elif p["basename"] == current:
                p["status"] = "processing"

    return {
        "ok": True,
        "indexed": indexed_docs,
        "pending": pending,
        "indexed_count": len(indexed_docs),
        "pending_count": len(pending),
        "rebuild_progress": rebuild_progress,
    }


@app.get("/ui/document-preview")
async def ui_document_preview(doc_id: str = "") -> JSONResponse:
    return JSONResponse(document_preview(doc_id))


@app.get("/api/v1/download")
async def api_v1_download(filename: str = "") -> Response:
    """Download a source document by filename."""
    if not filename:
        return JSONResponse({"error": "требуется имя файла"}, status_code=400)
    root = _root_dir()
    cfg_path = _config_path()
    try:
        with cfg_path.open(encoding="utf-8") as f:
            import yaml

            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    paths = cfg.get("paths", {}) or {}
    archived_rel = paths.get("sources_archived", "sources/archived")
    incoming_rel = paths.get("sources_incoming", "sources/incoming")
    for dir_rel in [archived_rel, incoming_rel]:
        dir_path = root / dir_rel
        if not dir_path.is_dir():
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and p.name == filename:
                from starlette.responses import FileResponse

                return FileResponse(str(p), filename=filename)
    return JSONResponse({"error": f"Файл не найден: {filename}"}, status_code=404)


@app.post("/ui/upload")
async def ui_upload(
    request: Request,
    files: Annotated[list[UploadFile], File()],
) -> Response:
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

    # Return JSON for fetch/XHR requests, 303 for traditional form POSTs.
    is_xhr = request.headers.get("x-requested-with") == "XMLHttpRequest"
    if is_xhr:
        return JSONResponse(
            {
                "ok": saved > 0,
                "saved": saved,
                "dups": dups,
                "errors": errors,
            }
        )

    qdict: dict[str, str] = {}
    if saved > 0:
        qdict["up_saved"] = str(saved)
    if errors:
        qdict["up_err"] = str(len(errors))
        qdict["up_msg"] = errors[0][:280]
    if dups:
        qdict["up_dup"] = str(len(dups))
        qdict["up_dup_msg"] = dups[0][:280]
    loc = "/ui-next/?" + urlencode(qdict)
    return RedirectResponse(url=loc, status_code=303)


@app.post("/ui/rebuild")
async def ui_rebuild(request: Request) -> Response:
    force = request.query_params.get("force", "").lower() in ("1", "true", "yes")
    asyncio.create_task(_run_rebuild_background(force=force))
    return RedirectResponse(url="/ui-next/", status_code=303)


@app.post("/ui/rebuild-index")
async def ui_rebuild_index() -> Response:
    asyncio.create_task(
        _run_index_job_background(
            "rebuild-index", lambda: _pipeline_rebuild_index(str(_config_path()))
        )
    )
    return RedirectResponse(url="/ui-next/", status_code=303)


@app.post("/ui/rebuild-skip-index")
async def ui_rebuild_skip_index() -> Response:
    """Full rebuild: parse + chunk only, no vector index (for local GPU processing)."""
    asyncio.create_task(_run_rebuild_background(force=True, skip_index=True))
    return RedirectResponse(url="/ui-next/", status_code=303)


@app.post("/ui/rebuild-docs")
async def ui_rebuild_docs(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "невалидный JSON"}, status_code=400)
    doc_ids = body.get("doc_ids") if isinstance(body, dict) else None
    filenames = body.get("filenames") if isinstance(body, dict) else None
    if not isinstance(doc_ids, list):
        doc_ids = []
    if not isinstance(filenames, list):
        filenames = []
    doc_ids = [str(d) for d in doc_ids if d]
    filenames = [str(f) for f in filenames if f]
    if not doc_ids and not filenames:
        return JSONResponse(
            {"ok": False, "error": "требуются doc_ids или filenames"}, status_code=400
        )
    asyncio.create_task(_run_rebuild_docs_background(doc_ids, filenames))
    total = len(doc_ids) + len(filenames)
    return JSONResponse({"ok": True, "message": f"rebuild-docs запущен для {total} документ(ов)"})


@app.post("/ui/ingest")
async def ui_ingest() -> Response:
    # Fire-and-forget
    asyncio.create_task(_run_ingest_background())
    return RedirectResponse(url="/ui-next/", status_code=303)


def _busy_response() -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": "Идёт ingest/rebuild. Дождитесь завершения и повторите."},
        status_code=409,
    )


@app.post("/ui/delete")
async def ui_delete(doc_ids: str = Form("")) -> JSONResponse:
    ids = [s.strip() for s in (doc_ids or "").split(",") if s.strip()]
    if not ids:
        return JSONResponse({"ok": False, "error": "doc_ids не указаны"}, status_code=400)
    try:
        async with _exclusive_job("delete"):
            result = await asyncio.to_thread(_pipeline_delete_documents, str(_config_path()), ids)
    except _JobBusyError:
        return _busy_response()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/wipe")
async def ui_wipe(confirm: str = Form("")) -> JSONResponse:
    if (confirm or "").strip() != "DELETE":
        return JSONResponse(
            {"ok": False, "error": "Подтверждение: введите слово DELETE."},
            status_code=400,
        )
    try:
        async with _exclusive_job("wipe"):
            result = await asyncio.to_thread(_pipeline_wipe_index, str(_config_path()))
    except _JobBusyError:
        return _busy_response()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/clean-orphans")
async def ui_clean_orphans() -> JSONResponse:
    try:
        async with _exclusive_job("clean-orphans"):
            result = await asyncio.to_thread(_pipeline_clean_orphans, str(_config_path()))
    except _JobBusyError:
        return _busy_response()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.post("/ui/clear-incoming")
async def ui_clear_incoming() -> JSONResponse:
    try:
        async with _exclusive_job("clear-incoming"):
            result = await asyncio.to_thread(_pipeline_clear_incoming, str(_config_path()))
    except _JobBusyError:
        return _busy_response()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, **result})


@app.get("/mcp")
async def mcp_get(request: Request) -> Response:
    # SSE stream endpoint (optional per spec)
    origin = request.headers.get("origin")
    if not _origin_allowed(origin):
        return PlainTextResponse("Origin not allowed", status_code=403)

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
