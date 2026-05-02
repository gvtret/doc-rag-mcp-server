from __future__ import annotations

"""MCP server over Streamable HTTP transport.

Implements a single MCP endpoint (default: /mcp) that supports:
- HTTP POST: JSON-RPC messages (application/json response)
- HTTP GET : optional SSE stream (text/event-stream) for server -> client notifications/requests

This follows the MCP Streamable HTTP transport spec (2025-03-26+).
"""

import asyncio
import html
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

from fastapi import FastAPI, Request, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse

from doc_rag.server.retrieval import document_preview, indexed_catalog

Json = Union[Dict[str, Any], List[Any]]

_TOOL_SEM = asyncio.Semaphore(int(os.environ.get("DOC_RAG_MAX_CONCURRENCY", "4")))

_RL_LOCK = asyncio.Lock()
_RL_STATE: Dict[str, Tuple[float, float]] = {}


def _rate_limit_params() -> Tuple[float, float]:
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


def _api_key_required() -> Optional[str]:
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


async def _save_upload_to_path(file: UploadFile, dst: Path) -> Optional[str]:
  """Write upload to dst. Returns error message or None."""
  max_mb = int(os.environ.get("DOC_RAG_UI_MAX_UPLOAD_MB", "100"))
  limit = max(1, max_mb) * 1024 * 1024
  written = 0
  try:
    with open(dst, "wb") as out:
      while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
          break
        written += len(chunk)
        if written > limit:
          out.close()
          try:
            dst.unlink(missing_ok=True)
          except Exception:
            pass
          return f"{file.filename or dst.name}: файл больше лимита ({max_mb} MiB)"
        out.write(chunk)
  except Exception as e:
    try:
      dst.unlink(missing_ok=True)
    except Exception:
      pass
    return f"{file.filename or dst.name}: {e}"
  return None


async def _save_upload_to_incoming(file: UploadFile, incoming: Path) -> Optional[str]:
  name = _sanitize_filename(file.filename or "")
  ext = os.path.splitext(name.lower())[1]
  if ext not in (".pdf", ".docx"):
    lab = name or (file.filename or "unnamed")
    return f"{lab}: только .pdf или .docx"
  dst = _incoming_unique_path(incoming, name)
  return await _save_upload_to_path(file, dst)


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


def _mcp_config_payload(name: str, url: str, api_key: str) -> Dict[str, Any]:
  cfg: Dict[str, Any] = {
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


def _download_json(payload: Dict[str, Any], filename: str) -> Response:
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
  _INGEST_UI_LOG_POLL_MS = max(250, min(600_000, int(os.environ.get("DOC_RAG_UI_INGEST_POLL_MS", "2000"))))
except Exception:
  _INGEST_UI_LOG_POLL_MS = 2000


def _server_http_ui_log_max_lines() -> int:
  try:
    n = int(os.environ.get("DOC_RAG_UI_HTTP_LOG_MAX_LINES", "500"))
  except Exception:
    n = 500
  return max(50, min(5000, n))


_INGEST_LOG_LOCK = threading.Lock()
_INGEST_UI_LOG_LINES: Deque[str] = deque(maxlen=_ingest_ui_log_max_lines())
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_INGEST_STATE: Dict[str, Any] = {
  "running": False,
  "job": None,
  "last_started": None,
  "last_finished": None,
  "last_ok": None,
  "last_error": None,
}


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


def _snapshot_ingest_ui_log_tail() -> List[str]:
  with _INGEST_LOG_LOCK:
    return list(_INGEST_UI_LOG_LINES)


_SERVER_HTTP_LOG_LOCK = threading.Lock()
_SERVER_HTTP_UI_LOG_LINES: Deque[str] = deque(maxlen=_server_http_ui_log_max_lines())


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


def _snapshot_server_http_ui_log_tail() -> List[str]:
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
      self._emit_line(f"[doc-rag][ui-log] … промежуточный вывод stderr обрезан ({len(scr)} симв.) …")
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
    if ok:
      _append_ingest_ui_line(f"[doc-rag][INFO] {job} завершился успешно.", ansi_strip=False)
    elif err:
      _append_ingest_ui_line(f"[doc-rag][ERROR] {job} завершился с ошибкой: {err}", ansi_strip=False)


async def _run_ingest_background() -> None:
  await _run_index_job_background("ingest", _sync_run_ingest_with_ui_logging)


async def _run_rebuild_background() -> None:
  await _run_index_job_background("rebuild", _sync_run_rebuild_with_ui_logging)


def _origin_allowed(origin: Optional[str]) -> bool:
  """Basic DNS-rebinding mitigation (configurable).

  By default:
  - If Origin header is absent: allow (native clients often omit it)
  - If Origin is present: allow only if it matches DOC_RAG_ALLOWED_ORIGINS (comma-separated)
  """
  if not origin:
    return True
  allowed = [o.strip() for o in os.environ.get("DOC_RAG_ALLOWED_ORIGINS", "").split(",") if o.strip()]
  if not allowed:
    # Be conservative: if Origin is present but allow-list is empty, deny.
    return False
  return origin in allowed


def _accepts_sse(request: Request) -> bool:
  accept = request.headers.get("accept", "")
  return "text/event-stream" in accept.lower()


def _is_notification(req: Dict[str, Any]) -> bool:
  # JSON-RPC: notifications are requests without an id, or with id=null
  return ("id" not in req) or (req.get("id") is None)


def _ok(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
  return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> Dict[str, Any]:
  payload: Dict[str, Any] = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
  # For errors to requests, include id (can be null)
  payload["id"] = req_id
  return payload


def _log_line(line: str) -> None:
  """Write a single log line to stderr, optional file, and UI ring buffer."""
  ts = time.strftime("%Y-%m-%d %H:%M:%S")
  msg = f"[doc-rag][http] {ts} {line}"
  print(msg, file=sys.stderr, flush=True)
  _append_server_http_ui_line(msg)
  log_path = os.environ.get("DOC_RAG_HTTP_LOG", "").strip()
  if log_path:
    try:
      with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(msg + "\n")
    except Exception:
      # Don't crash on logging issues
      pass


@dataclass
class _SseClient:
  queue: "asyncio.Queue[str]"
  created_ts: float


class _SseHub:
  """Very small SSE hub for server->client notifications."""

  def __init__(self) -> None:
    self._clients: List[_SseClient] = []
    self._lock = asyncio.Lock()

  async def add(self) -> _SseClient:
    client = _SseClient(queue=asyncio.Queue(), created_ts=time.time())
    async with self._lock:
      self._clients.append(client)
    return client

  async def remove(self, client: _SseClient) -> None:
    async with self._lock:
      self._clients = [c for c in self._clients if c is not client]

  async def broadcast_jsonrpc(self, msg: Dict[str, Any]) -> None:
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

def _sse_frame(msg: Dict[str, Any]) -> str:
  payload = json.dumps(msg, ensure_ascii=False).replace("\n", "\\n")
  return f"data: {payload}\n\n"


def _handle_one(req: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
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
      "serverInfo": {"name": "doc-rag", "version": "1.0"},
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
      return 200, _ok(req_id, {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True})

    # Lazy import to keep base install lightweight
    from doc_rag.server.search_tool import doc_search_tool

    try:
      content = doc_search_tool(arguments)
      return 200, _ok(req_id, {"content": content, "isError": False})
    except Exception as exc:
      return 200, _ok(req_id, {"content": [{"type": "text", "text": f"doc_search failed: {exc}"}], "isError": True})

  # Default: method not found
  if _is_notification(req):
    return 202, None
  return 200, _err(req_id, -32601, f"Method not found: {method}")


def _handle_jsonrpc(payload: Json) -> Tuple[int, Optional[Json]]:
  """Handle a JSON-RPC object or batch."""
  if isinstance(payload, list):
    responses: List[Dict[str, Any]] = []
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


app = FastAPI(title="doc-rag MCP HTTP", version="1.0")


@app.middleware("http")
async def _http_log_mw(request: Request, call_next):
  start = time.time()
  response: Optional[Response] = None
  try:
    client_host = getattr(getattr(request, "client", None), "host", None) or "unknown"
    # Don't rate-limit health checks (often polled).
    if request.url.path != "/health":
      if not await _rate_limit_allow(client_host):
        response = PlainTextResponse("Too Many Requests", status_code=429)
        return response
    response = await call_next(request)
    return response
  finally:
    dur_ms = int((time.time() - start) * 1000)
    code = getattr(response, "status_code", "ERR")
    path = request.url.path or ""
    # Avoid drowning the UI buffer with health checks and UI JSON polling.
    if path not in ("/health", "/ui/status", "/ui/document-preview"):
      _log_line(f"{request.method} {request.url.path} -> {code} ({dur_ms}ms)")


@app.get("/health")
async def health() -> JSONResponse:
  return JSONResponse({"status": "ok", "name": "doc-rag", "version": "1.0"})


def _ui_status_payload() -> Dict[str, Any]:
  out = dict(_INGEST_STATE)
  out["log_tail"] = _snapshot_ingest_ui_log_tail()
  out["http_log_tail"] = _snapshot_server_http_ui_log_tail()
  log_path = (os.environ.get("DOC_RAG_HTTP_LOG") or "").strip()
  out["http_log_file"] = log_path if log_path else None
  try:
    out["indexed"] = indexed_catalog()
  except Exception as exc:
    out["indexed"] = {"error": str(exc), "documents": [], "document_count": 0}
  return out


def _indexed_documents_table_rows_html(catalog: Dict[str, Any], max_rows: int = 300) -> Tuple[str, str]:
  """Return (tbody inner HTML, optional note HTML)."""
  docs = catalog.get("documents") if isinstance(catalog.get("documents"), list) else []
  note = ""
  shown = docs[:max_rows]
  if len(docs) > max_rows:
    note = f'<p class="muted">Показаны первые {max_rows} из {len(docs)} документов.</p>'
  rows: List[str] = []
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
    sh = d.get("sha256")
    sh_s = html.escape(str(sh)[:16] + "…" if sh and len(str(sh)) > 16 else (str(sh) if sh else "—"), quote=False)
    rows.append(
      f'<tr><td>{i}</td><td title="{sf}"><button type="button" class="doc-preview-btn" data-doc-id="{did_attr}">{bn_disp}</button></td>'
      f'<td class="doc-id">{did_cell}</td><td>{cc_s}</td><td class="muted">{sh_s}</td></tr>'
    )
  return ("\n".join(rows), note)


def _indexed_documents_summary_html(catalog: Dict[str, Any]) -> str:
  if catalog.get("error"):
    return f'<p class="muted">Не удалось прочитать индекс: <code>{html.escape(str(catalog.get("error")), quote=False)}</code></p>'
  n = int(catalog.get("document_count") or 0)
  mg = catalog.get("manifest_generated_at_utc")
  mg_s = html.escape(str(mg), quote=False) if mg else "—"
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
  um = (qp.get("up_msg") or "").strip()
  if us > 0 or ue > 0:
    hint = ""
    if um:
      hint = f" Пример: «{html.escape(um)}»"
    upload_banner = (
      f'<div class="upload-banner">Загружено файлов: {us}. Ошибок: {ue}.{hint}</div>'
    )

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
    http_log_file_note = f"<p class=\"muted\">Тот же текст дублируется в файл: <code>{escaped_hl}</code> (<code>DOC_RAG_HTTP_LOG</code>).</p>"
  base = _public_base_url(request)
  mcp_url = f"{base}/mcp"
  try:
    ic0 = indexed_catalog()
  except Exception as exc:
    ic0 = {"error": str(exc), "documents": [], "document_count": 0}
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
        table.idx {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        table.idx th, table.idx td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
        table.idx th {{ background: #f9fafb; font-weight: 600; }}
        table.idx td.doc-id {{ font-size: 11px; color: #6b7280; word-break: break-all; max-width: min(280px, 28vw); }}
      </style>
    </head>
    <body>
      <h2>doc-rag — управление индексом</h2>
      <p class="muted">LAN UI. MCP endpoint: <code>/mcp</code></p>

      {upload_banner}

      <div class="row">
        <div class="card">
          <h3>Загрузка документов</h3>
          <form action="/ui/upload{key_q}" method="post" enctype="multipart/form-data">
            <input type="hidden" name="key" value="{key}"/>
            <input type="file" name="files" accept=".pdf,.docx,.PDF,.DOCX" multiple required />
            <div style="height: 12px;"></div>
            <button type="submit" {"disabled" if state.get("running") else ""}>Загрузить в sources/incoming</button>
          </form>
          <p class="muted">PDF, DOCX; можно выбрать несколько файлов сразу (лимит: env <code>DOC_RAG_UI_MAX_UPLOAD_FILES</code>, по умолчанию 48).</p>
        </div>

        <div class="card">
          <h3>Ingest / rebuild</h3>
          <form action="/ui/ingest{key_q}" method="post">
            <input type="hidden" name="key" value="{key}"/>
            <button type="submit" {"disabled" if state.get("running") else ""}>Запустить ingest</button>
          </form>
          <div style="height:12px;"></div>
          <form action="/ui/rebuild{key_q}" method="post">
            <input type="hidden" name="key" value="{key}"/>
            <button type="submit" class="secondary" onclick="return confirm('Полный rebuild очистит build/docs markdown и chunks, затем пересканирует archived и incoming. Продолжить?');" {"disabled" if state.get("running") else ""}>Rebuild индекса</button>
          </form>
          <p class="muted">Фоновая задача: <code id="ingest-running-badge">{(state.get('job') or ('busy')) if state.get('running') else 'idle'}</code></p>
          <p class="muted">Последний результат: <code id="ingest-last-ok">{state.get('last_ok')}</code></p>
          <p class="muted">Ошибка: <code id="ingest-last-error">{(state.get('last_error') or '-')}</code></p>
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
          <h3>Проиндексированные документы</h3>
          <div id="indexed-summary">{idx_summary_html}</div>
          <div id="indexed-cap-note">{idx_cap_html}</div>
          <div style="overflow:auto; max-height: min(55vh, 520px); border: 1px solid #e5e7eb; border-radius: 10px;">
            <table class="idx" id="indexed-table">
              <thead>
                <tr>
                  <th style="width:36px;">#</th>
                  <th>Файл</th>
                  <th>doc_id</th>
                  <th style="width:88px;">Чанков</th>
                  <th style="width:120px;">SHA256</th>
                </tr>
              </thead>
              <tbody id="indexed-tbody">{idx_tbody_html}</tbody>
            </table>
          </div>
          <p class="muted" style="margin-top:10px;">Список берётся из <code>build/manifest.json</code> после ingest/rebuild. После завершения задачи таблица обновится автоматически.</p>
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
          var bits = [
            'Записей в <code>manifest</code>: <strong>' + n + '</strong>.',
            'Файл manifest: <strong>' + (idx.manifest_present ? 'есть' : 'нет') + '</strong>.',
            '<code>chunks.jsonl</code>: <strong>' + (idx.chunks_jsonl_present ? 'есть' : 'нет') + '</strong>.',
            'Векторный индекс (FAISS): <strong>' + (idx.semantic_index_present ? 'есть' : 'нет') + '</strong>.',
            'Лексический поиск (doc_search): <strong>' + (idx.lexical_search_ready ? 'готов' : 'не готов') + '</strong>.',
            'Семантический поиск: <strong>' + (idx.semantic_search_ready ? 'готов' : 'не готов') + '</strong>.',
            'Время генерации manifest (UTC): ' + mg + '.'
          ];
          sumEl.innerHTML = '<p class="muted">' + bits.join(' ') + '</p>';
          var docs = idx.documents || [];
          var maxR = 300;
          var slice = docs.slice(0, maxR);
          if (capEl) {{
            capEl.innerHTML = docs.length > maxR ? '<p class="muted">Показаны первые ' + maxR + ' из ' + docs.length + ' документов.</p>' : '';
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
            var sh = d.sha256 ? String(d.sha256) : '';
            var shDisp = sh.length > 16 ? esc(sh.slice(0, 16)) + '…' : esc(sh || '—');
            return '<tr><td>' + (i + 1) + '</td><td title="' + sf + '">' + nameCell + '</td><td class="doc-id">' + didCell + '</td><td>' + cc + '</td><td class="muted">' + shDisp + '</td></tr>';
          }}).join('');
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
              var lk = document.getElementById("ingest-last-ok");
              if (lk) lk.textContent = (j.last_ok === null || j.last_ok === undefined) ? "-" : String(j.last_ok);
              var le = document.getElementById("ingest-last-error");
              if (le) le.textContent = (j.last_error != null && String(j.last_error).length) ? String(j.last_error) : "-";
              renderIndexed(j.indexed);
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
  files: Annotated[List[UploadFile], File()],
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

  errors: List[str] = []
  saved = 0
  for uf in files:
    msg = await _save_upload_to_incoming(uf, incoming)
    if msg:
      errors.append(msg)
    else:
      saved += 1

  qdict: Dict[str, str] = {}
  k = (key or "").strip()
  if k:
    qdict["key"] = k
  if saved > 0:
    qdict["up_saved"] = str(saved)
  if errors:
    qdict["up_err"] = str(len(errors))
    qdict["up_msg"] = errors[0][:280]
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

  return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


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
      status, out = await asyncio.wait_for(asyncio.to_thread(_handle_jsonrpc, payload), timeout=timeout_sec)
    except asyncio.TimeoutError:
      return JSONResponse(_err(None, -32001, "Request timed out"), status_code=504)
  if out is None:
    return Response(status_code=status)

  # For now we return JSON. SSE streaming for POST is optional; GET provides notifications streaming.
  return JSONResponse(out, status_code=status)
