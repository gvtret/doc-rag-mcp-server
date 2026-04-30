from __future__ import annotations

"""MCP server over Streamable HTTP transport.

Implements a single MCP endpoint (default: /mcp) that supports:
- HTTP POST: JSON-RPC messages (application/json response)
- HTTP GET : optional SSE stream (text/event-stream) for server -> client notifications/requests

This follows the MCP Streamable HTTP transport spec (2025-03-26+).
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Request, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse

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
_INGEST_STATE: Dict[str, Any] = {"running": False, "last_started": None, "last_finished": None, "last_ok": None, "last_error": None}


async def _run_ingest_background() -> None:
  async with _INGEST_LOCK:
    if _INGEST_STATE.get("running"):
      return
    _INGEST_STATE["running"] = True
    _INGEST_STATE["last_started"] = time.time()
    _INGEST_STATE["last_error"] = None
    _INGEST_STATE["last_ok"] = None

  def _do():
    from doc_rag.raglib.pipeline import ingest
    ingest(str(_config_path()))

  try:
    await asyncio.to_thread(_do)
    ok = True
    err = None
  except Exception as exc:
    ok = False
    err = str(exc)

  async with _INGEST_LOCK:
    _INGEST_STATE["running"] = False
    _INGEST_STATE["last_finished"] = time.time()
    _INGEST_STATE["last_ok"] = ok
    _INGEST_STATE["last_error"] = err


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
  """Write a single log line to stderr and optional file."""
  ts = time.strftime("%Y-%m-%d %H:%M:%S")
  msg = f"[doc-rag][http] {ts} {line}"
  print(msg, file=sys.stderr, flush=True)
  log_path = os.environ.get("DOC_RAG_HTTP_LOG", "")
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
    _log_line(f"{request.method} {request.url.path} -> {code} ({dur_ms}ms)")


@app.get("/health")
async def health() -> JSONResponse:
  return JSONResponse({"status": "ok", "name": "doc-rag", "version": "1.0"})


@app.get("/ui")
async def ui(request: Request, key: str = "") -> HTMLResponse:
  if not _ui_key_ok(request, key):
    return HTMLResponse(
      "<h3>Unauthorized</h3><p>Append <code>?key=...</code> to URL or send Authorization header.</p>",
      status_code=401,
    )

  state = dict(_INGEST_STATE)
  incoming = _sources_incoming_dir()
  incoming.mkdir(parents=True, exist_ok=True)
  try:
    files = sorted([p.name for p in incoming.iterdir() if p.is_file()])[:200]
  except Exception:
    files = []

  key_q = f"?key={key}" if key else ""
  base = _public_base_url(request)
  mcp_url = f"{base}/mcp"
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
        .muted {{ color:#6b7280; }}
        button {{ padding: 8px 12px; border-radius: 10px; border: 1px solid #111827; background:#111827; color:white; cursor:pointer; }}
        button:disabled {{ opacity:0.5; cursor:not-allowed; }}
      </style>
    </head>
    <body>
      <h2>doc-rag — управление индексом</h2>
      <p class="muted">LAN UI. MCP endpoint: <code>/mcp</code></p>

      <div class="row">
        <div class="card">
          <h3>Загрузка документов</h3>
          <form action="/ui/upload{key_q}" method="post" enctype="multipart/form-data">
            <input type="hidden" name="key" value="{key}"/>
            <input type="file" name="file" accept=".pdf,.docx" required />
            <div style="height: 12px;"></div>
            <button type="submit">Upload → sources/incoming</button>
          </form>
          <p class="muted">Поддерживаются: PDF, DOCX.</p>
        </div>

        <div class="card">
          <h3>Ingest</h3>
          <form action="/ui/ingest{key_q}" method="post">
            <input type="hidden" name="key" value="{key}"/>
            <button type="submit" {"disabled" if state.get("running") else ""}>Запустить ingest</button>
          </form>
          <p class="muted">Статус: <code>{'running' if state.get('running') else 'idle'}</code></p>
          <p class="muted">Последний результат: <code>{state.get('last_ok')}</code></p>
          <p class="muted">Ошибка: <code>{(state.get('last_error') or '-')}</code></p>
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
  return JSONResponse(dict(_INGEST_STATE))


@app.post("/ui/upload")
async def ui_upload(
  request: Request,
  file: UploadFile = File(...),
  key: str = Form(""),
) -> Response:
  if not _ui_key_ok(request, key):
    return PlainTextResponse("Unauthorized", status_code=401)

  name = _sanitize_filename(file.filename or "")
  ext = os.path.splitext(name.lower())[1]
  if ext not in (".pdf", ".docx"):
    return PlainTextResponse("Only .pdf or .docx allowed", status_code=400)

  incoming = _sources_incoming_dir()
  incoming.mkdir(parents=True, exist_ok=True)
  dst = incoming / name
  # Avoid overwrite
  if dst.exists():
    stem = dst.stem
    dst = incoming / f"{stem}__{int(time.time())}{dst.suffix}"

  max_mb = int(os.environ.get("DOC_RAG_UI_MAX_UPLOAD_MB", "100"))
  limit = max(1, max_mb) * 1024 * 1024
  written = 0
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
        return PlainTextResponse("Upload too large", status_code=413)
      out.write(chunk)

  return RedirectResponse(url=f"/ui?key={key}", status_code=303)


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
