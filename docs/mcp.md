# MCP integration (Cursor / Claude / other clients)

`doc-rag` ships an HTTP MCP server speaking the **Streamable HTTP** transport on a single
endpoint. One process serves MCP, the Web UI, and `/health` from port 3333.

## Endpoints

| Path | Purpose |
| --- | --- |
| `POST /mcp` | JSON-RPC request/response (`application/json`) |
| `GET  /mcp` | SSE stream (`text/event-stream`) — `doc_rag/ready` notification + keepalives |
| `GET  /health` | `{"status":"ok",...}` health check |
| `GET  /ui` | Web UI |
| `GET  /ui/mcp/cursor.json` | drop-in MCP config for Cursor |
| `GET  /ui/mcp/vscode.json` | drop-in MCP config for VS Code |
| `GET  /health/live` | liveness probe — always 200 |
| `GET  /health/ready` | readiness probe — 503 if no manifest or job in flight |
| `GET  /metrics` | Prometheus text exposition (requires `[metrics]` extra) |

## Tools exposed

- **`doc_search`** — semantic + lexical search over the indexed corpus.
  Arguments: `query: str` (required), `top_k: int` (default 6, capped at 50).
  Returns ranked chunks with `score`, `doc_id`, `chunk_id`, `source_file`, `text`.

If the FAISS index isn't available (e.g. corpus was wiped or rebuild is in progress),
`doc_search` falls back to lexical search and prepends a warning content-item to the
response telling the client that quality is degraded and pointing them at the
"Rebuild индекса" / "Rebuild index" button in the UI. See
[ui.md](ui.md#degraded-mode-banner) for the matching UI signal.

## API stability (SemVer)

The MCP surface of `doc-rag` is SemVer-protected per
[docs/roadmap.md § 1](roadmap.md). Concretely, for the v1.x release line:

- **The list of exposed tools is fixed at `[doc_search]`.** Adding a
  new tool is a MINOR bump. Removing or renaming a tool is a MAJOR
  bump.
- **`doc_search` argument schema is stable.** `query: str` (required)
  and `top_k: int` (default 6, capped at 50) — adding optional fields
  is a MINOR bump, removing or renaming is MAJOR. The capping rule
  itself is also part of the contract.
- **Response shape is stable.** Each result item carries `score`
  (float), `doc_id` (str), `chunk_id` (str), `source_file` (str), and
  `text` (str). Adding new fields is a MINOR bump; removing or
  renaming is MAJOR.
- **Degraded-mode warning content-item is part of the contract.**
  When the FAISS index is missing, the first `content` element is a
  `{"type": "text", "text": "<warning>"}` entry. The warning string
  itself is human-readable and may change wording in patch releases —
  do not parse it; check `cat["semantic_search_ready"]` via
  `/ui/status` if you need programmatic detection.
- **HTTP transport details** (`POST /mcp`, `GET /mcp` SSE) are
  governed by the upstream Streamable HTTP MCP specification, not by
  us. We commit to following whatever the published spec says for the
  protocol version we advertise in `initialize`.

What is **not** SemVer-protected:

- Internal Python module structure (`src/doc_rag/server/*`).
- Log line formats. Structured-log shape is governed by its own
  `schema_version` field (see [docs/deploy.md](deploy.md#observability)).
- The Web UI HTML markup. Routes under `/ui/*` are stable; the markup
  they return is not.

If you build an MCP client against `doc-rag` v1.x, the above is what
you can rely on through every v1.y.z release.

## Run the server

Local:

```bash
bash scripts/run_mcp_http.sh
# default: http://127.0.0.1:3333/mcp
```

If `config/` or `build/` lives outside the installed package tree, set
`DOC_RAG_ROOT` to the repository root.

Logging to file:

```bash
export DOC_RAG_HTTP_LOG="$PWD/build/http.log"
bash scripts/run_mcp_http.sh
```

## Manual checks

```bash
# initialize
curl -sS -X POST http://127.0.0.1:3333/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# tools/list
curl -sS -X POST http://127.0.0.1:3333/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# SSE stream
curl -N -H 'Accept: text/event-stream' http://127.0.0.1:3333/mcp
```

The bundled helper does both initialize and tools/list:

```bash
bash scripts/verify_mcp.sh
```

## Cursor

Two ways to register the server.

### Project-local

Copy `src/doc_rag/server/mcp_cursor_http.json` into `.cursor/mcp.json` at the project
root (or merge under `mcpServers`). Restart Cursor.

### Global

When the Agent doesn't see `doc_search` despite the server being listed as enabled in
"Installed MCP Servers" (a known Cursor quirk), promote the config to your home dir:

```bash
bash scripts/print_global_mcp_config.sh
# or, on WSL/Windows if the script emits CRLF noise:
python3 scripts/write_global_mcp_config.py
```

Paste the printed JSON into `~/.cursor/mcp.json`, restart Cursor, open a **new** chat.

### Remote server

```json
{
  "mcpServers": {
    "doc-rag-remote": {
      "transport": "streamableHttp",
      "url": "http://192.168.1.118:3333/mcp"
    }
  }
}
```

If your client sends an `Origin` header, allow it via `DOC_RAG_ALLOWED_ORIGINS` (CSV).

## Auth (optional)

LAN deployments can run open by default. To require an API key:

```bash
export DOC_RAG_API_KEY="change-me"
bash scripts/run_mcp_http.sh
```

Clients must send `Authorization: Bearer <key>` or `X-Api-Key: <key>` for `/mcp` and
`/ui`. The UI accepts `?key=<DOC_RAG_API_KEY>` in the URL as a convenience.

## Rate limit (optional)

Per-client token-bucket:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOC_RAG_RATE_LIMIT_RPS` | `0` (off) | sustained requests per second |
| `DOC_RAG_RATE_LIMIT_BURST` | `5` | burst capacity |
