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

## Tools exposed

- **`doc_search`** — semantic + lexical search over the indexed corpus.
  Arguments: `query: str` (required), `top_k: int` (default 6, capped at 50).
  Returns ranked chunks with `score`, `doc_id`, `chunk_id`, `source_file`, `text`.

If the FAISS index isn't available (e.g. corpus was wiped or rebuild is in progress),
`doc_search` falls back to lexical search and prepends a warning content-item to the
response telling the client that quality is degraded and pointing them at the
"Rebuild индекса" button in the UI. See [ui.md](ui.md#degraded-mode-banner) for the
matching UI signal.

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
