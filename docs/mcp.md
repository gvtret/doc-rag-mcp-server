# MCP integration (Cursor / Claude / other clients)

`doc-rag` ships an HTTP MCP server speaking the **Streamable HTTP** transport on a single
endpoint. One process serves MCP, the Web UI, and `/health` from port 3333.

## Endpoints

| Path | Purpose |
| --- | --- |
| `POST /mcp` | JSON-RPC request/response (`application/json`) |
| `GET  /mcp` | SSE stream (`text/event-stream`) — `doc_rag/ready` notification + keepalives |
| `GET  /health` | `{"status":"ok",...}` health check |
| `GET  /ui` | Redirects to the Web UI at `/ui-next/` |
| `GET  /ui/mcp/cursor.json` | drop-in MCP config for Cursor |
| `GET  /ui/mcp/vscode.json` | drop-in MCP config for VS Code |
| `GET  /health/live` | liveness probe — always 200 |
| `GET  /health/ready` | readiness probe - 503 if no manifest, a job is in flight, or (semantic/hybrid) the model is missing (`model_missing`) / the index is stale (`index_stale`) |
| `GET  /metrics` | Prometheus text exposition (requires `[metrics]` extra) |

## Tools exposed

- **`doc_search`** — semantic + lexical search over the indexed corpus.
  Arguments: `query: str` (required), `top_k: int` (default 6, capped at 50),
  `namespace: str` (default `"default"`), `doc_id: str` (filter by document),
  `section_path: str` (filter by section-path prefix), `tables_only: bool`
  (default `false`, return only table chunks).
  Returns ranked chunks with `score`, `doc_id`, `chunk_id`, `source_file`, `text`.
- **`doc_download`** — download a source document by filename.
  Arguments: `filename: str` (required). Returns the file content base64-encoded.
- **`doc_search_docs`** — like `doc_search` but document-level: returns unique
  documents (not chunks) with relevance scores and download URLs.
  Arguments: `query: str` (required), `top_k: int` (default 10).

`doc_search` prepends a warning content-item when semantic results cannot be
trusted. Three cases:

- the embedding model did not load - the notice says the model is unavailable
  and points at `scripts/fetch_model.py`; lexical results are returned;
- the FAISS index is missing or not built - the notice points at the
  "Пересобрать индекс" / "Rebuild index" action; lexical results are returned;
- the index was built with different weights (`index_stale`) - the notice says
  the semantic results may be degraded; they are still returned (in hybrid mode
  merged with lexical ones).

See [ui.md](ui.md#degraded-mode-banner) for the matching UI signal.

## API stability (SemVer)

The MCP surface of `doc-rag` is SemVer-protected per
[docs/roadmap.md § 1](roadmap.md):

- **The list of exposed tools is currently `[doc_search, doc_download,
  doc_search_docs]`.** Adding a new tool is a MINOR bump. Removing or
  renaming a tool is a MAJOR bump.
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

If you build an MCP client against a given `doc-rag` major version, the above is
what you can rely on through every minor/patch release of that major version.

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

## No authentication

`doc-rag` has no auth mechanism — `/mcp` and every `/ui/*` route are open to anyone who
can reach the port. This is by design for a trusted-LAN deployment; do not expose this
port to an untrusted network. See [SECURITY.md](../SECURITY.md) for the full threat
model and hardening guidance.

## Rate limit (optional)

Per-client token-bucket:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOC_RAG_RATE_LIMIT_RPS` | `0` (off) | sustained requests per second |
| `DOC_RAG_RATE_LIMIT_BURST` | `5` | burst capacity |
