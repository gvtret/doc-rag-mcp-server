# Web UI

The MCP HTTP server bundles a single-page Web UI at `/ui`. No build step, no JS
framework — it ships in the server response as plain HTML + a small `<script>` block.

```
http://<host>:3333/ui
```

If `DOC_RAG_API_KEY` is set, append `?key=<DOC_RAG_API_KEY>`.

## What it does

- **Upload documents** (`.pdf` / `.docx` / `.doc` / `.md` / `.txt`, multi-file).
- **Detect duplicates on upload** — files whose sha256 already lives in the manifest or
  the incoming queue are skipped; the UI shows a yellow banner listing them and why.
- **Run ingest / rebuild** asynchronously; status polls live (started → running →
  finished/error).
- **Browse indexed documents** in a table — doc_id, source file, chunks, parser used,
  OCR coverage, ingest timestamp.
- **Preview a document's markdown** with one click.
- **Delete documents** (per-row ✕ button, or bulk checkbox plus
  "Удалить выбранные" / "Delete selected").
- **Danger zone** — three additional buttons:
  - **"Удалить всё"** / **"Delete everything"** (`POST /ui/wipe`) —
    clears archived sources, build artefacts, manifest, and index.
    Requires explicit confirmation.
  - **Clean orphans** (`POST /ui/clean-orphans`) — drops md/chunks/vectors not in the
    manifest.
  - **Clear incoming** (`POST /ui/clear-incoming`) — empties `sources/incoming/`.

All write endpoints return `409 Busy` when ingest or rebuild is already running, so the
UI can't trigger overlapping mutations.

## Degraded-mode banner

When the configured retrieval mode is `semantic` but the FAISS index isn't ready
(missing, corrupted, or being rebuilt), the UI shows a persistent banner at the top
with a one-click **"Запустить rebuild"** / **"Start rebuild"** button. The banner
is tied to the same polling that updates document status, so it auto-clears the
moment the index is back.

This complements the MCP-side signal: a matching warning content-item is prepended to
`doc_search` responses so MCP clients also know quality is degraded. See
[mcp.md](mcp.md#tools-exposed).

## Config editor

The Svelte Configuration page has three tabs:

- **Форма** — a structured form over `config/config.yaml` (typed fields,
  per-field validation, hints). Backed by `GET /ui/config/parsed` (parsed
  config) and `POST /ui/config/patch` (field-level, **comment-preserving**
  writes via a `ruamel.yaml` round-trip — only the edited keys change).
- **Advanced (raw YAML)** — the original full-file editor, backed by
  `GET /ui/config/raw` + `POST /ui/config/save` (400 on YAML parse errors).
- **Сервис (env)** — runtime/service settings that live in the environment,
  not in `config.yaml`. Backed by `GET /ui/env` + `POST /ui/env/save`.

All three share a **Restart service** button (`POST /ui/restart`, only works
when sudoers grants `systemctl restart doc-rag-mcp` and the server has
`DOC_RAG_UI_RESTART_ENABLED=1` + `DOC_RAG_UI_RESTART_CMD`).

### Service env editor

`POST /ui/config/save`/`patch` edit the *pipeline* config. The **Сервис (env)**
tab edits the *service runtime* — bind host/port, CORS origins, HTTP log path,
rate limits, restart command, log level/format. These are written to a
UI-managed `<root>/.env` (override path with `DOC_RAG_ENV_FILE`) that
`scripts/run_mcp_http.sh` sources at startup, **overriding** systemd's
root-owned `/etc/default/doc-rag` (which the service user cannot write).
Changes apply on the next **service restart**.

`DOC_RAG_API_KEY` is a **secret**: the editor shows only whether it is set,
never its value, and refuses to write it — manage the key in env/systemd on
the server.

> **Note:** config and env writes only take effect on the **systemd** deploy.
> Under Docker, `config/` is mounted read-only and env is set in
> `docker-compose.yml`, so edit those there instead.

These are intended for trusted LAN/admin use. Keep the API key set in production.

## Endpoints summary

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/ui/upload` | multipart upload to `sources/incoming/`; reports duplicates |
| `POST` | `/ui/ingest` | run ingest in background |
| `POST` | `/ui/rebuild` | rebuild chunks/embeddings/FAISS in background |
| `POST` | `/ui/delete` | delete one or more `doc_id`s |
| `POST` | `/ui/wipe` | clear everything (manifest, build, archived) |
| `POST` | `/ui/clean-orphans` | drop unreferenced artefacts |
| `POST` | `/ui/clear-incoming` | empty `sources/incoming/` |
| `GET`  | `/ui/status` | JSON status for live polling |
| `GET`  | `/ui/document-preview` | rendered markdown for a `doc_id` |
| `GET`  | `/ui/config/parsed` | parsed `config.yaml` for the structured form |
| `POST` | `/ui/config/patch` | field-level, comment-preserving config write |
| `POST` | `/ui/config/validate` | validate field updates without writing |
| `GET`  | `/ui/env` | editable service env (secrets masked) |
| `POST` | `/ui/env/save` | write service env to `<root>/.env` |
| `GET`  | `/api/v1/manifest` | raw `build/manifest.json` |

In addition, the server exposes the standard probe and observability
endpoints described in [docs/deploy.md](deploy.md#observability):
`GET /health/live`, `GET /health/ready`, `GET /metrics`. The Web UI
itself does not use them — they exist for k8s/uptime monitors and
Prometheus scrapers.
