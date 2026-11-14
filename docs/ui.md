# Web UI

The MCP HTTP server bundles a Svelte + Vite single-page app, built to `ui/dist` and
served at `/ui-next/` (bare `GET /ui` redirects there). Building it is a hard
requirement — `bootstrap.sh` / `install_server_native.sh` fail the install if
Node ≥ 20 isn't available, since there is no fallback.

```
http://<host>:3333/ui
```

## What it does

- **Upload documents** (`.pdf` / `.docx` / `.doc` / `.md` / `.txt`, multi-file).
- **Detect duplicates on upload** — files whose sha256 already lives in the manifest or
  the incoming queue are skipped; the UI shows a yellow banner listing them and why.
- **Run ingest / rebuild** asynchronously; status polls live (started → running →
  finished/error).
- **Model / index banners** on the Documents page - ERROR when the
  embedding model is missing (`model_ready: false`), WARN when the index
  is stale (`index_stale`, weights fingerprint mismatch -> rebuild
  needed), INFO when a new weights revision is available on the HF Hub
  (`model_update_available`), all from `GET /ui/status`.
- **Browse indexed documents** in a table — doc_id, source file, chunks, parser used,
  ingest timestamp, quality badge (green/yellow/red, from `GET /ui/quality`), OCR badge
  (shown when RapidOCR fired on the document; tooltip has pages recognized + mean
  confidence, from `coverage.ocr` in `GET /ui/documents`).
- **Preview a document's markdown** with one click, or **download the source file**
  (`GET /api/v1/download`).
- **Document statistics page** (`GET /ui/doc-stats`) — per-document block-type
  breakdown (heading/paragraph/table/figure/...) as stacked bars, expandable detail
  rows, sortable columns.
- **Selective rebuild** — checkboxes on the Documents page, "Rebuild selected" runs
  `POST /ui/rebuild-docs` for just those documents. Separate "index only"
  (`POST /ui/rebuild-index`) and "skip index" (`POST /ui/rebuild-skip-index`) modes
  support the GPU-offload workflow: parse+chunk on the server, embed elsewhere, build
  the index back on the server.
- **Logs page** — live tail of the ingest/rebuild log and the HTTP access log (reads
  the `log_tail`/`http_log_tail` fields from `GET /ui/status`).
- **RU/EN language switch** in the header (localStorage-persisted).
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
  A key is patchable from the form only when it is present in
  `config.yaml`; the server additionally validates patched values
  against its field-constraint list. Everything else (e.g.
  `embeddings.model_path`) goes through the Advanced tab. The
  **Эмбеддинги** section exposes `update_check_interval_hours` (int,
  min 0; 0 disables the read-only HF Hub weight check).
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

> **Note:** config and env writes only take effect on the **systemd** deploy.
> Under Docker, `config/` is mounted read-only and env is set in
> `docker-compose.yml`, so edit those there instead.

These editors have no authentication of their own — see
[SECURITY.md](../SECURITY.md) for the server's overall (lack of) auth model.

## Endpoints summary

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/ui/upload` | multipart upload to `sources/incoming/`; reports duplicates |
| `POST` | `/ui/ingest` | run ingest in background |
| `POST` | `/ui/rebuild` | rebuild chunks/embeddings/FAISS in background |
| `POST` | `/ui/rebuild-docs` | rebuild specific documents (selective rebuild) |
| `POST` | `/ui/rebuild-index` | rebuild only the FAISS index from existing chunks |
| `POST` | `/ui/rebuild-skip-index` | parse+chunk only, skip the vector index |
| `POST` | `/ui/delete` | delete one or more `doc_id`s |
| `POST` | `/ui/wipe` | clear everything (manifest, build, archived) |
| `POST` | `/ui/clean-orphans` | drop unreferenced artefacts |
| `POST` | `/ui/clear-incoming` | empty `sources/incoming/` |
| `POST` | `/ui/restart` | restart the service (requires `DOC_RAG_UI_RESTART_ENABLED`) |
| `GET`  | `/ui/status` | JSON status for live polling |
| `GET`  | `/ui/documents` | indexed + pending documents, cross-referenced with the manifest |
| `GET`  | `/ui/document-preview` | rendered markdown for a `doc_id` |
| `GET`  | `/ui/doc-stats` | per-document block-type statistics |
| `GET`  | `/ui/quality` | quality summary (`build/quality/summary.json`) |
| `GET`  | `/api/v1/download` | download a source document by filename |
| `GET`  | `/ui/config/raw` | raw `config.yaml` text for the Advanced editor |
| `POST` | `/ui/config/save` | write raw `config.yaml` text |
| `GET`  | `/ui/config/parsed` | parsed `config.yaml` for the structured form |
| `POST` | `/ui/config/patch` | field-level, comment-preserving config write |
| `POST` | `/ui/config/validate` | validate field updates without writing |
| `GET`  | `/ui/env` | editable service env (secrets masked) |
| `POST` | `/ui/env/save` | write service env to `<root>/.env` |
| `GET`  | `/ui/mcp/cursor.json` | drop-in MCP config for Cursor |
| `GET`  | `/ui/mcp/vscode.json` | drop-in MCP config for VS Code |
| `GET`  | `/api/v1/manifest` | raw `build/manifest.json` |

In addition, the server exposes the standard probe and observability
endpoints described in [docs/deploy.md](deploy.md#observability):
`GET /health/live`, `GET /health/ready`, `GET /metrics`. The Web UI
itself does not use them — they exist for k8s/uptime monitors and
Prometheus scrapers.
