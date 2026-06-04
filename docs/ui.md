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

`/ui` also exposes:

- `GET  /ui/config/raw` — current `config/config.yaml` (auth-gated).
- `POST /ui/config/save` — write back. Returns 400 on YAML parse errors.
- `POST /ui/restart` — restart the systemd unit (only if sudoers grants
  `systemctl restart doc-rag-mcp`).

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
| `GET`  | `/api/v1/manifest` | raw `build/manifest.json` |
