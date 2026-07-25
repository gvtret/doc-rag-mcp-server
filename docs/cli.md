# CLI

All commands run via the `doc-rag` console script (installed by `pip install -e .`).
Add `--config <path>` to point at a non-default config; defaults to `config/config.yaml`.

## Supported file formats

| Extension | Default parser | Optional alternative | Notes |
| --- | --- | --- | --- |
| `.pdf` | Docling | — | structure-aware (tables, headings, formulas, multi-column); OCR via RapidOCR built in |
| `.docx` | `python-docx` | Docling (`docx_backend: docling`) | python-docx is fast; Docling is structure-aware but much slower |
| `.doc` | `antiword` (preferred) or `catdoc` | — | requires the system tool on PATH |
| `.md` | direct read | — | blank lines kept (semantic) |
| `.txt` | direct read | — | blank lines kept |

Drop files into `sources/incoming/` (subfolders are preserved). v1.5+
routes parsers by file **content** (magic bytes via `filetype`), not by
the filename extension — so a misnamed `report.pdf` that is really a
DOCX is parsed correctly.

Since v2.0 PDF parsing always goes through Docling — PyMuPDF and PyPDF2
have been removed. The first PDF parse downloads ~300 MB of ML model
weights into `~/.cache/docling/`; later parses are offline. Per-document
wall time is ~10–20 s/page on a modern CPU, so schedule large ingests
as cron jobs rather than as synchronous user operations.

---

## `doc-rag ingest`

Incremental ingest from `sources/incoming/`.

```bash
doc-rag ingest
```

Pipeline:

1. Parse documents (PDF / DOCX / DOC / MD / TXT)
2. Normalize text and split by sections
3. Export Markdown → `build/docs_md/`
4. Extract tables → `build/tables_json/`
5. Chunk text → `build/chunks_jsonl/chunks.jsonl`
6. Build embeddings → `build/embeddings/`
7. Build FAISS index → `build/index/faiss.index`
8. Archive source files → `sources/archived/`

**Incremental behavior:** files whose sha256 is already in `build/manifest.json` are
skipped (still archived/deduped). To force full overwrite, set
`sources.incremental_ingest: false`.

## `doc-rag rebuild`

Rebuild chunks → embeddings → FAISS from `build/docs_md/` (skips parsing).
Use when you change chunk size, embedding model, or recover from a corrupt index.

```bash
doc-rag rebuild
```

Flags:

- `--force` — full rebuild: wipe `docs_md`/chunks and reparse everything (default is
  incremental).
- `--skip-index` — parse and chunk only, skip building the vector index. For local GPU
  processing: run the heavy embedding step elsewhere, then `rebuild-index` on the server.

If the MCP server is already running, `rebuild` (without `--skip-index`) is sent through
`POST /ui/rebuild` instead of running in-process, so progress is visible in the Web UI.

## `doc-rag rebuild-docs <doc_id> [<doc_id> ...]`

Rebuild specific documents by `doc_id` — reparse, rechunk, and re-embed just those
documents. Note: the vector index itself is still rebuilt in full from all chunks, not
incrementally, so this is not necessarily cheaper than `rebuild` for a large corpus.

```bash
doc-rag rebuild-docs a3f1b9c2 7d4e0a55
```

## `doc-rag rebuild-index`

Rebuild only the FAISS index from the existing `build/chunks_jsonl/chunks.jsonl`
(no reparsing, no rechunking). Use after `rebuild --skip-index` or to recover a corrupt
index without re-running embeddings.

```bash
doc-rag rebuild-index
```

## `doc-rag serve`

Legacy debug HTTP server (separate from the MCP server). Most users want
`bash scripts/run_mcp_http.sh` instead — that one serves MCP, the UI, and `/health`.

## `doc-rag delete <doc_id> [<doc_id> ...]`

Remove one or more documents and all derived artefacts (source, md, chunks, vectors,
manifest entry). FAISS vectors of the deleted chunks are dropped *without re-encoding*:
the remaining vectors are reconstructed from the existing index and rewritten in place.

```bash
doc-rag delete a3f1b9c2 7d4e0a55
```

Output is JSON: removed sources, chunks, vectors, kept counts.

Find `doc_id`s in `build/manifest.json` or via the Web UI document table.

## `doc-rag wipe --confirm DELETE`

Nuclear option: clear `sources/archived/`, `build/*`, manifest, and the FAISS index.
Won't run without the literal flag `--confirm DELETE`.

```bash
doc-rag wipe --confirm DELETE
```

## `doc-rag clean-orphans`

Drop md / chunks / vectors files that aren't referenced by the current manifest.
Useful after a failed ingest left dangling artefacts.

```bash
doc-rag clean-orphans
```

## `doc-rag clear-incoming`

Delete every file under `sources/incoming/` (does not touch the index).

```bash
doc-rag clear-incoming
```

## `doc-rag migrate-vectors --to <backend> [--from <backend>] [--collection <name>]`

Migrate vectors between vector-store backends (e.g. `faiss` → `qdrant` or `pgvector`).
`--from` defaults to `faiss`; `--to` is required.

```bash
doc-rag migrate-vectors --to qdrant --collection default
```

## `doc-rag migrate`

Bring `build/manifest.json` up to the schema version this build of
`doc-rag` understands. Currently a stub: there are no migrations
defined, so the command always reports the supported and detected
schema versions and exits cleanly. The CLI surface exists so future
schema bumps have a stable place to land.

```bash
doc-rag migrate
```

Sample output:

```json
{
  "supported_schema_version": 1,
  "found_schema_version": 1,
  "migrations_applied": [],
  "message": "no migrations defined for this version"
}
```

If `doc-rag` refuses to operate on a manifest because of a higher
schema version than it knows about, `migrate` is the path forward —
see also `docs/troubleshooting.md` for the matching error message.

---

All destructive commands print a JSON summary so they can be piped into other tools.
The Web UI exposes the same operations as buttons — see [ui.md](ui.md).
