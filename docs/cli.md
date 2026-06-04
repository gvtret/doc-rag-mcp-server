# CLI

All commands run via the `doc-rag` console script (installed by `pip install -e .`).
Add `--config <path>` to point at a non-default config; defaults to `config/config.yaml`.

## Supported file formats

| Extension | Parser | Notes |
| --- | --- | --- |
| `.pdf` | PyMuPDF (with PyPDF2 fallback) | OCR for scanned pages when Tesseract is available |
| `.docx` | `python-docx` | tables and headings preserved |
| `.doc` | `antiword` (preferred) or `catdoc` | requires the system tool on PATH |
| `.md` | direct read | blank lines kept (semantic) |
| `.txt` | direct read | blank lines kept |

Drop files into `sources/incoming/` (subfolders are preserved).

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
