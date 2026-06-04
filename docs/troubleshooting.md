# Troubleshooting

## "manifest schema_version=N is newer than this build supports"

You are running an older `doc-rag` against a `build/manifest.json` that
was written by a newer version. Two options:

1. **Upgrade.** Pull the latest `doc-rag`, run `doc-rag migrate`, then
   retry the failing command.
2. **Restore.** If you cannot upgrade, restore a backup produced by an
   older `doc-rag` (see `scripts/restore.sh`).

Manifests without a `schema_version` key are treated as legacy and read
without complaint; only explicitly higher versions are refused.

## Search returns no results

1. **Empty index?** `doc-rag ingest` was never run, or the corpus was wiped. Drop files
   into `sources/incoming/` and ingest.
2. **Index missing → degraded mode?** Check `GET /ui/status` or look for the
   "Семантический поиск недоступен" / "Semantic search unavailable" banner. The
   fix is `doc-rag rebuild` (or the "Запустить rebuild" / "Start rebuild" button).
   Lexical fallback should still return *something*; if it doesn't, the manifest
   itself is empty.

## `doc_search` hangs / 60+ second timeouts

The search request must never trigger a full FAISS encoding. If you see hangs:

1. Confirm `build/index/faiss.index` exists and matches the chunk count in
   `build/manifest.json`. If not, run `doc-rag rebuild` *outside* a request (CLI or via
   the UI button — it goes through a background task).
2. Check CPU: a runaway rebuild loop will pin all cores. `journalctl -u doc-rag-mcp -f`
   should be telling you what it's doing.
3. The fix that prevents the original incident: `semantic_search` returns `None` if the
   index is missing, instead of building it inline (see `src/doc_rag/server/retrieval.py`).

## FAISS rebuild is taking hours

Expected on CPU. With `bge-large-en-v1.5` and ~4500 chunks, a CPU-only rebuild on
8 cores takes roughly 3 hours. Use `--gpu` at install time, or a smaller model
(`bge-small-en-v1.5`) if you can't.

While rebuild runs the server stays up and `doc_search` falls back to lexical search
with a clear degraded-mode warning.

## MCP not visible to Cursor / Agent says "no doc_search tool"

1. Restart Cursor fully — workspace MCP configs are loaded once.
2. Settings → MCP: doc-rag listed and toggled on; `doc_search` enabled in the tools list.
3. In Chat → "Available Tools": confirm `doc_search` is enabled for the conversation.
4. If still missing, switch to global config: `bash scripts/print_global_mcp_config.sh`
   (or `python3 scripts/write_global_mcp_config.py` on WSL where CRLF mangles the
   shell output), paste into `~/.cursor/mcp.json`, restart Cursor, open a **new** chat.
5. Verify the server itself works: `bash scripts/verify_mcp.sh` should print `doc_search`
   in `tools/list`.

## Torch / CUDA issues

Reinstall torch via the helper scripts (they pin to a compatible wheel and use the
project's `.venv/bin/python` directly to avoid system-Python confusion):

```bash
pip uninstall torch -y
bash scripts/install_torch_gpu.sh    # or install_torch_cpu.sh
```

## PEP 668 / `externally-managed-environment`

Means pip tried to install into system Python. This project installs into `.venv`.

- Run `bash scripts/bootstrap.sh` (handles venv creation), **or**
- Run pip via `.venv/bin/python -m pip ...`

The torch install scripts already use `.venv/bin/python` directly.

## `.doc` parsing fails

`.doc` (legacy binary Word) needs `antiword` or `catdoc` on PATH:

```bash
sudo apt install antiword
```

Both `docker/Dockerfile` and `scripts/install_server_native.sh` install `antiword`
automatically. If the file is very small, antiword may complain about "text stream
too small to handle" — that's a parser quirk, not a system issue.

## OCR not working

Required system packages:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus
# optional, for formulas (may be missing on some distros):
sudo apt install tesseract-ocr-equ
```

Python side: `pip install pytesseract Pillow pymupdf` (the bootstrap installs these when
`DOC_RAG_BOOTSTRAP_OCR=Y`).

Verify:

```bash
.venv/bin/python -c "import pytesseract, PIL; print('ok')"
tesseract --version
```

## Duplicate uploads aren't detected

Dedup uses sha256 against both `build/manifest.json` and the current `sources/incoming/`
queue. If a file is dropped onto disk *outside* the UI (e.g. `cp file.pdf
sources/incoming/`) and ingested later, manifest-side dedup still works on the *next*
upload of the same file — but the on-disk drop won't be flagged at the moment of copy
(no upload event to react to).

## Stale archived files after delete

`delete_documents` removes the source file by both the manifest-recorded path and the
basename in `sources/archived/` / `sources/incoming/`, to cope with manifest path drift
across ingest archiving. If a file persists after delete, check:

1. Was it actually archived under a different filename (suffix added on hash collision)?
2. Is the manifest entry consistent with the path on disk?

A safe nuclear option is `doc-rag clean-orphans` followed by `doc-rag clear-incoming`.
