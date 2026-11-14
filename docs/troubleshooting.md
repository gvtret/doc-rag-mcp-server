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
2. **Index or model missing?** Check `GET /ui/status`: `indexed.model_ready`
   false means the embedding model did not load, `indexed.semantic_search_ready`
   false means no usable semantic index. A missing index has no banner of its
   own - the Documents summary shows "сем.: выкл" / "sem.: off" (and the footer
   shows "деградация" / "degraded" while the probe is not ready). The fix is
   `doc-rag rebuild` (or "Пересобрать индекс" / "Rebuild index" on the Manage
   page). Lexical fallback should still return *something*; if it does not, the
   manifest itself is empty.

## Model / index diagnostics (Web UI banners)

The Documents page shows three status banners driven by `GET /ui/status`
(`indexed.model_ready`, `indexed.index_stale`,
`indexed.model_update_available`):

### "Модель эмбеддингов не найдена..." (ERROR)

`model_ready: false` - the embedding model is missing or failed to load,
so semantic search is off. `doc_search` in semantic/hybrid mode does not
hang or error out: it prepends a degraded-mode notice (the model is
unavailable, rebuild will not help) and serves lexical fallback results.

1. Check the model directory: `models/bge-m3/config.json` and
   `modules.json` must exist (or `embeddings.model_path` must point at a
   valid snapshot).
2. Offline host: fetch the model once on a connected machine -
   `uv run python scripts/fetch_model.py` (see
   [install.md](install.md#offline-model-delivery-fetch_modelpy)), then
   ship it by one of the channels in
   [deploy.md](deploy.md#model-delivery-channels).
3. Tail `journalctl -u doc-rag-mcp` for the loader's error message. Failed loads
   are not cached, so a search request retries on its own; `/health/ready` and
   the banner only change after a successful attempt - restart the service to
   re-run the startup warm-up and refresh the readiness state.

Lexical search keeps working while the model is down.

### "Индекс устарел..." (WARN)

`index_stale: true` - the weights fingerprint of the loaded model no
longer matches `build/index/index_meta.json` (`weights_sha256`). The
FAISS vectors were built with different weights, so semantic results
would be silently degraded.

Fix: run `doc-rag rebuild` (or "Пересобрать индекс" / "Rebuild index" on the Manage page) to
re-embed the corpus with the current weights. `stale_reason` in
`/ui/status` names the mismatch. After re-embedding, the banner clears
by itself.

### "На HF Hub доступна новая ревизия весов..." (INFO)

`model_update_available: true` - the background, read-only Hub check
found a newer weights revision. Nothing is downloaded or swapped
automatically; the local index keeps working. If you decide to update:
fetch with `scripts/fetch_model.py` (the replace is atomic; add
`--force` only if a leftover `.staging` directory blocks the run),
restart the service, then rebuild (the stale-index banner above will
guide you). Disable the check with
`embeddings.update_check_interval_hours: 0` (default interval: 72 h).

## `doc_search` hangs / 60+ second timeouts

Historical 504s (client timeout on a hanging `doc_search`) were caused
by the search request triggering a full FAISS encode when the model was
missing. Fixed: the model is warmed up at startup, `semantic_search`
returns `None` instead of building anything inline, `doc_search`
prepends a degraded-mode notice and serves lexical results, and the
readiness probe reports `model_missing` / `index_stale` reasons instead
of serving a doomed request. If you still see a hang:

1. Confirm `build/index/faiss.index` exists and matches the chunk count in
   `build/manifest.json`. If not, run `doc-rag rebuild` *outside* a request (CLI or via
   the UI button — it goes through a background task).
2. Check CPU: a runaway rebuild loop will pin all cores. `journalctl -u doc-rag-mcp -f`
   should be telling you what it's doing.
3. The fix that prevents the original incident: `semantic_search` returns `None` if the
   index is missing, instead of building it inline (see `src/doc_rag/server/retrieval.py`).

## FAISS rebuild is taking hours

Expected on CPU. With `bge-large-en-v1.5` and ~4500 chunks, a CPU-only rebuild on
8 cores takes roughly 3 hours (the shipped default is `bge-m3`; these numbers are
for `bge-large-en-v1.5`). Use `--gpu` at install time, or a smaller model
(`bge-small-en-v1.5`) if you can't.

While rebuild runs the server stays up. Search keeps using the previous index
until the new one is written; once the index is being rewritten the request may
fall back to lexical search with a degraded-mode notice.

## MCP not visible to Cursor / Agent says "no doc_search tool"

1. Restart Cursor fully — workspace MCP configs are loaded once.
2. Settings → MCP: doc-rag listed and toggled on; `doc_search` enabled in the tools list.
3. In Chat → "Available Tools": confirm `doc_search` is enabled for the conversation.
4. If still missing, switch to global config: `bash scripts/print_global_mcp_config.sh`
   (or `python3 scripts/write_global_mcp_config.py` on WSL where CRLF mangles the
   shell output), paste into `~/.cursor/mcp.json`, restart Cursor, open a **new** chat.
5. Verify the server itself works: `bash scripts/verify_mcp.sh` should print `doc_search`
   in `tools/list`.

## uv install fails / `uv: command not found`

Since v2.1 the only supported installer is [uv](https://docs.astral.sh/uv/).
`bash scripts/bootstrap.sh` installs it via the Astral one-liner if it
is missing on PATH. To install manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If after install the shell still cannot find `uv`, source the rc
snippet printed by the installer, or restart the shell.

## `uv sync --frozen` complains about lockfile drift

`pyproject.toml` was edited locally without updating `uv.lock`. Either:

- run `uv lock` to regenerate the lockfile (commit it alongside the
  `pyproject.toml` change), or
- set `DOC_RAG_BOOTSTRAP_FROZEN=0` to let bootstrap run `uv sync`
  without `--frozen` (one-off; not for production).

## `.doc` parsing fails

`.doc` (legacy binary Word) needs `antiword` or `catdoc` on PATH:

```bash
sudo apt install antiword
```

Both `docker/Dockerfile` and `scripts/install_server_native.sh` install `antiword`
automatically. If the file is very small, antiword may complain about "text stream
too small to handle" — that's a parser quirk, not a system issue.

## OCR not working on scanned PDFs

Since v2.0 OCR is handled by Docling internally via RapidOCR — there is
no Tesseract install to maintain. If a scanned PDF comes back with no
text:

1. **First-run download.** Docling fetches RapidOCR weights on first
   parse. Check `~/.cache/docling/` exists and is writable by the
   service user; tail `journalctl -u doc-rag-mcp` for download errors.
2. **Empty cache after offline restore.** Re-run a PDF parse from a
   connected host to repopulate `~/.cache/docling/`, then rsync it
   back.
3. **Truly empty pages.** RapidOCR returns nothing if a page is blank
   or all-graphics-no-text. Verify by exporting a page to PNG and
   running RapidOCR by hand.
4. **arch_config.yaml missing (rapidocr 3.8.2).** rapidocr 3.8.2 is a
   yanked release that ships without `arch_config.yaml`. If OCR fails
   with `No such file or directory: '.../arch_config.yaml'`, the
   bootstrap script should have patched it automatically. If running
   without bootstrap (e.g. manual Docker build), either downgrade:
   `uv pip install rapidocr==3.8.1`, or copy
   `scripts/rapidocr/arch_config.yaml` into the venv's
   `rapidocr/inference_engine/pytorch/networks/` directory.

## "pdf_backend is no longer supported"

You hit `RuntimeError: parsing.pdf_backend='pymupdf' is no longer
supported.` (or `'pypdf2'`, or `'auto'`-without-Docling). This is the
v2.0 hard gate: PyMuPDF/PyPDF2 were removed, Docling-based parsing is
the only supported path.

Fix: in `config/config.yaml` set `parsing.pdf_backend` to one of the two
valid values:

- `docling` (default) — Docling only.
- `cascade` (v2.3+) — Docling first, falls back to Unstructured (if
  installed) when Docling's quality score is below
  `parsing.quality_fallback_threshold`. See [install.md](install.md#pdf-quality-cascade).

Or remove the key entirely — `docling` is the default.

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
