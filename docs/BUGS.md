# Bugs and gotchas

Living log of defects and pitfalls found while working on doc-rag. Each entry
has source context, symptom, analysis and status (FIXED / WORKAROUND / NOT FIXED).

---

## BUG-006 — incremental index update ignores a changed embedding model

**Status:** FIXED

**Where:** `src/doc_rag/raglib/vector_index.py` — `VectorIndex.build()`,
incremental branch (`if not force and store.size > 0:` … line ~148).

**Symptom:** after changing `embeddings.model_name` (e.g.
`BAAI/bge-large-en-v1.5` → `BAAI/bge-m3`), an ingest logs

```
[doc-rag][index] Incremental update: +N chunks
```

instead of rebuilding from scratch. The resulting `faiss.index` is then a
**mix of vectors from two different models** while `index_meta.json` records
only the new `model_name` — semantically wrong and undetectable at query time.
Observed: `index_meta` model `BAAI/bge-large-en-v1.5` with 8466 ids while
`chunks.jsonl` had 18413 chunks on the new config.

**Analysis / comment:** the full-build branch reads the config model and calls
`store.init(dim=…, model_name=…)`, but the incremental branch never compares the
configured `model_name` / `dim` / `normalize` against the on-disk
`IndexMeta`. It only checks chunk-id set membership (`new_ids`), so every
existing id is assumed still valid.

Note there are **two** index paths and they diverge:

- `indexer.ensure_faiss_index()` (used by `rebuild`/`rebuild-index`) **does**
  guard this:
  ```python
  if meta.model_name != model_name or meta.metric != metric or meta.normalize != normalize:
      log("[doc-rag][index] Index settings changed -> forcing rebuild")
      meta = None
  ```
- `VectorIndex.build()` (used by `ingest()` via `pipeline.py:773`) does **not**.

So `ingest` can silently corrupt an index that `rebuild` would have rebuilt
correctly. Same class of staleness as Gotcha-004 (index out of sync with
`chunks.jsonl`).

**Fix applied:** `VectorIndex.build()` now calls `_store_matches_config()` before
taking the incremental branch. For the FAISS backend it compares the on-disk
`IndexMeta.model_name`/`normalize` against the current config; on mismatch it
falls through to a full rebuild. It also compares `IndexMeta.weights_sha256`
with the aggregate weight fingerprint of the loaded model and forces a full
rebuild when they differ or when the on-disk index predates fingerprinting.
Tests:
`tests/test_vector_index.py::TestModelChangeForcesRebuild` (model swap rebuilds;
same model still updates incrementally).

**Side effect observed during the bug:** RSS climbed to ~11.8 GB and one
`asyncio_0` thread pinned ~100% CPU while `_encode` looped over many small
incremental batches; GPU sat at 100% util but only ~9 MiB free. A single full
batch pass is both correct and far cheaper.

---

## BUG-007 — `ingest` rebuilt the whole index per document (O(n²), GBs of RSS)

**Status:** FIXED

**Where:** `src/doc_rag/raglib/pipeline.py` — incremental ingest branch,
the "Per-file index update" block (now removed).

**Symptom:** an `ingest` over a real 60-document corpus stalled around
45/60 with one `asyncio_0` thread pinned at ~100% CPU and GPU at 100% for
20+ minutes; RSS grew to 6–11 GB; the index contained only 219 of 18899
chunks after an hour.

**Analysis / comment:** on every parsed document the code called
`vi.build(file_chunk_dicts, force=False)`, i.e. it rebuilt/rewrote the whole
FAISS index once per file. That is O(n²) embedding-adjacent work, repeatedly
serialises the index, and leaks memory across the loop. It also meant the
index tracked documents one-by-one instead of once at the end.

**Fix applied:** removed the per-file `vi.build()`; `ingest` now parses and
chunks **all** documents first (appending to `chunks.jsonl`), then builds the
index **once** from the complete chunk set via `_build_index_with_progress()`
(shared with `rebuild`, `rebuild-index`, `rebuild-docs`, so all entry points
report the indexing phase). Regression test:
`tests/test_incremental_persist.py::test_ingest_builds_index_once_after_all_documents`.
Observed after the fix: RSS stayed ~1.5 GB and the run proceeded past the
previous stall point.

---

## Gotcha-005 — GPU embedding throughput is dominated by batch size

On the GTX 1650 (4 GB) with bge-m3, measured on real ~260-token chunks:

| batch_size | chunks/s | peak VRAM |
| --- | --- | --- |
| 8 | 1.9 | 2287 MiB |
| 16 | 6.7 | 2399 MiB |
| **32** | **7.9** | 2625 MiB |
| 48 | 5.6 | 2852 MiB (VRAM exhausted) |
| 64 | 0.7 | 3076 MiB (thrashing) |

So the previous `batch_size: 8` was ~4x slower than necessary; `32` is the
sweet spot, `48+` degrades. Also cap `embeddings.max_seq_length` to the chunk
size (512) — bge-m3 otherwise defaults to 8192 and inflates cost. Both are
now set in `config/config.yaml` and honoured by `_load_embedder()`.

---

## BUG-005 — broken-xref PDF fails with `Stream has ended unexpectedly`

**Status:** FIXED (code fallback) + WORKAROUND (source file repaired)

**Where:** `src/doc_rag/raglib/docling_backend.py` — `_parse_pdf_chunked()`,
`_parse_pdf()`; triggering file `IEC-61850/IEC TR 61850-90-4-2013.pdf`.

**Symptom**

```
[doc-rag][ERROR] failed: .../IEC TR 61850-90-4-2013.pdf: Both Docling and
Unstructured failed ... Docling error: Stream has ended unexpectedly.
```

**Analysis / comment:** the PDF ships with a **broken xref**: `startxref`
points to an xref at offset 6411176, but that xref has no leading `xref`
keyword (`b'xref\\n0\\n3277...'` parses only when mutool rebuilds it), and
`pypdf` aborts with `PdfStreamError: Stream has ended unexpectedly`
("EOF marker not found"). Because the file is >3 MB, `_parse_pdf` routes it
through `_parse_pdf_chunked`, which uses `pypdf` to slice pages — so the whole
document was dropped even though poppler/`pdftotext` and Docling can read it
fine. Reproduce: `python -c "from pypdf import PdfReader;
PdfReader('…90-4-2013.pdf')"`.

**Fix (code):** `_parse_pdf_chunked` now catches the `PdfReader` failure and
retries on a `mutool clean -gggg`-repaired temporary copy
(`_repair_pdf_with_mutool`); if `mutool` is absent it re-raises as before.
Tests: `tests/test_docling_backend.py::test_repair_pdf_with_mutool_*`.

**Workaround applied:** the source file under `/mnt/e/Docs/IEC-61850/` was
repaired in place with `mutool clean -gggg` (original backed up at
`/tmp/opencode/repair/90-4_orig_backup.pdf`). Verified: `pypdf` now reports
268 pages.

---

## BUG-003 — `/ui/documents` shows no live progress during `ingest`

**Status:** FIXED

**Where:** `src/doc_rag/server/mcp_http.py` — `_build_documents_list()`.

**Symptom:** during an `ingest` run the Documents page never updates: no
progress bar, no per-row `processing`/`processed` status, count stuck at the
old value (until the run finishes).

**Analysis / comment:** the live-progress block was gated on
`"rebuild" in job`:

```python
if _INGEST_STATE.get("running") and "rebuild" in (_INGEST_STATE.get("job") or ""):
```

`ingest()` has `job == "ingest"`, so `rebuild_progress` stayed `None` and no
`d["status"]` was ever set, even though the same log-marker progress tracking
(`found N file(s)`, `ok:/skip:/failed:`) is populated for ingest too. The UI
polls `/ui/documents` every 5 s but had nothing to render.

**Fix:** also enter the block when `docs_total` is known
(`... or _INGEST_STATE.get("docs_total") is not None`). Regression test:
`tests/test_ingest_progress.py::test_documents_list_exposes_live_progress_during_ingest`.

**Related:** the same function read `_root_dir()` directly, ignoring the
`DOC_RAG_ROOT` override that every other helper honors — fixed inline so tests
(and overrides) point at the right corpus.

---

## BUG-004 — flooded with `RapidOCR main.py:132: The text detection result is empty`

**Status:** FIXED

**Where:** `src/doc_rag/raglib/docling_backend.py` — `_get_converter()`;
message raised in `rapidocr/main.py:298`.

**Symptom:** the ingest log fills with
`[WARNING] ... [RapidOCR] main.py:132: The text detection result is empty`.

**Analysis / comment:** RapidOCR raises `RapidOCRError("The text detection
result is empty")` for any image/region whose detector finds no text (blank
pages, figures, graphic-only areas). Docling catches it and logs a WARNING per
occurrence — high volume on scanned/hybrid PDFs, no actionable signal.

**Fix:** attach a `logging.Filter` (`_DropEmptyTextDetectionWarning`) to the
`RapidOCR` logger that drops exactly that message and keeps everything else.
Test: `tests/test_docling_backend.py::test_rapidocr_empty_text_warning_filtered`.

---

## BUG-001 — `pypdf` / `cryptography` used but not declared as dependencies

**Status:** WORKAROUND (installed manually)

**Where**

- `src/doc_rag/raglib/docling_backend.py:471` — `from pypdf import PdfReader`
  (page-count probe for large files).
- `src/doc_rag/raglib/docling_backend.py:522` — `from pypdf import PdfReader, PdfWriter`
  (chunked parsing of files >3 MB / >200 pages).
- `pyproject.toml` — neither `pypdf` nor `cryptography` is listed in
  `dependencies` or in any `optional-dependencies` extra.

**Symptom**

```
[doc-rag][ERROR] failed: ...: Both Docling and Unstructured failed ...
Docling error: No module named 'pypdf'
```

and, after installing `pypdf`, for AES-encrypted PDFs:

```
Docling error: cryptography>=3.1 is required for AES algorithm
```

**Analysis / comment**

Both packages normally arrive *transitively* via the `unstructured` extra
(`unstructured-client` → `pypdf`; `pypdf` needs `cryptography` for AES-encrypted
PDFs). A deployment that installs `server`, `faiss`, `embeddings`, `metrics`
but **not** `unstructured` therefore has a broken PDF path:

- every PDF >3 MB or >200 pages fails (`pypdf` missing);
- every encrypted PDF fails (`cryptography` missing).

This is a packaging bug: code in the main package depends on a package that is
only guaranteed by an unrelated optional extra. Fix should be to add
`pypdf>=6` (and `cryptography`) to `[project.dependencies]`, or to import them
lazily with a clear actionable error instead of silently failing the document.

**Repro**

```bash
uv sync --frozen --extra server --extra faiss --extra embeddings --extra metrics
.venv/bin/python -c "import pypdf"   # ModuleNotFoundError
```

**Workaround applied**

```bash
uv pip install --python .venv/bin/python pypdf cryptography
```

---

## BUG-002 — `scripts/install_torch_gpu.sh` fails on a uv-created venv

**Status:** WORKAROUND

**Where:** `scripts/install_torch_gpu.sh:15`

```bash
"${VENV_PY}" -m pip install -U pip
"${VENV_PY}" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
```

**Symptom**

```
/home/trgv/doc-rag/.venv/bin/python: No module named pip
```

**Analysis / comment**

`bootstrap.sh` creates `.venv` via `uv sync`, and uv-created venvs do **not**
ship `pip`. The script assumes pip is present, so the documented GPU install
path (`docs/install.md` → “GPU install (NVIDIA / CUDA)”) fails on the very
venv the project tells you to use.

**Workaround applied**

```bash
uv pip install --python .venv/bin/python --reinstall \
  --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

Note: `torchvision` must be reinstalled from the *same* cu124 index, otherwise
`torch+cu124` + `torchvision+cpu` is ABI-incompatible and
`sentence_transformers` import dies on `operator torchvision::nms does not exist`.

---

## Gotcha-001 — `uv run` reverts the CUDA torch build

`pyproject.toml` pins torch/torchvision to the CPU index
(`[tool.uv.sources]` → `pytorch-cpu`). After installing CUDA wheels into
`.venv`, a plain `uv run doc-rag ...` re-syncs the project env and silently
reinstalls `torch+cpu`. Run the CLI directly (`.venv/bin/doc-rag`) or with
`uv run --no-sync` on a GPU box.

---

## Gotcha-002 — `uv sync --frozen` without extras strips the RAG stack

`uv sync --frozen` installs only the default dependency group. The vector/RAG
stack lives in extras (`embeddings`, `faiss`, `server`, `metrics`,
`unstructured`), so a bare sync removes `sentence-transformers`, `faiss`,
etc. Always pass the extras the deployment needs (see `scripts/bootstrap.sh`).

---

## Gotcha-004 — FAISS index can be silently out of sync with `chunks.jsonl`

**Status:** NOT FIXED (observed on the shipped `build/`)

Note (2026-11): `ensure_faiss_index` was removed in 65e7e35; its logic
now lives in `VectorIndex.build` (with the integrity checks the gotcha
asks for -- a meta/index mismatch degrades to unavailable instead of
serving wrong chunks).

`build/index/faiss.index` held **3243** vectors while `build/chunks_jsonl/chunks.jsonl`
held **13190** chunks (`index_meta.json` also listed 3243 `chunk_ids`). Semantic
search therefore only covered ~25% of the corpus, with no warning. `ensure_faiss_index`
only rebuilds when the model/metric/normalize changed or when the index is *shorter*
than chunks *and* `meta.chunk_ids` is present — a stale-but-long-enough meta is not
detected. Worth an integrity check (compare `index.ntotal` / `meta.chunk_ids` against
the chunk count and force a rebuild on mismatch).

---

## Gotcha-003 — `quality.fail_on_severity` is not enforced

`quality.fail_on_severity` is validated as a config field
(`src/doc_rag/server/mcp_http.py`) but is never read on the ingest path — it
does not abort or gate anything today. Documents with warnings are ingested
regardless.
