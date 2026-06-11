# Changelog

All notable changes to `doc-rag` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project does not yet ship versioned tags, so entries are grouped by date.

## v2.1.3 — 2026-06-11

Two regression fixes uncovered during the post-uv-migration server
re-bootstrap. Behavioural surface unchanged (parsing, manifest, MCP).

### Fixed
- **Semantic search silently degraded to lexical after a fresh uv
  sync.** `[tool.uv.sources]`'s CPU index override redirected
  `torch` to the PyTorch CPU index, but `torchvision` — being a
  transitive dep of `docling-ibm-models` / `rapidocr` — kept resolving
  from default PyPI. The resulting ABI mismatch
  (`torch+cpu` vs. `torchvision` default) made
  `from sentence_transformers import SentenceTransformer` raise
  `RuntimeError: operator torchvision::nms does not exist`, which
  `semantic_search`'s catch-all swallowed; `doc_search` fell back to
  lexical without any visible signal beyond integer-shaped scores
  in the response. Fix: list `torchvision` directly in the
  `[embeddings]` extra so the source override applies.
- **`sentence-transformers` + `transformers` resolution.** uv lock
  picked `sentence-transformers 5.5.1` + `transformers 5.8.1`; ST
  5.5.x imports `PreTrainedModel` from a transformers module
  surface that 5.8 no longer exposes. Pinned to
  `sentence-transformers<5.5` and `transformers<5.8`.
- **`scripts/install_server_native.sh` would wipe `sources/`.**
  The rsync step protected `build/` but not `sources/`; running the
  script as an upgrade on an existing install (with archived
  documents) would have deleted them via `--delete`. Fix: add
  `--exclude 'sources'` + `--filter 'protect sources/'` (mirrors the
  existing `build/` defence).

### Internal version label
- `pyproject.version` + the three hardcoded strings in
  `src/doc_rag/server/mcp_http.py` bump to `2.1.3`.

## v2.1.2 — 2026-06-11

CI tooling patch on top of v2.1.1. No code, deps, or behavioural
surface changes.

### Changed
- **buildx + uv caches moved to the runner host (`type=local`).**
  Both write to `/opt/github-runner/_cache/` instead of the
  Azure-backed GHA `actions/cache` service:
  - `build.yml`: `cache-from/to: type=local,src=/opt/github-runner/_cache/buildx,mode=max`
  - `tests.yml` + `lint.yml` via `setup-uv@v3`: `cache-local-path: /opt/github-runner/_cache/uv`

  Skips the 10 GB per-repo GHA cache quota and survives Azure
  outages (one of which silently dropped the uv cache during v2.1.1
  CI). Expected outcome: warm-cache docker build drops from ~14 min
  to seconds when `pyproject.toml` / `uv.lock` / `docker/Dockerfile`
  have not changed since the previous run.

### Removed
- `jlumbroso/free-disk-space@main` step in `build.yml`. Was added
  in v2.1.0 to free GHA-hosted preinstalled toolchains; a no-op on
  a clean self-hosted runner.

### Internal version label
- `pyproject.version` + the three hardcoded strings in
  `src/doc_rag/server/mcp_http.py` bump to `2.1.2`.

## v2.1.1 — 2026-06-11

CI + lockfile patch on top of v2.1.0. Behavioural surface (parsing,
manifest schema, MCP) unchanged.

### Changed
- **CI runs on a self-hosted runner.** All three workflows
  (`tests`, `lint`, `build`) swap `runs-on: ubuntu-latest` →
  `runs-on: self-hosted`. The `jlumbroso/free-disk-space@main` step
  in `build.yml` becomes a no-op on a clean self-hosted runner;
  leaving it in is harmless and defensive.
- **Torch is locked to the CPU index.** `pyproject.toml` adds a
  `[tool.uv]` source override pointing `torch` and `torchvision`
  at `https://download.pytorch.org/whl/cpu`. `uv.lock` regenerated:
  0 `nvidia-*` packages (down from 10), `torch` resolves to
  `2.12.0+cpu`, fresh venv drops from ~5 GB to **1.6 GB**. Aligns
  with the project's CPU-only deployment policy.
- **`faiss-cpu` capped at `<1.14`.** 1.14.x wheels emit AVX/AVX2
  ops unconditionally and SIGILL on hosts that only expose SSE4.2
  (the QEMU-virtualised production server). 1.13.x stays
  compatible.

### Internal version label
- `pyproject.version` + the three hardcoded strings in
  `src/doc_rag/server/mcp_http.py` bump to `2.1.1`.

## v2.1.0 — 2026-06-11

uv migration. `uv` is now the only officially supported installer for
`doc-rag`; pip is no longer documented or scripted. Behavioural surface
(parsing, manifest schema, MCP) unchanged.

### Added
- `uv.lock` (committed). Every transitive dependency pinned across
  Python 3.10-3.14 and Linux/macOS/Windows resolution markers. CI's
  `uv sync --frozen` and a local dev `bash scripts/bootstrap.sh` now
  hit the exact same dependency graph.
- `ruff` added to the `[dev]` extra so it lands in the lockfile (it
  was previously installed ad-hoc outside the venv).

### Changed
- `scripts/bootstrap.sh` rewritten around `uv sync`. Installs uv via
  Astral's official installer (`curl … | sh`) if missing; prompts for
  the same extras as before (FAISS / embeddings / dev / metrics).
  Always installs the `server` extra. `DOC_RAG_BOOTSTRAP_FROZEN=0`
  relaxes `--frozen` when iterating on `pyproject.toml`.
- `scripts/install_server_native.sh`: `--gpu` / `--cpu` flags removed
  (GPU was never on the deployment matrix). `--minimal` retained
  (skip embeddings extra → MCP only, no semantic search).
- `docker/Dockerfile`: `pip install -e .` → `COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv` + `uv sync --frozen`. Layer cache splits dependency install from project install for fast rebuilds.
- `.github/workflows/tests.yml`, `lint.yml`: use `astral-sh/setup-uv@v3`
  with cache enabled; `uv sync --frozen --extra …` replaces
  `pip install -e ".[…]"`; lint uses `uvx ruff` for zero-venv lint.
  Stale `[pdf,ocr]` extras in tests.yml (left over from v1.x) removed
  in the same pass.
- `docs/install.md` rewritten around uv. `docs/troubleshooting.md`
  drops the torch-helper and PEP-668 entries (no longer relevant);
  adds two uv-specific FAQ entries. README quickstart updated.

### Removed
- `requirements.txt`. The single source of dependency truth is now
  `pyproject.toml` + `uv.lock`. `scripts/install_torch_cpu.sh` and
  `install_torch_gpu.sh` stay in the tree for advanced manual use but
  are no longer referenced from docs or bootstrap.

### Internal version label
- `pyproject.version` + the three hardcoded strings in
  `src/doc_rag/server/mcp_http.py` bump to `2.1.0`.

## v2.0.1 — 2026-06-11

UI quality-of-life patch on top of v2.0.0. No behavioural changes to
parsing, the manifest schema, or the MCP surface.

### Added
- **Per-document progress in the Web UI during ingest/rebuild.** While
  a job runs, the Status panel shows `Сейчас: <basename> · X/Y ·
  осталось ~<eta>`. ETA is extrapolated from elapsed time after the
  first document finishes; before that the line stays at the current
  basename and counter only. Derived from existing pipeline log
  markers (`parse:`/`ok:`/`skip:`/`failed:`/`found N file(s)`), no
  new instrumentation in `pipeline.py`.
- **Per-document OCR-applied indicator.** Each row in the indexed
  documents table now carries a small `OCR` badge when RapidOCR fired
  on at least one page; tooltip shows pages count and mean
  confidence. The signal is read from Docling's
  `ConversionResult.confidence.pages[i].ocr_score` (non-NaN positive
  = OCR ran) and propagated through `stats.ocr.applied` /
  `pages_recognized` / `confidence` into the manifest. New
  documents ingested under v2.0.1+ are tagged correctly; old entries
  ingested under v2.0.0 (or earlier) keep showing `applied: False`
  until they are re-parsed.

### Fixed
- `stats.ocr` shape was a stub in v2.0.0 — `applied` always reported
  `false`. v2.0.1 fills it with the real Docling signal.

### Server label
- `serverInfo.version` in the MCP `initialize` response and the
  FastAPI app title bump to `2.0.1`.

## v2.0.0 — 2026-06-05

**Breaking release.** Two coupled changes ship together:
PyMuPDF and PyPDF2 are removed from the project, leaving Docling as
the single PDF backend; with PyMuPDF gone, the AGPL load-bearing dep
is also gone and the project relicenses from AGPL-3.0-or-later to MIT.

### Why

A comparative parse of a real СТО (СТО 34.01-5.1-006, 12 pages, with
Приложение И — a multi-page table) on the production server:

| | PyMuPDF (auto) | Docling |
| --- | --- | --- |
| Wall time | 4.16 s | 216.69 s |
| Blocks total | 1 paragraph | 82 (paragraph 32 / heading 12 / table 17 / list_item 21) |
| Headings | 0 | 12 |
| Tables (structured grid) | 0 | 17 |

PyMuPDF collapsed the document into one paragraph; Docling reproduced
the heading hierarchy and reconstructed Приложение И as a 7-row
labelled grid. The wall-time penalty (~50×) is real but acceptable
for an offline-ingest workflow where ingest already runs out-of-band.

### Removed
- `pymupdf` (AGPL-3.0). Removed from base deps; the `[pdf]` extra is
  gone. All `fitz`-using code paths in `parsers.py` are deleted —
  `_pdf_fitz_extract_with_ocr`, `_extract_page_text_structured`,
  `_ocr_page_text`, `_page_has_embedded_images`,
  `_detect_pdf_is_scan`.
- `PyPDF2` (BSD-3, but only ever a fallback for the PyMuPDF path).
  Removed from base deps; `_parse_pdf_pypdf2` deleted.
- `pytesseract` + `Pillow`. The Tesseract-based OCR loop is gone; OCR
  for scanned PDFs is now handled inside Docling by RapidOCR. The
  `[ocr]` extra is removed. The Tesseract apt packages
  (`tesseract-ocr*`) are no longer installed by
  `scripts/install_server_native.sh` or the Docker image.
- `[docling]` extra. Docling is now a base dep — no opt-in flag.
- `parsing.ocr.*` config block. Keys are silently ignored; the
  `stats.ocr` shape in `parse_document` output is preserved for
  manifest schema v1 compatibility but always reports
  `applied: false`.
- `parsing.edition_year.from_pdf_metadata`. The auto-read of PDF
  `/CreationDate` / `/ModDate` is gone (it required PyMuPDF or
  PyPDF2). The remaining `by_basename` / `by_source_rel_path` /
  `by_sha256` / `filename_regex` cascade is unchanged.

### Changed
- **License: AGPL-3.0-or-later → MIT.** All AGPL load-bearing deps
  removed in the same release; relicensing was the whole point.
  `LICENSE` rewritten, `NOTICE` rewritten, `pyproject.toml` carries
  `license = "MIT"`, README badge updated.
- `parsing.pdf_backend` config key kept for migration compatibility,
  but only `"docling"` and `"auto"` (alias) are accepted. Anything
  else raises `RuntimeError` at `parse_document` time with an
  actionable message. `parsing.docx_backend` still accepts both
  `"python-docx"` (default, fast) and `"docling"` (structure-aware,
  slow).
- `docs/install.md`, `docs/cli.md`, `docs/troubleshooting.md`,
  `docs/roadmap.md`, `README.md` updated for the new world.
- `scripts/bootstrap.sh` no longer prompts for PyMuPDF or OCR install
  paths; Docling is pulled in by the base `pip install -e .`.

### Migration

For an existing v1.4.x install:

1. Wipe the venv and reinstall — Docling brings ~5 GB of deps.
2. Edit `config/config.yaml`: set `parsing.pdf_backend: docling` (or
   delete the key — `docling` is the default). Remove the
   `parsing.ocr` block; it has no effect.
3. If you relied on automatic edition-year extraction from PDF
   metadata, add explicit entries under
   `parsing.edition_year.by_basename` (or `by_sha256`).
4. First PDF parse will download ~300 MB of ML weights into
   `~/.cache/docling/` — one-time cost; subsequent parses are
   offline. Plan the first ingest accordingly.
5. Per-doc wall time is ~10-20 s/page on CPU. A ~4500-chunk corpus
   that used to rebuild in ~36 min may take 25-30 hours; move
   full-corpus ingest from a nightly cron to a weekly one.

Manifest schema and FAISS index layout are unchanged (still
`schema_version: 1`); no `doc-rag migrate` step required.

## v1.4.0 — 2026-06-04 (first public release)

This release is the cumulative result of internal Sprints 1 through 4
(see `docs/roadmap.md`). Earlier intermediate version tags
(`v1.1.0`–`v1.3.0`) existed only locally and were never published.

### Added (Sprint 4: benchmarks, polish, release readiness)
- `scripts/bench.py` — self-contained, reproducible benchmark for the
  embedding-encode + FAISS build + FAISS reconstruct paths. Synthetic
  corpus generated in-process; JSON output with a `schema_version: 1`
  envelope. Supports CPU and CUDA (`--device cpu|cuda|auto`).
- `docs/bench-results.md` — reference numbers for `bge-large-en-v1.5`
  on three host classes (GTX 1650 GPU, i7-class CPU, QEMU server CPU).
  Documents the index-size memory and latency profile of
  `IndexFlatIP`, including when to switch to an approximate index
  (v2 territory).
- `docs/install.md` § "GPU install (NVIDIA / CUDA)" — full end-to-end
  GPU install path with explicit hardware requirements, WSL2 caveat,
  install verification snippet, and a config example. Verified on
  GTX 1650 + WSL2, `torch 2.6.0+cu124`.
- `docs/mcp.md` § "API stability (SemVer)" — explicit list of what is
  SemVer-protected on the MCP surface for the v1.x line (tools,
  argument schema, response shape, degraded-mode contract) and what
  is not (internal modules, log format, HTML markup).
- Triage: no `# TODO` / `# FIXME` / `# XXX` / `# HACK` left in
  `src/`, `scripts/`, or `tests/`.

### Added (Sprint 3: production observability and operability)
- `/health/live` (always 200 while the process answers) and
  `/health/ready` (503 if no manifest or a background ingest/rebuild is
  in flight). The legacy `/health` keeps returning 200 and now carries
  `ready` and `reasons` fields for clients that want to migrate.
- `/metrics` endpoint in Prometheus exposition format. Counters:
  `doc_rag_mcp_requests_total{tool,status}`, `doc_rag_ingest_documents_total`,
  `doc_rag_ingest_errors_total`. Gauge: `doc_rag_faiss_index_size`.
  Histogram: `doc_rag_mcp_request_duration_seconds`. Requires the new
  `[metrics]` extra (`pip install -e .[metrics]`); without it the
  endpoint returns 503 with an informative body.
- `manifest.schema_version` (currently `1`) stamped on every write.
  Reads refuse manifests with a higher schema version than the running
  build supports, with an actionable error pointing at `doc-rag migrate`.
  `ManifestSchemaTooNew` is raised at the boundary so callers can react.
- `doc-rag migrate` subcommand stub. Prints the supported and detected
  schema version and exits cleanly when no migrations are defined; the
  surface exists so future migrations have a stable place to land.
- `build/audit.log` — append-only JSONL of destructive operations
  (`delete`, `wipe`, `clean_orphans`, `clear_incoming`). Schema version
  pinned at `1`; the file is best-effort (I/O errors do not propagate).
- `scripts/backup.sh` and `scripts/restore.sh`. Backups carry an
  embedded `MANIFEST.sha256` and `restore.sh` refuses to overwrite a
  populated `build/` without `--force`. Verified end-to-end on a
  synthetic corpus.
- Graceful-shutdown configuration: `scripts/run_mcp_http.sh` now passes
  `--timeout-graceful-shutdown` to uvicorn (default 30 s, override via
  `DOC_RAG_SHUTDOWN_TIMEOUT`). The systemd unit declares matching
  `KillSignal=SIGTERM` and `TimeoutStopSec=60`.

### Added — tests (Sprint 3)
- `tests/test_health_metrics.py` — live/ready/legacy/metrics endpoint
  contract.
- `tests/test_audit_log.py` — one record per destructive op, JSONL
  parseability, `read_recent` tail.
- `tests/test_schema_version.py` — guard accepts current/legacy,
  refuses future, `doc-rag migrate` CLI smoke test.

### Added (Sprint 2: test coverage)
- Reusable test fixtures in `tests/conftest.py`: `tmp_corpus_root`,
  `synthetic_chunks`, `synthetic_embeddings`, `built_corpus`,
  `make_md`, `make_txt`, `make_docx`.
- Parser tests for `.md`, `.txt`, `.docx` (with table), `.doc`
  (against bundled fixture if antiword is on PATH), `.pdf` (text mode
  via PyMuPDF).
- CLI destructive-op tests: `delete`, `wipe` (refuses without
  `--confirm DELETE`), `clean-orphans`, `clear-incoming`, plus CLI
  wrapper smoke tests via subprocess.
- FAISS reconstruct regression test — the headline invariant that
  remaining vectors after prune are byte-identical to originals.
- Degraded-mode contract tests: `semantic_search` returns `None` when
  the index is missing; `doc_search_tool` prepends the warning
  content-item; no warning when retrieval mode is `lexical`.
- Upload dedup tests: same payload twice in one batch, re-upload of an
  already-archived file, sanity that distinct payloads pass through.
- `tests/fixtures/sample.docx` — a realistic ~44 kB Word document with
  headings, paragraphs, bold/italic runs, bulleted and numbered lists,
  a 4×3 table, and an embedded PNG schematic.
- `tests/fixtures/sample.doc` — legacy Word 97 binary derived from the
  same source, enables the `.doc` parser test.
- `tests/fixtures/_build_sample_docx.py` — committed generator so the
  fixture can be rebuilt.

### Fixed (Sprint 2)
- `tests/test_mcp_http.py::test_ui_multi_upload_writes_incoming` — the
  fixture payloads `batch_a.pdf` and `batch_b.pdf` now have distinct
  sha256s, so the test exercises happy-path multi-upload as intended
  (the dedup behaviour is covered separately in `tests/test_dedup.py`).

### Changed (Sprint 2)
- Coverage gate enforced in CI at 40 % (doubled from the pre-sprint
  baseline of 22 %; 70 % deferred to Sprint 3 after a `pipeline.py`
  testability refactor — see `docs/roadmap.md`).

### Added (Sprint 1: legal, CI, observability foundation)
- `LICENSE` (AGPL-3.0-or-later) and `NOTICE` listing third-party
  dependencies with their licenses.
- `SECURITY.md` with vulnerability reporting policy and response SLAs.
- `CONTRIBUTING.md` with dev setup, test instructions, commit style, and
  the AGPL contributor agreement.
- `docs/roadmap.md` — master plan with SemVer policy and acceptance gates
  for the path to v1.x.y public release.
- GitHub Actions CI: `.github/workflows/tests.yml` (pytest on Python
  3.10/3.11/3.12 with coverage), `.github/workflows/lint.yml`
  (`ruff check` + `ruff format --check`), `.github/workflows/build.yml`
  (Docker build smoke test).
- Structured logging (`src/doc_rag/server/logging_setup.py`): selectable
  `text` / `json` format via `DOC_RAG_LOG_FORMAT`, log level via
  `DOC_RAG_LOG_LEVEL`, per-request `X-Request-ID` correlation id
  propagated into every log line emitted during a request and echoed
  back to clients in the response header.
- Ruff config and pytest config blocks in `pyproject.toml`.

### Added
- `.md`, `.txt`, and legacy `.doc` (via `antiword` / `catdoc`) as first-class source
  formats alongside `.pdf` and `.docx`.
- CLI commands for content management: `doc-rag delete <doc_id> ...`,
  `doc-rag wipe --confirm DELETE`, `doc-rag clean-orphans`, `doc-rag clear-incoming`.
- Matching Web UI controls: per-row delete, bulk-delete with checkboxes, and a
  "Danger zone" card exposing wipe / clean-orphans / clear-incoming.
- Smart FAISS rebuild on delete — remaining vectors are reconstructed from the existing
  index instead of being re-encoded from scratch.
- Upload-time deduplication: files whose sha256 already exists in the manifest or in the
  incoming queue are skipped and reported in a yellow banner.
- Degraded-mode signalling: when semantic search is configured but FAISS is unavailable,
  `doc_search` falls back to lexical search and prepends a warning content-item; the UI
  shows a persistent banner with a one-click "Запустить rebuild" button. The banner
  auto-clears once the index is back.
- `antiword` installed by `docker/Dockerfile` and `scripts/install_server_native.sh`.

### Fixed
- `semantic_search` no longer triggers a FAISS rebuild from inside an HTTP request
  (a single missing-index search could hang for hours and pin all CPUs). The function
  now returns `None` and the caller falls back to lexical search.
- Stale FAISS index files are removed before a rebuild starts.
- `delete_documents` tolerates manifest path drift between `sources/incoming/` and
  `sources/archived/` after archive moves.

### Changed
- **License switched from MIT to AGPL-3.0-or-later.** This is consistent
  with the project's use of PyMuPDF (AGPL) in the default Docker image
  and native installer. See `LICENSE` and `NOTICE` for details.
- `pyproject.toml` license expression updated to PEP 639 form; build
  requires `setuptools>=77`.
- `README.md` now in English; project layout, license, and "Talks and
  articles" sections added.
- `docs/{mcp,ui,troubleshooting}.md` — UI button labels are now shown in
  both Russian (as in the interface) and English for international
  readers.
- README split into focused docs under `docs/` (install, cli, mcp, ui, deploy,
  troubleshooting); the README is now an overview + quickstart.
- Removed empty leftover packages `server/` and `tools/raglib/` (legacy stubs from an
  earlier refactor).

## 2025-05

- Table extraction improvements + fuzzy chunk deduplication (`2cb364c`).
- OCR pipeline: install + verification in bootstrap and native installer
  (`bcedf85`, `29739b4`).
- Manifest stores OCR coverage and parser metadata (`9c45a81`).
- UI: button accessibility and state management (`e155c54`).
- Document indexing and preview features in the MCP HTTP server (`94ee4dd`).
- Logging refactor; logrotate dependency dropped (`59ed435`).
- Multi-file upload and ingest from the UI (`7daa756`).

## 2025-04

- Initial commit: doc-rag MCP HTTP server with Web UI (`3425b6f`).
