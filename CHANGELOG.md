# Changelog

All notable changes to `doc-rag` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project does not yet ship versioned tags, so entries are grouped by date.

## Unreleased — toward v1.1.0 (first public release)

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
