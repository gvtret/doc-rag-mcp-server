# Changelog

All notable changes to `doc-rag` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project does not yet ship versioned tags, so entries are grouped by date.

## Unreleased

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
