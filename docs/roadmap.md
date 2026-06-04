# Roadmap to public v1.x.y

This document is the master plan for taking `doc-rag` from its current state
(a working local RAG with private code at version `1.0.0`) to a
production-ready, publicly published release. It is organised around
four sprints. Each sprint has an acceptance gate; the gate decides
whether the next sprint can start.

The document itself is normative — if the code and this roadmap disagree,
update one of them deliberately, do not let them drift.

## 1. Versioning policy

`doc-rag` follows **[Semantic Versioning 2.0.0](https://semver.org/)**
strictly. Versions are `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible changes. Includes:
  - removing or renaming a CLI subcommand;
  - changing the shape of `build/manifest.json` such that older installs
    cannot read it;
  - changing the FAISS index serialization in a way that requires a
    re-encode (this is what `manifest.schema_version` will track from
    Sprint 3);
  - removing or renaming an MCP tool exposed by the server;
  - dropping support for a Python version that was previously supported.
- **MINOR** — backwards-compatible features. New CLI subcommand, new MCP
  tool, new optional config key, new supported source format.
- **PATCH** — backwards-compatible fixes. Bug fixes, documentation, CI,
  dependency bumps within their own SemVer range.

### Public surface

The following are SemVer-protected:

1. The CLI: `doc-rag {ingest,rebuild,serve,delete,wipe,clean-orphans,clear-incoming}`
   — names, positional argument shapes, exit codes.
2. The MCP tool surface exposed over `POST /mcp` (currently: `doc_search`).
3. The HTTP endpoints used by the Web UI (`/ui/*`) — their **paths**, not
   their HTML markup.
4. `config/config.yaml` — top-level keys and their semantics.
5. The on-disk layout of `build/` — manifest schema, chunks JSONL schema,
   FAISS index serialization.

The following are explicitly **not** SemVer-protected and may change in
any release:

- Internal Python module structure under `src/doc_rag/`.
- Log line formats (machine-readable JSON shape, when introduced in Sprint 1,
  is governed by an explicit schema with its own version field).
- Bundled `.cursor/mcp.json` and equivalent client config files
  (the *fields* are governed by the MCP spec, not by us).

### Release cadence and tagging

- Releases are tagged `vX.Y.Z` on `main` after CI is green and `CHANGELOG.md`
  has an entry under that version.
- Pre-1.x releases are not anticipated — the project is already `1.0.0`
  internally. The first **public** release will be the first tag after the
  Sprint 1 gate passes (target: `1.1.0`).
- Patch releases (`1.1.x`) are cut as needed; minor releases follow each
  sprint that ships user-facing features.

### Deprecation policy

When something on the public surface needs to go away:

1. The replacement lands in `MINOR` release `N`. The old thing keeps working.
2. A deprecation notice goes into `CHANGELOG.md` and into the running tool
   output (CLI warning, MCP response warning, UI banner — same
   degraded-mode signalling pattern we already use).
3. Removal happens no earlier than the next `MAJOR` release.

## 2. Sprint 1 — Legal, CI, observability foundation

**Target tag:** `v1.1.0` (first public release).

**Acceptance gate:** the project is legally publishable on GitHub
without ambiguity; an external contributor can clone it, run tests, and
see CI status; logs are machine-readable.

### Sprint 1.1 — Legal

- [ ] `LICENSE` — full text of AGPL-3.0-or-later at repo root.
- [ ] `pyproject.toml` — `license = "AGPL-3.0-or-later"` plus matching
      `classifiers`.
- [ ] `NOTICE` — third-party dependencies and their licenses (the
      project is AGPL, but bundled deps span MIT, BSD, Apache-2.0,
      MPL-2.0, HPND, etc. — make this visible).
- [ ] `SECURITY.md` — supported versions, vulnerability reporting
      address, response SLA.
- [ ] `CONTRIBUTING.md` — dev setup, test instructions, commit style,
      explicit note that contributions are accepted under AGPL.

### Sprint 1.2 — CI

- [ ] `.github/workflows/tests.yml` — pytest on Python 3.10/3.11/3.12,
      with `[server,faiss,pdf]` extras installed. Runs on push and PR.
- [ ] `.github/workflows/lint.yml` — `ruff check` + `ruff format --check`.
      Ruff config in `pyproject.toml`.
- [ ] `.github/workflows/build.yml` — `docker build` against the bundled
      `docker/Dockerfile`. Runs on push and PR.
- [ ] Status badges in `README.md`.

### Sprint 1.3 — Observability foundation

- [ ] Structured logging: introduce `python-json-logger`. Two modes
      selectable via `DOC_RAG_LOG_FORMAT` (`text` default, `json` for
      production).
- [ ] Correlation ID middleware in `mcp_http.py` — `request_id` (uuid4)
      injected into every log line covering a request.
- [ ] `DOC_RAG_LOG_LEVEL` env var (default `INFO`).
- [ ] All explicit `print()` and ad-hoc `logging.info` calls in the
      server path audited and routed through the new logger.

### Sprint 1.4 — Documentation language switch

- [ ] `README.md` translated to English. Slim overview format
      preserved. AGPL notice added. Link to the published Habr article
      added under "Talks and articles".
- [ ] `docs/install.md`, `docs/cli.md`, `docs/mcp.md`, `docs/ui.md`,
      `docs/deploy.md`, `docs/troubleshooting.md` translated to English.
- [ ] `CHANGELOG.md` updated for `1.1.0` with all sprint-1 changes.
- [ ] Russian copies tracked under `docs/ru/` and may lag behind by
      design; only the English versions are considered authoritative.

## 3. Sprint 2 — Test coverage

**Target tag:** `v1.2.0`.

**Acceptance gate:** branch coverage on `src/doc_rag/raglib/` and
`src/doc_rag/server/retrieval.py` ≥ 70 %; destructive CLI operations
have at least one happy-path test each; FAISS reconstruct path has a
regression test.

### Items

- [ ] Parser tests for every supported format: `.pdf` (text), `.pdf`
      (scanned, requires Tesseract), `.docx` (with table), `.doc` (via
      antiword), `.md`, `.txt`. Fixtures live under `tests/fixtures/`.
- [ ] CLI tests for `delete`, `wipe`, `clean-orphans`, `clear-incoming` —
      each operation has a happy-path test against an isolated
      temp-rooted corpus.
- [ ] **FAISS reconstruct regression test** — given an index with N
      chunks, deleting K of them must produce an index whose remaining
      `index.reconstruct(i)` vectors match the originals byte-for-byte.
- [ ] **Degraded-mode contract test** — with `faiss.index` removed,
      `doc_search` returns a content array whose first item is the
      configured warning text.
- [ ] **Dedup test** — uploading the same sha256 twice via `/ui/upload`
      yields a 200 with the dup reported, not a second manifest entry.
- [ ] Coverage report uploaded to CI artefacts; ≥ 70 % gate enforced in
      `tests.yml`.

## 4. Sprint 3 — Production observability and operability

**Target tag:** `v1.3.0`.

**Acceptance gate:** a fresh production install can be monitored by a
generic Prometheus scrape; backup/restore is documented and tested; an
operator can tell whether an instance is alive vs. ready.

### Items

- [ ] Split `/health` into `/health/live` (always 200) and
      `/health/ready` (200 only when manifest exists and no rebuild is
      in flight; 503 otherwise).
- [ ] `/metrics` endpoint in Prometheus exposition format via
      `prometheus-client`. Counters: `mcp_requests_total{tool,status}`,
      `ingest_documents_total`, `ingest_errors_total`. Gauges:
      `faiss_index_size`. Histograms: `embedding_duration_seconds`,
      `mcp_request_duration_seconds`.
- [ ] `manifest.schema_version: 1` introduced. `doc-rag` refuses to
      operate on a manifest with a higher schema version than it
      understands and prints an actionable message.
- [ ] `doc-rag migrate` subcommand stub — empty handler, but the
      surface exists so future migrations have a place to land.
- [ ] `build/audit.log` — append-only line-delimited JSON, one entry
      per destructive operation: timestamp, operation, doc_ids if
      relevant, principal (if auth enabled), counts before/after.
- [ ] `scripts/backup.sh` — produces a timestamped tar.gz from
      `build/manifest.json`, `build/chunks_jsonl/`, `build/index/`,
      `build/embeddings/`, optionally `sources/archived/`. Includes a
      `MANIFEST.sha256` file inside.
- [ ] `scripts/restore.sh` — extracts a backup, validates sha256s,
      refuses to overwrite without `--force`.
- [ ] Graceful shutdown verified: SIGTERM during a request finishes the
      in-flight request within `DOC_RAG_SHUTDOWN_TIMEOUT` (default 30 s).
      Background ingest/rebuild interrupt cleanly (no half-written index).

## 5. Sprint 4 — Performance, polish, public release

**Target tag:** `v1.4.0`.

**Acceptance gate:** repeatable benchmarks committed; GPU install path
documented end-to-end; CHANGELOG, README, and docs are coherent;
public release announced.

### Items

- [ ] `scripts/bench.py` — produces JSON with: time per document parse,
      time per chunk to embed (CPU and GPU paths), full-corpus rebuild
      time at sizes 1 000 / 10 000 / 50 000 chunks (use synthetic
      Lorem-Ipsum-equivalent data).
- [ ] GPU install path end-to-end documented and tested at least once.
      `docs/install.md` GPU section calls out tested torch/CUDA combos.
- [ ] API stability section in `docs/mcp.md` makes the SemVer
      commitment explicit. The list of public MCP tools is fixed at
      `[doc_search]` for v1.x.
- [ ] All `# TODO` and `# FIXME` comments in the codebase triaged: each
      becomes a GitHub issue or is removed.
- [ ] CHANGELOG entry for `1.4.0`; GitHub release with release notes;
      public announcement on Habr or equivalent.

## 6. Out of scope for v1.x

These items are deliberately deferred and tracked separately. They are
not blockers for any v1.x.y tag.

- Multi-collection / namespaces (a future v2 feature).
- Distributed search across multiple FAISS shards.
- Web UI authentication beyond the existing single-key model
  (multi-user authn, SSO).
- Background re-embedding when the embedding model changes mid-flight.
- A "RAG quality" evaluation harness (retrieval@k, faithfulness scores).
- A first-party hosted variant.

## 7. How to update this roadmap

This file is part of the public surface in spirit, not in SemVer. When
priorities shift:

1. Edit this file in the same PR as the work that proves the shift.
2. Move items between sprints rather than deleting them.
3. Acceptance gates may be tightened but not loosened without an
   explicit note in `CHANGELOG.md`.
