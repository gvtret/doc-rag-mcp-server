# Roadmap

This document is the master plan for taking `doc-rag` from its initial
internal state (a working local RAG, private code, internal version
`1.0.0`) to a production-ready, publicly published release line whose
first published tag was `v1.4.0` — the cumulative result of Sprints 1
through 4. **v2.0.0** then collapsed the post-v1.4 plan: instead of a
gradual `v1.5 (Docling opt-in) → v1.6 (cascade) → v1.7 (QA) → … → v2.0
(vector store)` cadence, the Docling-only switch was accelerated as a
single SemVer-major release that also relicensed the project from
AGPL-3.0-or-later to MIT (PyMuPDF was the only AGPL-load-bearing
dependency).

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

1. The CLI: `doc-rag {ingest,rebuild,rebuild-docs,rebuild-index,serve,delete,wipe,
   clean-orphans,clear-incoming,migrate-vectors,migrate}`
   — names, positional argument shapes, exit codes.
2. The MCP tool surface exposed over `POST /mcp` (currently: `doc_search`,
   `doc_download`, `doc_search_docs`).
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
- Pre-1.x releases are not anticipated. Internally the project moved
  through `v1.0.0` (pre-publication baseline) and the intermediate
  Sprint-1/2/3 milestones `v1.1.0` / `v1.2.0` / `v1.3.0`, none of
  which were ever published. The first **public** tag is `v1.4.0`,
  which is what users get when they clone or download the released
  source tree.
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

- [x] `LICENSE` — full text of AGPL-3.0-or-later at repo root. (Superseded:
      relicensed to MIT in v2.0.0, see § "v1.5 → v2.0" below — `LICENSE` now
      carries MIT text, `pyproject.toml` `license = "MIT"`.)
- [x] `pyproject.toml` — `license = "AGPL-3.0-or-later"` plus matching
      `classifiers`. (Superseded: now `license = "MIT"`, see above.)
- [x] `NOTICE` — third-party dependencies and their licenses (the
      project is AGPL, but bundled deps span MIT, BSD, Apache-2.0,
      MPL-2.0, HPND, etc. — make this visible).
- [x] `SECURITY.md` — supported versions, vulnerability reporting
      address, response SLA.
- [x] `CONTRIBUTING.md` — dev setup, test instructions, commit style,
      explicit note that contributions are accepted under AGPL.

### Sprint 1.2 — CI

- [x] `.github/workflows/tests.yml` — pytest on Python 3.10/3.11/3.12,
      with `[server,faiss,pdf]` extras installed. Runs on push and PR.
      (3.10 dropped in v5.0.0 — incompatible wheels in `uv.lock`; matrix
      is now 3.11/3.12.)
- [x] `.github/workflows/lint.yml` — `ruff check` + `ruff format --check`.
      Ruff config in `pyproject.toml`.
- [x] `.github/workflows/build.yml` — `docker build` against the bundled
      `docker/Dockerfile`. Runs on push and PR.
- [x] Status badges in `README.md`.

### Sprint 1.3 — Observability foundation

- [x] Structured logging: introduce `python-json-logger`. Two modes
      selectable via `DOC_RAG_LOG_FORMAT` (`text` default, `json` for
      production).
- [x] Correlation ID middleware in `mcp_http.py` — `request_id` (uuid4)
      injected into every log line covering a request.
- [x] `DOC_RAG_LOG_LEVEL` env var (default `INFO`).
- [x] All explicit `print()` and ad-hoc `logging.info` calls in the
      server path audited and routed through the new logger.

### Sprint 1.4 — Documentation language switch

- [x] `README.md` translated to English. Slim overview format
      preserved. Link to the published Habr article added under
      "Talks and articles".
- [x] `docs/install.md`, `docs/cli.md`, `docs/mcp.md`, `docs/ui.md`,
      `docs/deploy.md`, `docs/troubleshooting.md` translated to English.
- [ ] `CHANGELOG.md` updated for `1.1.0` with all sprint-1 changes.
      (Moot: `v1.1.0`/`v1.2.0`/`v1.3.0` were internal-only and never
      published — see § "Release cadence and tagging" above. The first
      public tag `v1.4.0` has its own CHANGELOG entry instead.)
- [ ] Russian copies tracked under `docs/ru/` and may lag behind by
      design; only the English versions are considered authoritative.
      (Not done: `docs/ru/` was never created; docs are English-only.)

## 3. Sprint 2 — Test coverage

**Target tag:** `v1.2.0`.

**Acceptance gate:** total branch coverage of the `doc_rag` package
≥ 40 %, doubled from the pre-sprint baseline of 22 %; destructive CLI
operations have at least one happy-path test each; FAISS reconstruct
path has a regression test; degraded-mode contract and upload dedup
each have a contract test.

The original 70 % goal turned out to be unreachable in one sprint
without first refactoring `pipeline.py` (605 statements) and
`indexer.py` (149 statements) to be easily testable without a live
embedding model. That refactor is deferred to Sprint 3.

### Items

- [x] Parser tests for every supported format: `.pdf` (text), `.pdf`
      (scanned, requires Tesseract), `.docx` (with table), `.doc` (via
      antiword), `.md`, `.txt`. Fixtures live under `tests/fixtures/`.
- [x] CLI tests for `delete`, `wipe`, `clean-orphans`, `clear-incoming` —
      each operation has a happy-path test against an isolated
      temp-rooted corpus. (`tests/test_cli_destructive.py`.)
- [x] **FAISS reconstruct regression test** — given an index with N
      chunks, deleting K of them must produce an index whose remaining
      `index.reconstruct(i)` vectors match the originals byte-for-byte.
      (`tests/test_faiss_reconstruct.py`.)
- [x] **Degraded-mode contract test** — with `faiss.index` removed,
      `doc_search` returns a content array whose first item is the
      configured warning text. (`tests/test_degraded_mode.py`.)
- [x] **Dedup test** — uploading the same sha256 twice via `/ui/upload`
      yields a 200 with the dup reported, not a second manifest entry.
      (`tests/test_dedup.py`.)
- [x] Coverage report uploaded to CI artefacts; ≥ 40 % gate enforced in
      `tests.yml` (`--cov-fail-under=40`). Raising the gate is a Sprint 3
      task tied to a `pipeline.py` testability refactor.

## 4. Sprint 3 — Production observability and operability

**Target tag:** `v1.3.0`.

**Acceptance gate:** a fresh production install can be monitored by a
generic Prometheus scrape; backup/restore is documented and tested; an
operator can tell whether an instance is alive vs. ready.

### Items

- [x] Split `/health` into `/health/live` (always 200) and
      `/health/ready` (200 only when manifest exists and no rebuild is
      in flight; 503 otherwise).
- [x] `/metrics` endpoint in Prometheus exposition format via
      `prometheus-client`. Counters: `mcp_requests_total{tool,status}`,
      `ingest_documents_total`, `ingest_errors_total`. Gauges:
      `faiss_index_size`. Histograms: `embedding_duration_seconds`,
      `mcp_request_duration_seconds`.
- [x] `manifest.schema_version: 1` introduced. `doc-rag` refuses to
      operate on a manifest with a higher schema version than it
      understands and prints an actionable message. (`tests/test_schema_version.py`.)
- [x] `doc-rag migrate` subcommand stub — empty handler, but the
      surface exists so future migrations have a place to land.
- [x] `build/audit.log` — append-only line-delimited JSON, one entry
      per destructive operation: timestamp, operation, doc_ids if
      relevant, principal (if auth enabled), counts before/after.
      (`tests/test_audit_log.py`.)
- [x] `scripts/backup.sh` — produces a timestamped tar.gz from
      `build/manifest.json`, `build/chunks_jsonl/`, `build/index/`,
      `build/embeddings/`, optionally `sources/archived/`. Includes a
      `MANIFEST.sha256` file inside.
- [x] `scripts/restore.sh` — extracts a backup, validates sha256s,
      refuses to overwrite without `--force`.
- [x] Graceful shutdown verified: SIGTERM during a request finishes the
      in-flight request within `DOC_RAG_SHUTDOWN_TIMEOUT` (default 30 s).
      Background ingest/rebuild interrupt cleanly (no half-written index).

## 5. Sprint 4 — Performance, polish, public release

**Target tag:** `v1.4.0`.

**Acceptance gate:** repeatable benchmarks committed; GPU install path
documented end-to-end; CHANGELOG, README, and docs are coherent;
public release announced.

### Items

- [x] `scripts/bench.py` — produces JSON with: time per document parse,
      time per chunk to embed (CPU and GPU paths), full-corpus rebuild
      time at sizes 1 000 / 10 000 / 50 000 chunks (use synthetic
      Lorem-Ipsum-equivalent data). (Results: `docs/bench-results.md`.)
- [x] GPU install path end-to-end documented and tested at least once.
      `docs/install.md` GPU section calls out tested torch/CUDA combos.
      (`docs/install.md` § "GPU install (NVIDIA / CUDA)" / § "GPU install
      (AMD / ROCm)" now exist and match the headings § 7 below references.
      CUDA: verified on GTX 1650/WSL2 per § 7. ROCm: documented — manual
      `pip install` against the matching ROCm wheel index, no bundled
      script, not independently tested here.)
- [x] API stability section in `docs/mcp.md` makes the SemVer
      commitment explicit. (Tool list has since grown to `[doc_search,
      doc_download, doc_search_docs]` — additions are MINOR bumps per
      the policy in § 1, so this doesn't violate the contract; wording
      updated to match current state.)
- [x] All `# TODO` and `# FIXME` comments in the codebase triaged: each
      becomes a GitHub issue or is removed. (None found under `src/`.)
- [x] CHANGELOG entry for `1.4.0`; GitHub release with release notes;
      public announcement on Habr or equivalent.

## 6. Post-v1.x: planned major work

Each version below is a coherent slice of the broader **production
ingest pipeline overhaul** that the project is moving towards. They
land in order — later items depend on earlier ones (most obviously
v2.6's recursive chunker depends on the typed blocks layer introduced
in v2.0). Each is its own MINOR release with its own acceptance gate.

The motivation for this overhaul, in one sentence: the current pipeline
turns documents into unstructured Markdown and then into fixed-size
chunks. Both steps lose information that, if preserved, would visibly
improve retrieval quality on the project's primary corpus (СТО / ГОСТы
/ specifications), where tables, formulas, multi-column layouts, and
explicit section hierarchies are the rule rather than the exception.

### v1.5 → v2.0 (released) — Docling-only parsing, MIT license

Originally planned as an opt-in Docling backend alongside PyMuPDF.
After a comparative test on СТО 34.01-5.1-006 (including the
multi-page table in Приложение И) the asymmetry was decisive: Docling
yielded 17 structured tables and 12 headings where PyMuPDF returned
zero structure (one paragraph). The pivot:

- **PyMuPDF and PyPDF2 removed entirely.** Docling is the only PDF
  backend, shipped as a base dependency (no `[docling]` extra).
- **License pivoted to MIT** in the same release — PyMuPDF was the
  sole AGPL-load-bearing dep.
- **`build/blocks/<doc_id>.jsonl`** intermediate layer (schema v1) and
  **filetype magic-bytes routing** were absorbed into v2.0.

Wall-time cost: ~10-20 s/page on CPU (vs. PyMuPDF's ~0.3 s/page),
which moves "full corpus ingest" from "nightly cron" to "weekend
cron" territory. Acceptable in exchange for the structural fidelity
shown by the comparison.

Quality-checks, recursive chunker, cascade, and Unstructured fallback
keep their slots below.

### v2.1 (released) — uv migration + CI hardening

Tooling-only line. Three patch releases (`v2.1.0`–`v2.1.2`) collapsed
the install pipeline onto [uv](https://docs.astral.sh/uv/) as the only
supported package manager and moved CI off `ubuntu-latest` onto a
self-hosted runner with local on-disk caches. Highlights:

- `uv.lock` committed (148 packages locked across the 3.10-3.14 /
  Linux+macOS+Windows resolution matrix); `requirements.txt` removed.
- `torch` + `torchvision` pinned to the PyTorch CPU index — fresh
  venv drops from ~5 GB to ~1.6 GB; `nvidia-*` transitives gone.
- `faiss-cpu` capped at `<1.14` so the wheel imports on SSE4.2-only
  QEMU hosts (matches the production server).
- All 3 GitHub Actions workflows on `runs-on: self-hosted` with
  `cache-from/to: type=local`; warm-cache docker build drops from
  ~14 min to ~3.5 min.

No behavioural change to parsing, manifest, or MCP surface. Documented
here so the version line stays continuous; the originally-planned
v2.1 (Cascading parser fallback) slides to v2.3 below.

### v2.2 — Svelte-based UI (incremental, part 1)

Single-file inline HTML/JS in `src/doc_rag/server/mcp_http.py` has
grown past the point where another inline feature stops being
readable. Introduce a real frontend toolchain — [Svelte](https://svelte.dev/)
+ [Vite](https://vitejs.dev/) — and migrate the busiest page first.
Other pages stay on the server-rendered inline path; they migrate in
later versions.

Concrete deliverables:

- New top-level `ui/` directory:
  ```
  ui/
  ├── src/                    # Svelte components, state stores, API client
  ├── package.json
  ├── vite.config.ts
  └── dist/                   # build output, served by FastAPI
  ```
- `app.mount("/ui", StaticFiles(directory="ui/dist"))` in
  `mcp_http.py` replacing the inline `_render_ui` handler for the
  main page.
- Vite dev server (port 5173) with proxy `/api/*` → 3333 for hot
  reload during development; production is a static bundle.
- API contract is the existing JSON `/ui/status` + `/ui/document-preview`
  + the upload/delete POST endpoints — no server-side refactor.
- Migrated page: the main `/ui` (status panel, document table,
  ingest/rebuild buttons, progress line, OCR badge, semantic-search
  banner, danger zone). Doc-preview modal migrates with it because
  it's the same page.
- Not migrated in v2.2: `/ui/logs`-style log tails, `/ui/status` raw
  JSON view, MCP-config download endpoints. These stay on the inline
  path; migration of the remaining pages is scheduled across follow-up
  MINOR releases (exact version pinning happens when each page lands).
  **Update, post-v5.1**: by v3.1 (Manage + Logs pages, below) every page
  had a Svelte equivalent, leaving the inline `GET /ui` handler
  (`_render_ui` and its HTML-building helpers) unreachable dead weight —
  it was deleted outright, along with the optional-Node fallback in
  `bootstrap.sh`/`docker/Dockerfile`. `/ui` is now a redirect to
  `/ui-next/`; the Svelte build is a hard install requirement.
- Build pipeline: `npm run build` inside `ui/`; `scripts/bootstrap.sh`
  installs Node via `npm install` if a `package.json` is detected;
  `docker/Dockerfile` runs the build as a multi-stage layer; CI
  builds the bundle and runs `npm run lint` / `npm run check`.

Acceptance:

- Visiting `/ui` after deploy serves the Svelte bundle — same
  visual surface, same functionality (verified by manually walking
  through ingest → search → delete on the staging server).
- `_render_ui` and the dependent JS-string helpers for the migrated
  page removed from `mcp_http.py`.
- CI gains an `npm-build` job; bundle is cached on the runner host
  alongside the existing `uv` and `buildx` caches.
- README + `docs/install.md` updated with `Node >= 20` prerequisite
  and the new `ui/` build step.

### v2.3 — Cascading parser fallback

Add [Unstructured](https://unstructured.io/) (`hi_res` strategy) as
a second-tier fallback for PDFs that Docling cannot handle (broken
layout, exotic fonts, OCR-only scans where Docling's preprocessing
fails). Cascade order (no AGPL backends — they were removed in v2.0):

```
Docling → Unstructured (hi_res)
```

The cascade triggers on (a) parser exception, or (b) quality score
below a configurable threshold (see v2.4 for the quality module).
**Note:** cascade depends on the quality scoring introduced in v2.4 —
either we ship cascade first with only the "parser exception" trigger
and bolt on the score-based trigger in v2.4, or we swap the order and
ship v2.4 first. Both options are open; sequence pinned at sprint
start.

Concrete deliverables:

- New config: `pdf_backend: cascade`.
- Logging line emitted on each fallback hop, with the reason.
- Per-document `source_backend` field in blocks/manifest so a reader
  can tell which backend produced each chunk.

Acceptance:

- On a curated set of "problematic" PDFs (multi-column, heavy tables,
  formula-rich), cascade extracts non-empty text from ≥ 90 % of them
  vs. ≤ 60 % for any single backend alone.

### v2.4 — Document quality checks + per-doc reports

Mandatory pre-indexing QA on the typed blocks. The pipeline emits
`build/quality/<doc_id>.json` for every document and a roll-up under
`build/quality/summary.json`.

Checks (each yields a severity and a human-readable message):

- Empty pages and pages with suspiciously low text density.
- Broken tables (empty-cells ratio above threshold, mismatched column
  counts row-to-row).
- Lost images (count detected by parser vs. count of figure-blocks).
- Formula garbage (heuristic: fraction of unknown Unicode blocks).
- Reading-order anomalies (positional vs. logical order mismatches).
- Duplicate headers and footers (likely header/footer pollution).
- Unreadable-character percentage above threshold.

Report shape (`schema_version: 1`):

```json
{
  "doc_id": "...",
  "pages": 120,
  "blocks": {"heading": 24, "paragraph": 706, "table": 24,
             "figure": 17, "formula": 11},
  "warnings": [
    {"severity": "info",    "code": "low_text_density",
     "page": 45, "message": "..."},
    {"severity": "warn",    "code": "broken_table",
     "block_id": "...",   "message": "table 12: empty cells ratio > 40%"}
  ],
  "score": 0.87,
  "schema_version": 1
}
```

Concrete deliverables:

- `src/doc_rag/raglib/quality.py` module.
- UI surfacing: badge on each document row in the Web UI table (green
  / yellow / red) tied to severity.
- Manifest entry: `quality_score`, `quality_warning_count`.
- New config: `quality.fail_on_severity: error | warn | never`
  (default `never` — warnings do not block ingest).

Acceptance:

- All seven check types have unit tests with synthetic block fixtures.
- A document with a known broken table is correctly flagged.
- UI shows the badge.

### v2.5 — Unified DOC / DOCX ingest path

Replace the current `antiword` shell-out for `.doc` with a
LibreOffice-headless DOC→DOCX conversion in the pipeline, then let
either python-docx (lightweight) or Docling (rich) handle the DOCX.

Concrete deliverables:

- New config: `docx_backend: python-docx | docling`.
- LibreOffice headless invoked as a subprocess only for `.doc`; the
  resulting DOCX flows through the same path as a native `.docx`.
- `antiword` removed from default install path (its Apache-2.0 vs.
  GPL-2.0 license is irrelevant; the licence-clean alternative is
  python-docx).

Acceptance:

- `.doc` parser test from `tests/test_parsers.py` passes against the
  new path.
- Existing `sample.doc` fixture remains the regression target.

### v2.6 — Structure-aware (recursive) chunker

Now trivial because `blocks.jsonl` from v2.0 supplies the structure.
A recursive splitter walks the typed blocks, grouping siblings into
chunks while respecting headings as hard boundaries and tables as
atomic units.

Concrete deliverables:

- `src/doc_rag/raglib/chunker_recursive.py`.
- New config: `chunking.strategy: fixed | recursive` (default
  `recursive` once retrieval-quality benchmarks confirm no regression
  on a curated Q/A set; otherwise default stays `fixed`).
- Section heading carried as `chunk.section_path` metadata, exposed
  in `doc_search` results so the LLM has scope context.
- Tables stay as single chunks even when large, with a smaller
  "table_summary" sibling chunk so retrieval can still match on a
  query that does not name a specific cell.
- Coverage gate raised to 70 % (the long-deferred Sprint 2 target —
  feasible now because `pipeline.py` can be tested at the blocks layer
  without a real embedding model).

Acceptance:

- Retrieval@k on a small curated Q/A set ≥ baseline (fixed-size
  chunker on PyMuPDF blocks).
- Section headings appear in 95 % of returned chunks when the source
  document has any heading structure at all.

### v3.0 — Vector-store backends and namespaces

Pluggable vector store with FAISS as the default. Opt-in backends:
[Qdrant](https://qdrant.tech/) (hybrid search, payload filters,
multi-collection) and [pgvector](https://github.com/pgvector/pgvector)
(when the operator already runs Postgres). Adds the long-deferred
multi-collection / namespace concept — for example, separate
collections for СТО vs. РД vs. internal regulations.

Concrete deliverables:

- New config: `vector_store: faiss | qdrant | pgvector`.
- Migration tool: `doc-rag migrate --from faiss --to qdrant`.
- Namespace concept: documents and queries can be scoped to a
  collection.
- Hybrid search wired in for Qdrant backend (vector + metadata
  filter, e.g. `source LIKE 'СТО 34.*' AND section_path LIKE '5.%'`).

Acceptance:

- All three backends pass the existing `doc_search` contract test.
- Qdrant backend demonstrates a hybrid query that pure FAISS cannot
  answer.
- Schema migration v1 → v2 actually moves a corpus end-to-end.

### v3.1 — Hybrid search, RAG generate, quality badge

Incremental release building on v3.0 vector-store foundations. Ships
three features that were developed together and share a "retrieval →
consumption" arc:

- **Hybrid search (RRF)** — `retrieval_mode: hybrid` runs lexical
  (TF-IDF) and semantic search in parallel, merges via Reciprocal Rank
  Fusion (k=60). New config key `mcp.retrieval_mode` (default
  `semantic`). All three backends support hybrid mode.
- **RAG generate endpoint** — `POST /api/v1/generate` calls
  `doc_search()` to assemble context, formats numbered citations, and
  forwards to an OpenAI-compatible LLM API. LLM config via env vars
  (`DOC_RAG_LLM_BASE_URL`, `DOC_RAG_LLM_MODEL`, `DOC_RAG_LLM_API_KEY`)
  or config keys under `mcp.rag_generate.*`.
- **Quality badge** — per-document quality score surfaced as a
  green/yellow/red badge in the UI document table. `/ui/quality`
  endpoint returns scores and warning counts.
- **Manage + Logs pages** — full Svelte replacements for the remaining
  stub pages (document management, live log viewer).

Acceptance:

- Hybrid search returns results that combine lexical phrase matches with
  semantic similarity; pure-semantic and pure-lexical modes still work.
- `/api/v1/generate` returns a coherent answer with numbered citations
  when an LLM is configured.
- Quality badge renders correctly in the UI for all severity levels.
- Manage page supports document delete, Logs page tails server logs.

### v4.0 — Full RAG pipeline

Production-grade Retrieval-Augmented Generation. Builds on v3.1's
generate endpoint by exposing it as an MCP tool, adding streaming,
structured citations, and retrieval quality improvements.

Concrete deliverables:

- **`doc_generate` MCP tool** — registered in `tools/list` with proper
  JSON Schema. Params: `query`, `top_k`, `namespace`, `max_tokens`,
  `temperature`. Returns generated answer + structured citations array.
- **Streaming responses** — SSE-based token-by-token output for both
  `POST /mcp` (MCP streaming) and `POST /api/v1/generate` (HTTP SSE).
  Non-streaming fallback preserved for clients that don't support SSE.
- **Structured citations** — `doc_search` results include a
  `citations` array with `chunk_id`, `source_file`, `section_path`,
  `page`, `score`, and `text_preview`. `doc_generate` output includes
  the same structured citations alongside the generated answer.
- **Context assembly** — configurable context window management:
  `mcp.rag_generate.max_context_tokens` (default 6000), chunk
  truncation, relevance re-ordering, deduplication of overlapping
  chunks before LLM call.
- **MCP schema completeness** — `doc_search` schema declares all
  runtime params: `namespace`, `doc_id`, `section_path`, `tables_only`,
  `filters`.
- **Config consolidation** — all `mcp.retrieval_mode`,
  `mcp.rag_generate.*` keys documented in `config/config.yaml` with
  sensible defaults.

Acceptance:

- `doc_search` returns structured citations that include section_path
  and page number.
- `doc_generate` is callable from MCP clients (Cursor, VSCode) and
  returns a generated answer with source citations.
- Streaming works end-to-end: tokens arrive incrementally, final
  response includes full citations.
- Context assembly respects `max_context_tokens` — LLM never receives
  more context than configured.
- All existing tests pass; new tests cover citations, streaming, context
  assembly, and MCP schema.

### v4.1 — Incremental rebuild, selective rebuild, CLI→UI sync

Resilience and usability release. The previous `rebuild()` was
destructive — it wiped all generated outputs and re-parsed everything
from scratch, losing all progress on power outage. This release fixes
that and adds per-document rebuild capability.

Concrete deliverables:

- **Incremental rebuild** — `rebuild()` checks manifest sha256 + disk
  presence of `.md` files. Already-processed documents are skipped.
  Interrupted rebuilds resume where they stopped. `--force` flag for
  full wipe-and-reparse.
- **`rebuild-docs` command** — `doc-rag rebuild-docs <id1> <id2>`
  rebuilds only specific documents. `POST /ui/rebuild-docs` endpoint
  for selective rebuild from the UI.
- **CLI→UI sync** — CLI detects a running server and routes
  `rebuild`/`ingest`/`rebuild-docs` through the API so the UI shows
  job progress. Falls back to direct pipeline call when server is down.
- **Selective rebuild in UI** — checkboxes on the Documents page with
  a "Rebuild selected" button. Select indexed documents to reparse
  them, or select pending files to process them.

Acceptance:

- Interrupted rebuild resumes without re-parsing already-processed
  files (verified by checking parse log lines).
- `doc-rag rebuild-docs <id>` re-parses only the specified document.
- CLI `doc-rag rebuild` when server is running shows progress in
  the terminal and the UI reflects the running job.
- All existing tests pass; new tests for incremental rebuild logic.

### v4.2 (released) — i18n, quality cascade, GPU-offload workflow, OOM prevention

Usability and resilience line. Adds a client-facing language switch, tightens the
quality-cascade trigger introduced in v2.3/v2.4, and gives operators a way to run the
expensive embedding step on a separate GPU machine and ship just the vectors back.

Highlights:

- i18n: RU/EN language switcher in the Web UI header (localStorage-persisted).
- Quality cascade wired to `parsing.quality_fallback_threshold`: Docling scores below
  threshold automatically retry with Unstructured and keep the better result.
- **`rebuild --skip-index`** / **`rebuild-index` CLI command**: parse+chunk without
  building the vector index, then build the index separately — the GPU-offload workflow
  promised back in § 7 below (process on a GPU machine, `rebuild-index` on the CPU
  server). `POST /ui/rebuild-index` / `POST /ui/rebuild-skip-index` endpoints + Manage
  page buttons.
- Chunk postprocessing: dedup repeated headers/footers and OCR artefacts, pad
  mismatched table column counts.
- Incremental manifest persistence (written after each file, not just at the end) and a
  memory guard that skips remaining chunks when free RAM drops below 1 GB —
  OOM-prevention on the CPU-only production server.
- `hmac.compare_digest` for the UI/API key comparison (timing-safe).

No acceptance gate — this is a maintenance/hardening line documented after the fact so
the version history stays continuous; see `CHANGELOG.md` § v4.2.0 for the full diff.

### v5.0 (released) — production hardening: stats page, security, health checks

**Target gate (informal):** the project is comfortable calling itself
"production-ready" — visible per-document diagnostics, a tightened security posture,
and CI that reflects the actual supported Python matrix.

Highlights:

- **Document statistics page** (`/ui/doc-stats`): per-document block-type breakdown
  with stacked bar charts, expandable detail rows, sortable columns.
- **`/health/ready` subsystem checks**: reports FAISS index availability, embedding
  model status, and free disk space — not just "manifest exists".
- **Ruff `S` (bandit) rules** enabled in CI with targeted ignores.
- Python 3.10 dropped from the CI matrix (incompatible wheels in `uv.lock`); 3.11/3.12
  only from here on — see § "Public surface" note above.
- Docker `rapidocr` packaging bug worked around (missing `arch_config.yaml` patched
  into the venv post-install).

See `CHANGELOG.md` § v5.0.0 for the full diff.

### v5.1 (released) — upload UX, delete UI, single-source versioning

Small UX/polish line closing out gaps from v4.1's selective-rebuild work.

Highlights:

- Per-file upload result banners (saved/duplicate/error, one banner per file instead of
  one aggregate message).
- **Delete button** on the Documents page — select documents, confirm, remove from the
  index without a CLI round-trip.
- Tooltips on every interactive element across Manage/Documents/Stats pages.
- Single-source versioning: `_VERSION` reads from `importlib.metadata` at runtime
  instead of being hand-maintained in multiple files; UI sidebar pulls it from
  `/ui/status`.
- Stale rebuild-lock detection via manifest/lock timestamp comparison (see known
  limitation: a `delete`/`clean-orphans` run between an interrupted rebuild and the next
  status poll can produce a false "stale" verdict — tracked as a bug, not a roadmap
  item).

See `CHANGELOG.md` § v5.1.0 for the full diff.

## 7. Hardware acceleration paths

Independently of the version plan, three hardware acceleration paths
are supported across all v1.x and beyond:

- **CUDA (NVIDIA)** — primary GPU path. Code uses
  `torch.cuda.is_available()`; configuration uses `device: cuda` or
  `device: auto` in `config/config.yaml`. Verified end-to-end on
  GTX 1650 + WSL2 (see `docs/install.md` § "GPU install (NVIDIA /
  CUDA)"). The typical development workflow uses this path.
- **ROCm (AMD)** — supported through the PyTorch ROCm wheel for
  contributors who have a ROCm-compatible AMD GPU. Code path is
  identical to CUDA (PyTorch's HIP translation is transparent);
  FAISS stays on CPU because `faiss-gpu` is NVIDIA-only. See
  `docs/install.md` § "GPU install (AMD / ROCm)" for prerequisites.
- **CPU-only** — works always. Slower, predictable, no admin work.
  This is the assumed mode for production server deployments; large
  ingest jobs run on a schedule (see
  `docs/deploy.md` § "Scheduled ingest") so the long wall-clock time
  is absorbed off-hours.

The Docling/Unstructured pipeline introduced in v2.0+ inherits this
matrix automatically because both libraries route their inference
through PyTorch.

## 8. Out of scope for v1.x

These items are deliberately deferred and tracked separately. They are
not blockers for any v1.x.y tag.

- Distributed search across multiple FAISS shards.
- Any form of authentication. The `DOC_RAG_API_KEY` single-key model was
  removed (post-v5.1, unreleased at time of writing): it only ever
  worked through the legacy inline `/ui` (query-param key) and `/mcp`
  headers — the Svelte UI never sent it anywhere, so once the inline
  page was also removed the mechanism had no working UI path left and
  was pulled rather than fixed. Multi-user authn/SSO remains unplanned.
- Background re-embedding when the embedding model changes mid-flight.
- A "RAG quality" evaluation harness (retrieval@k, faithfulness scores
  with golden answers).
- A first-party hosted variant.

Multi-collection / namespaces was previously here and is now scoped
into v3.0 instead.

## 9. How to update this roadmap

This file is part of the public surface in spirit, not in SemVer. When
priorities shift:

1. Edit this file in the same PR as the work that proves the shift.
2. Move items between sprints rather than deleting them.
3. Acceptance gates may be tightened but not loosened without an
   explicit note in `CHANGELOG.md`.
