# doc-rag — local RAG over your documents

[![tests](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/tests.yml/badge.svg)](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/tests.yml)
[![lint](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/lint.yml/badge.svg)](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/lint.yml)
[![build](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/build.yml/badge.svg)](https://github.com/gvtret/doc-rag-mcp-server/actions/workflows/build.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

`doc-rag` is a local, offline-first knowledge base for engineering documentation:

```
PDF / DOCX / DOC / MD / TXT  →  Markdown  →  Chunks  →  FAISS  →  MCP / Cursor / Web UI
```

- **Offline after first model download.** No data leaves your machine.
- **Cursor / Claude MCP integration** via Streamable HTTP on a single endpoint.
- **Web UI** for upload, ingest, delete, and live status — no terminal required.
- **Graceful degradation**: if FAISS isn't ready, lexical search keeps working and the user is told.

## Requirements

- Linux or WSL2 (Python ≥ 3.10)
- **Ollama server with `bge-m3` pulled** (the production embeddings provider;
  weights live there, not on this host). Unreachable Ollama degrades to
  lexical search — no crash, no hang
- ~2 GB RAM only for the legacy local-weights branch (torch +
  sentence-transformers); the ollama provider needs none of it
- ~1 GB extra for Docling models on first parse
- `antiword` is optional (only needed for legacy `.doc` files)
- OCR for scanned PDFs is built into Docling (RapidOCR); no separate Tesseract install required
- Node >= 20 (required to build the Svelte UI in `ui/`; there is no inline
  fallback - `bootstrap.sh`, `install_server_native.sh` and
  `docker/Dockerfile` fail without it)
- ~2.5 GB disk for `models/bge-m3` — **legacy branch only** (fetched by
  `bootstrap.sh` when the local `embeddings` extra is selected and
  `provider: ollama` is not set)

## Quickstart

```bash
# 1. Ollama with the embedding model (the production embeddings provider):
#    https://ollama.com/download, then:
ollama pull bge-m3                # name must match embeddings.model_name

# 2. Python side: uv is the official installer since v2.1; the bootstrap
#    script installs it if missing.
git clone https://github.com/gvtret/doc-rag-mcp-server
cd doc-rag
bash scripts/bootstrap.sh        # installs uv, `uv sync --frozen`, creates .venv,
                                 # verifies Ollama reachability and the model
cp YOUR_FILES.pdf sources/incoming/
uv run doc-rag ingest
bash scripts/run_mcp_http.sh     # MCP/UI on http://127.0.0.1:3333
```

Then open `http://127.0.0.1:3333/ui` or point Cursor at `http://127.0.0.1:3333/mcp`.

> Legacy local-weights mode (no Ollama, model in `models/bge-m3`): set
> `embeddings.provider` unset in `config/config.yaml` and install the
> `embeddings` extra — bootstrap asks for it and fetches the model.
> See [docs/install.md](docs/install.md#embeddings).

## Documentation

| Guide | What's inside |
| --- | --- |
| [docs/install.md](docs/install.md) | Ollama + venv setup, torch (CPU/GPU) for the legacy branch, system packages, config reference |
| [docs/cli.md](docs/cli.md) | `doc-rag ingest / rebuild / delete / wipe / clean-orphans / clear-incoming` |
| [docs/mcp.md](docs/mcp.md) | Cursor / Claude integration, Streamable HTTP + SSE, auth, rate limit |
| [docs/ui.md](docs/ui.md) | Web UI: upload, dedup, ingest, delete, danger zone, degraded-mode banner |
| [docs/deploy.md](docs/deploy.md) | Docker Compose, native systemd, deploy archive, remote MCP |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Ollama unreachable, Torch/CUDA, MCP not visible, FAISS rebuild, PEP 668, OCR |
| [docs/roadmap.md](docs/roadmap.md) | Versioning policy, SemVer-protected public surface, release history |
| [docs/bench-results.md](docs/bench-results.md) | `scripts/bench.py` output: parse/embed/rebuild timings |
| [docs/HANDOFF.md](docs/HANDOFF.md) | Living session handoff log (see the `handoff` workflow) |

See [CHANGELOG.md](CHANGELOG.md) for notable changes between releases.

## Project layout

```
doc-rag/
├── sources/
│   ├── incoming/      # drop new documents here
│   └── archived/      # processed files (moved automatically after ingest)
├── build/             # generated: docs_md/, chunks_jsonl/, embeddings/, index/, manifest.json
├── models/            # offline embedding model (bge-m3, gitignored; fetched on install)
├── config/config.yaml # main config
├── src/doc_rag/       # Python package
├── scripts/           # bootstrap, run_mcp_http, install_server_native, ...
├── docker/            # Dockerfile
├── systemd/           # service unit template
├── .github/workflows/ # CI (tests, lint, Docker build)
└── docs/              # see the table above
```

## Philosophy

Offline-first · reproducible · vendor-independent · long-term maintainable.
Designed for standards, specs, manuals, and research documents.

## Talks and articles

- *Russian*, June 2026 — [«Как я научил оракула читать ГОСТы: история doc-rag, рассказанная по-старорусски»](https://habr.com/ru/articles/1043346/) on Habr. A pet-project narrative covering the same architecture that ships in this repository.

## License

`doc-rag` is licensed under the **MIT License** — see [LICENSE](LICENSE).

Third-party dependency licenses are summarised in [NOTICE](NOTICE).
Releases before v2.0.0 were AGPL-3.0-or-later (because the default PDF
backend was PyMuPDF); v2.0 switched to Docling, which is MIT, and
relicensed the project to match.

## Contributing

Issues and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test
instructions, commit conventions, and the SemVer policy for the public
surface ([docs/roadmap.md § 1](docs/roadmap.md)).

To report a security issue, see [SECURITY.md](SECURITY.md). Please do
not open a public issue for vulnerabilities.
