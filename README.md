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
- ~2 GB RAM for embeddings; ~1 GB extra for Docling models on first parse
- `antiword` is optional (only needed for legacy `.doc` files)
- OCR for scanned PDFs is built into Docling (RapidOCR); no separate Tesseract install required
- Node ≥ 20 (optional, v2.2+ — only needed to build the new Svelte `/ui-next/` page; the legacy inline `/ui` works without Node)

## Quickstart

```bash
# uv is the official installer since v2.1; the bootstrap script installs it if missing.
git clone https://github.com/gvtret/doc-rag-mcp-server
cd doc-rag
bash scripts/bootstrap.sh        # installs uv, runs `uv sync --frozen`, creates .venv
cp YOUR_FILES.pdf sources/incoming/
uv run doc-rag ingest
bash scripts/run_mcp_http.sh     # MCP/UI on http://127.0.0.1:3333
```

Then open `http://127.0.0.1:3333/ui` or point Cursor at `http://127.0.0.1:3333/mcp`.

## Documentation

| Guide | What's inside |
| --- | --- |
| [docs/install.md](docs/install.md) | venv + torch (CPU/GPU), system packages, OCR, config reference |
| [docs/cli.md](docs/cli.md) | `doc-rag ingest / rebuild / delete / wipe / clean-orphans / clear-incoming` |
| [docs/mcp.md](docs/mcp.md) | Cursor / Claude integration, Streamable HTTP + SSE, auth, rate limit |
| [docs/ui.md](docs/ui.md) | Web UI: upload, dedup, ingest, delete, danger zone, degraded-mode banner |
| [docs/deploy.md](docs/deploy.md) | Docker Compose, native systemd, deploy archive, remote MCP |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Torch/CUDA, MCP not visible, FAISS rebuild, PEP 668, OCR |
| [docs/roadmap.md](docs/roadmap.md) | Versioning policy and the path to public v1.x.y |

See [CHANGELOG.md](CHANGELOG.md) for notable changes between releases.

## Project layout

```
doc-rag/
├── sources/
│   ├── incoming/      # drop new documents here
│   └── archived/      # processed files (moved automatically after ingest)
├── build/             # generated: docs_md/, chunks_jsonl/, embeddings/, index/, manifest.json
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
