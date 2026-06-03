# doc-rag — local RAG over your documents

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
- ~2 GB RAM for embeddings (CPU); GPU optional
- Tesseract is optional (only needed for scanned PDFs)

## Quickstart

```bash
git clone <repo> doc-rag
cd doc-rag
bash scripts/bootstrap.sh        # creates .venv, installs deps (interactive)
cp YOUR_FILES.pdf sources/incoming/
.venv/bin/doc-rag ingest
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

See [CHANGELOG.md](CHANGELOG.md) for a list of notable changes.

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
└── docs/              # see above
```

## Philosophy

Offline-first · reproducible · vendor-independent · long-term maintainable.
Designed for standards, specs, manuals, research docs.

## License

MIT — see [pyproject.toml](pyproject.toml).
