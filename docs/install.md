# Installation

`doc-rag` v2.1+ ships with [uv](https://docs.astral.sh/uv/) as the
official installer. A committed `uv.lock` makes every install
reproducible — `uv sync --frozen` always reconstructs the exact venv
that CI tests against.

Three install paths:

1. [Local dev (bootstrap script)](#local-dev-bootstrap) — recommended for hacking on the code.
2. [Docker Compose](#docker-compose) — single-command server, easiest for a LAN deploy.
3. [Native systemd](#native-systemd-no-docker) — production-style install under `/opt/doc-rag-mcp`.

For a deeper deploy walkthrough see [deploy.md](deploy.md).

---

## Prerequisites

- Linux or WSL2
- Python ≥ 3.10 (uv can install it for you if missing — see below)
- ~2 GB RAM for embeddings; ~1 GB extra cache for Docling models on first parse
- `antiword` (optional, only for legacy `.doc` files)
- OCR for scanned PDFs is built into Docling (RapidOCR); no separate Tesseract install
- Node ≥ 20 (optional, v2.2+ — only needed to build the Svelte `/ui-next/` page; the legacy inline `/ui` works without it)

### Installing uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The installer drops a single binary into `~/.local/bin`. If your shell
session does not pick it up, source the rc snippet the installer
prints, or restart the shell.

---

## Local dev (bootstrap)

```bash
bash scripts/bootstrap.sh
```

`bootstrap.sh` installs uv if missing, then prompts (with sensible
defaults) for which extras to include and runs `uv sync --frozen` to
materialise the venv into `.venv/`.

Extras the bootstrap exposes:

- `faiss` — `faiss-cpu` (semantic search index; default: Y)
- `embeddings` — `torch` + `sentence-transformers` (CPU; default: Y)
- `dev` — `pytest`, `pytest-cov`, `httpx`, `ruff` (default: N)
- `metrics` — `prometheus-client` (default: N)

`server` (FastAPI + Uvicorn + python-multipart) is always installed —
nothing else makes sense without the HTTP layer.

Non-interactive mode (CI / scripted deploy):

```bash
DOC_RAG_BOOTSTRAP_NONINTERACTIVE=1 \
DOC_RAG_BOOTSTRAP_FAISS=Y \
DOC_RAG_BOOTSTRAP_EMBEDDINGS=Y \
DOC_RAG_BOOTSTRAP_DEV=N \
DOC_RAG_BOOTSTRAP_METRICS=N \
DOC_RAG_BOOTSTRAP_INGEST=N \
bash scripts/bootstrap.sh
```

`DOC_RAG_BOOTSTRAP_FROZEN=0` relaxes `--frozen` (useful when iterating
on `pyproject.toml`).

### Manual uv sync (skipping the script)

```bash
uv sync --frozen --extra server --extra faiss --extra embeddings --extra dev
```

Add `--extra metrics` for the Prometheus exporter.

Run anything with the venv via `uv run`:

```bash
uv run pytest
uv run ruff check src/ tests/
uv run doc-rag ingest
```

### Docling models on first parse

Docling downloads ~300 MB of ML model weights (TableFormer, DocLayout,
RapidOCR) on the first PDF parse. The download is cached under
`~/.cache/docling/` and survives restarts. Subsequent parses are
offline.

If your install is air-gapped, copy the populated cache directory from
a connected box; the path is the same on all Linux hosts.

---

## Docker Compose

```bash
cp .env.example .env       # optional: tweak port, origins, API key
docker compose up -d --build
curl -sS http://127.0.0.1:3333/health
```

The image now builds via uv inside the container (see
`docker/Dockerfile`); the lockfile guarantees the deployed venv matches
whatever CI tested.

Volumes mounted by default:

- `./build` (manifest, chunks, FAISS — **persists across restarts**)
- `./config` (read-only)
- `./sources` (incoming/archived)

---

## Native systemd (no Docker)

For a Linux server you control, with auto-start on boot:

```bash
sudo bash scripts/install_server_native.sh
# optional: --minimal       only HTTP/MCP, skip torch/embeddings
# optional: <INSTALL_ROOT>  default: /opt/doc-rag-mcp
```

The installer:

- creates system user `docrag`
- installs apt packages: `python3`, `python3-venv`, `build-essential`, `rsync`, `antiword`
- syncs the repo into `INSTALL_ROOT` (preserves existing `build/` — manifest and FAISS survive upgrades)
- bootstraps the venv via `bash scripts/bootstrap.sh` non-interactively (which installs uv and runs `uv sync --frozen`)
- installs and starts `doc-rag-mcp.service`
- writes `/etc/default/doc-rag` (edit port / origins / API key, then `systemctl restart doc-rag-mcp`)

Run ingest as the service user:

```bash
sudo -u docrag -H bash -lc 'cd /opt/doc-rag-mcp && uv run doc-rag ingest'
```

Logs: `journalctl -u doc-rag-mcp -f`

---

## System packages

Required at runtime depending on what you parse:

| Feature | Package | Notes |
| --- | --- | --- |
| Legacy `.doc` | `antiword` (preferred) or `catdoc` | required for binary Word format |
| Scanned PDFs (OCR) | none | Docling runs RapidOCR internally |
| PDF tables / structure | none | Docling extracts grids, headings, formulas by default |

`scripts/install_server_native.sh` installs `antiword`. For Docker,
see `docker/Dockerfile`.

---

## Configuration

Main config: `config/config.yaml`. Key sections:

### Embeddings

```yaml
embeddings:
  model_name: "BAAI/bge-large-en-v1.5"   # 1024-dim, multilingual works fine
  device: "auto"                          # auto | cpu
  batch_size: 32
  normalize: true
```

For low-memory machines: `bge-small-en-v1.5` (~120 MB).

### Chunking

```yaml
chunking:
  target_tokens: 512
  overlap_tokens: 64
```

### Source archiving

```yaml
sources:
  archive_after_ingest: true   # move processed files to sources/archived/
  incremental_ingest: true     # skip files already in manifest (by sha256)
```

### Environment overrides

| Variable | Purpose |
| --- | --- |
| `DOC_RAG_ROOT` | Project root (when running outside editable install) |
| `DOC_RAG_HTTP_HOST` / `DOC_RAG_HTTP_PORT` | HTTP server bind (default `0.0.0.0:3333`) |
| `DOC_RAG_ALLOWED_ORIGINS` | Comma-separated CORS allow-list |
| `DOC_RAG_API_KEY` | Require `Authorization: Bearer <key>` / `X-Api-Key` on MCP and UI |
| `DOC_RAG_RATE_LIMIT_RPS` / `_BURST` | Per-client token-bucket limiter (default off) |
| `DOC_RAG_HTTP_LOG` | Path to write HTTP request log |

---

## Performance notes

- CPU-only build is the supported path. Full corpus rebuild on Docling is the heavier cost; plan large ingests off-hours.
- `bge-large-en-v1.5` ≈ 1.5 GB RAM during encoding.
- For low-RAM use `bge-small-en-v1.5`.
- FAISS index is `IndexFlatIP` (exact, no training step).

See [troubleshooting.md](troubleshooting.md) for common install issues.
