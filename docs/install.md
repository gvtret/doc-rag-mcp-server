# Installation

Three install paths:

1. [Local dev (bootstrap script)](#local-dev-bootstrap) — recommended for hacking on the code
2. [Docker Compose](#docker-compose) — single-command server, easiest for a LAN deploy
3. [Native systemd](#native-systemd-no-docker) — production-style install under `/opt/doc-rag-mcp`

For a deeper deploy walkthrough see [deploy.md](deploy.md).

---

## Local dev (bootstrap)

```bash
bash scripts/bootstrap.sh
```

Creates `.venv`, installs base deps, then prompts (with sensible defaults) for:

- FAISS (`faiss-cpu`) — needed for semantic search
- PyMuPDF — better PDF parsing + table extraction
- OCR stack (`pytesseract`, `Pillow`, `pymupdf`) — only needed for scanned PDFs
- Server extras (`fastapi`, `uvicorn`, `python-multipart`)
- torch + sentence-transformers (CPU or GPU)
- Initial `doc-rag ingest`

Non-interactive mode (for CI / scripts):

```bash
DOC_RAG_BOOTSTRAP_NONINTERACTIVE=1 \
DOC_RAG_BOOTSTRAP_FAISS=Y \
DOC_RAG_BOOTSTRAP_PDF=Y \
DOC_RAG_BOOTSTRAP_OCR=Y \
DOC_RAG_BOOTSTRAP_SERVER=Y \
DOC_RAG_BOOTSTRAP_TORCH=2 \
bash scripts/bootstrap.sh
```

`DOC_RAG_BOOTSTRAP_TORCH`: `1` = GPU, `2` = CPU, `3` = skip.

### Manual venv install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[server,faiss,pdf]"
bash scripts/install_torch_cpu.sh        # or install_torch_gpu.sh
pip install sentence-transformers
```

Torch is installed via a separate script on purpose — that lets you pick a CPU- or
CUDA-specific wheel without polluting `requirements.txt`.

---

## Docker Compose

```bash
cp .env.example .env       # optional: tweak port, origins, API key
docker compose up -d --build
curl -sS http://127.0.0.1:3333/health
```

Volumes mounted by default:

- `./build` (manifest, chunks, FAISS — **persists across restarts**)
- `./config` (read-only)
- `./sources` (incoming/archived)

---

## Native systemd (no Docker)

For a Linux server you control, with auto-start on boot:

```bash
sudo bash scripts/install_server_native.sh
# optional: --gpu          NVIDIA driver required
# optional: --minimal      MCP only, skip torch/embeddings
# optional: <INSTALL_ROOT> (default: /opt/doc-rag-mcp)
```

The installer:

- creates system user `docrag`
- installs apt packages: `python3-venv`, `build-essential`, `tesseract-ocr*`, `antiword`
- syncs the repo into `INSTALL_ROOT` (preserves existing `build/` — manifest and FAISS survive upgrades)
- runs `scripts/bootstrap.sh` non-interactively
- installs and starts `doc-rag-mcp.service`
- writes `/etc/default/doc-rag` (edit port / origins / API key, then `systemctl restart doc-rag-mcp`)

Run ingest as the service user:

```bash
sudo -u docrag -H bash -lc 'cd /opt/doc-rag-mcp && .venv/bin/doc-rag ingest'
```

Logs: `journalctl -u doc-rag-mcp -f`

---

## System packages

Required at runtime depending on what you parse:

| Feature | Package | Notes |
| --- | --- | --- |
| Scanned PDFs (OCR) | `tesseract-ocr` + `tesseract-ocr-{eng,rus,equ}` | `equ` may be missing on some distros |
| Legacy `.doc` | `antiword` (preferred) or `catdoc` | required for binary Word format |
| PDF tables | none (pure Python via PyMuPDF) | |

`scripts/install_server_native.sh` installs all of these. For Docker, see `docker/Dockerfile`.

---

## Configuration

Main config: `config/config.yaml`. Key sections:

### Embeddings

```yaml
embeddings:
  model_name: "BAAI/bge-large-en-v1.5"   # 1024-dim, multilingual works fine
  device: "auto"                          # auto | cpu | cuda
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

- GPU strongly recommended for >2 GB corpora; FAISS rebuild on 4000+ chunks takes ~3 h on 8 CPU cores.
- `bge-large-en-v1.5` ≈ 1.5 GB VRAM.
- For low VRAM use `bge-small-en-v1.5`.
- FAISS index is `IndexFlatIP` (exact, no training step).

See [troubleshooting.md](troubleshooting.md) if torch/CUDA misbehaves or pip complains about
PEP 668 / `externally-managed-environment`.
