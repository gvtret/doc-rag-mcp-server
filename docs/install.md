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
- Node ≥ 20 — required to build the Svelte Web UI (`ui/`). There is no fallback; without it `/ui` and `/ui-next/` don't serve anything. `bootstrap.sh` / `install_server_native.sh` fail loudly if Node/npm isn't on PATH.

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
- `unstructured` — PDF quality cascade fallback (default: N). When enabled, if Docling scores below `parsing.quality_fallback_threshold` (0.5), automatically retries with Unstructured.

`server` (FastAPI + Uvicorn + python-multipart) is always installed —
nothing else makes sense without the HTTP layer.

Non-interactive mode (CI / scripted deploy):

```bash
DOC_RAG_BOOTSTRAP_NONINTERACTIVE=1 \
DOC_RAG_BOOTSTRAP_FAISS=Y \
DOC_RAG_BOOTSTRAP_EMBEDDINGS=Y \
DOC_RAG_BOOTSTRAP_DEV=N \
DOC_RAG_BOOTSTRAP_METRICS=N \
DOC_RAG_BOOTSTRAP_UNSTRUCTURED=N \
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
Add `--extra unstructured` for PDF quality cascade fallback.

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
  device: "cpu"                          # "cpu" | "cuda" | "auto"
  batch_size: 32
  normalize: true
```

For low-memory machines: `bge-small-en-v1.5` (~120 MB). See
[GPU install](#gpu-install-nvidia--cuda) below for `device: cuda`.

### Offline model delivery (fetch_model.py)

By default sentence-transformers downloads the model from the HF Hub on
first use. For offline / air-gapped hosts, fetch the snapshot once on a
connected machine and ship it with the repo:

```bash
uv run python scripts/fetch_model.py
# optional overrides:
uv run python scripts/fetch_model.py --model BAAI/bge-m3 --dest models/bge-m3 --force
```

The script downloads the HF snapshot into `<dest>.staging`, validates the
sentence-transformers marker files, writes the weights fingerprint
(`WEIGHTS.sha256`), then atomically swaps the directory into place
(`models/bge-m3`). An existing copy is moved to `<dest>.old` first, so a
failed download never leaves a half-written model in place. Exit code is
non-zero on any download or validation failure and the staging directory
is always removed.

The resolved local directory is `models/<basename(model_name)>` next to
`config/config.yaml`; point `embeddings.model_path` at it explicitly to
manage the model manually (this also disables the background HF Hub
update check, since no Hub repo is implied).

`update_check_interval_hours` (default `72`, `0` disables) controls the
read-only background comparison of the local weights against the HF Hub
— no download, no swap; the result surfaces as
`model_update_available` in `/ui/status` and the Web UI banner.

### Chunking

```yaml
chunking:
  strategy: "fixed"   # "fixed" | "recursive"
  target_tokens: 512
  overlap_tokens: 64
```

`recursive` walks the typed block structure (headings, tables) instead of splitting on
raw token counts — see `docs/roadmap.md` § v2.6.

### Vector index & retrieval mode

```yaml
index:
  backend: "faiss"   # "faiss" | "qdrant" | "pgvector"
  metric: "ip"
  top_k: 6

mcp:
  retrieval_mode: "semantic"   # "semantic" | "hybrid"
```

`faiss` (the default) is a local `IndexFlatIP`/`IndexFlatL2` file — no server to run.
`qdrant`/`pgvector` require a running instance; migrate an existing FAISS index with
`doc-rag migrate-vectors --to qdrant` (see [cli.md](cli.md)). `retrieval_mode: hybrid`
runs lexical (TF-IDF) and semantic search in parallel and merges via Reciprocal Rank
Fusion — supported on all three backends.

### Quality gate

```yaml
quality:
  fail_on_severity: "never"   # "never" | "warn" | "error"
```

Controls whether documents with quality warnings (broken tables, low text density,
duplicate headers, ...) block ingest. `never` (default) only records the warnings;
`warn`/`error` raise the bar for what's allowed into the index.

### PDF quality cascade

```yaml
parsing:
  quality_fallback_threshold: 0.5  # 0 = disabled
```

When Docling scores below this threshold, automatically retries with Unstructured (if installed) and keeps the better result.

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
| `DOC_RAG_RATE_LIMIT_RPS` / `DOC_RAG_RATE_LIMIT_BURST` | Per-client token-bucket limiter (default off) |
| `DOC_RAG_HTTP_LOG` | Path to write HTTP request log |
| `DOC_RAG_LOG_LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR` (default `INFO`) |
| `DOC_RAG_LOG_FORMAT` | `text` (default) or `json` for structured logging |
| `DOC_RAG_MAX_CONCURRENCY` | Max concurrent MCP tool calls (default `4`) |
| `DOC_RAG_TOOL_TIMEOUT_SEC` | Per-call MCP tool timeout in seconds (default `30`) |
| `DOC_RAG_SSE_KEEPALIVE_SEC` | Keepalive interval for the `GET /mcp` SSE stream (default `25`) |
| `DOC_RAG_SHUTDOWN_TIMEOUT` | Graceful-shutdown drain time for in-flight requests (default `30`, see `scripts/run_mcp_http.sh`) |
| `DOC_RAG_UI_MAX_UPLOAD_MB` | Max size per uploaded file, MiB (default `100`) |
| `DOC_RAG_UI_MAX_UPLOAD_FILES` | Max files per `/ui/upload` request (default `48`) |
| `DOC_RAG_EXPOSE_SOURCE_PATHS` | If set truthy, `doc_search` results include the on-disk source path (default off) |
| `DOC_RAG_ENV_FILE` | Override the UI-managed env file path (default `<root>/.env`, see [ui.md](ui.md#service-env-editor)) |
| `DOC_RAG_UI_RESTART_ENABLED` / `DOC_RAG_UI_RESTART_CMD` | Enable the UI's "Restart service" button and the shell command it runs |
| `DOC_RAG_PYTHON` | Python interpreter to run under (default: `.venv/bin/python` if present, else `python3`) |

---

## GPU install (NVIDIA / CUDA)

GPU only accelerates the **embedding** step — the FAISS index itself is always
`faiss-cpu` (`pyproject.toml` never ships `faiss-gpu`), so this is about making
`rebuild`/`ingest` faster on a machine that has an NVIDIA GPU, not about the
production server (which is typically CPU-only, see [deploy.md](deploy.md)).

```bash
# after a normal install (venv already has CPU torch):
bash scripts/install_torch_gpu.sh    # installs a CUDA build of torch (cu124 index)
```

Then set in `config/config.yaml`:

```yaml
embeddings:
  device: "cuda"   # or "auto" to fall back to cpu when no GPU is present
```

Verified end-to-end on a GTX 1650 (WSL2). If you process on a GPU box that isn't the
production server, use the offload workflow instead of pointing `device: cuda` at a
config the server also uses: `doc-rag rebuild --skip-index` (parse+chunk only) on the
server, copy `build/chunks_jsonl/` to the GPU machine, embed there, then
`doc-rag rebuild-index` back on the server — see [cli.md](cli.md).

## GPU install (AMD / ROCm)

Same code path as CUDA — PyTorch's HIP translation is transparent, so `device: cuda`
in config still selects the GPU (there's no separate `rocm` device string). Install a
ROCm build of torch instead of the CUDA one:

```bash
"${VENV_PY:-.venv/bin/python}" -m pip install --no-cache-dir torch \
  --index-url https://download.pytorch.org/whl/rocm6.2   # match your ROCm version
```

There's no bundled `install_torch_rocm.sh` script (only CUDA is scripted, see
`scripts/install_torch_gpu.sh`) — run the `pip install` above manually with the wheel
index matching your installed ROCm version. FAISS still stays on CPU (`faiss-gpu` is
NVIDIA-only), same as the CUDA path above.

---

## Performance notes

- CPU-only build is the supported path. Full corpus rebuild on Docling is the heavier cost; plan large ingests off-hours.
- `bge-large-en-v1.5` ≈ 1.5 GB RAM during encoding.
- For low-RAM use `bge-small-en-v1.5`.
- FAISS index is `IndexFlatIP` (exact, no training step).

See [troubleshooting.md](troubleshooting.md) for common install issues.
