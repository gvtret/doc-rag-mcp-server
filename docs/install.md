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
- ~2.5 GB disk for `models/bge-m3` (fetched by `bootstrap.sh` when the
  embeddings extra is selected)

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
materialise the venv into `.venv/`. When `config/config.yaml` selects
`embeddings.provider: ollama` (the production default), it also verifies
the Ollama server is reachable and has the configured model, and skips
the local-model fetch.

Extras the bootstrap exposes:

- `faiss` — `faiss-cpu` (semantic search index; default: Y, needed in
  every mode)
- `embeddings` — `torch` + `sentence-transformers` (CPU). Only required
  for the legacy local-weights branch; with `provider: ollama` the
  default answer is N (the ollama provider talks HTTP, no torch needed)
- `dev` — `pytest`, `pytest-cov`, `httpx`, `ruff` (default: N)
- `metrics` — `prometheus-client` (default: N)
- `unstructured` — PDF quality cascade fallback (default: N). When enabled, if Docling scores below `parsing.quality_fallback_threshold` (0.5), automatically retries with Unstructured.

`server` (FastAPI + Uvicorn + python-multipart) is always installed —
nothing else makes sense without the HTTP layer.

Ollama requirements (checked by bootstrap, non-fatal warnings):

- the Ollama server is running (`systemctl status ollama` / `ollama serve`)
  and reachable at `embeddings.api_base`
- the configured model is pulled: `ollama pull bge-m3`

When the `embeddings` extra is selected for the legacy branch and
`models/bge-m3` is absent, `bootstrap.sh` runs `scripts/fetch_model.py`
(~2.2 GB). Set `DOC_RAG_BOOTSTRAP_FETCH_MODEL=Y/N` to override the choice;
a failed download only warns, so air-gapped installs still complete
(semantic search stays off until the model is provided).

Non-interactive mode (CI / scripted deploy):

```bash
DOC_RAG_BOOTSTRAP_NONINTERACTIVE=1 \
DOC_RAG_BOOTSTRAP_FAISS=Y \
DOC_RAG_BOOTSTRAP_EMBEDDINGS=Y \
DOC_RAG_BOOTSTRAP_DEV=N \
DOC_RAG_BOOTSTRAP_METRICS=N \
DOC_RAG_BOOTSTRAP_UNSTRUCTURED=N \
DOC_RAG_BOOTSTRAP_FETCH_MODEL=N \
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
# Ollama on the host with the model pulled (the embeddings provider):
ollama pull bge-m3
# Ollama must listen on more than container-localhost for the container to
# reach it: set OLLAMA_HOST=0.0.0.0 in the ollama service environment.
docker compose up -d --build
curl -sS http://127.0.0.1:3333/health
```

The production default (`embeddings.provider: ollama` in `config/config.yaml`)
needs no local model in the image: `docker-compose.yml` builds with
`REQUIRE_MODEL=0` and the container reaches the host Ollama via
`host.docker.internal` (`extra_hosts: host-gateway` is already in the compose
file — `api_base http://127.0.0.1:11434` inside the container resolves to the
host's loopback through the gateway route). For a remote Ollama server, point
`embeddings.api_base` at `http://<host>:11434` instead. When Ollama is
unreachable the container still starts and serves lexical results
(`/health/ready` reports `model_ready: false`).

Legacy local-weights mode: set `REQUIRE_MODEL: "1"` in `docker-compose.yml`
and fetch `models/bge-m3` first (`uv run python scripts/fetch_model.py`,
~2.2 GB) — the build fails without it, and the `embeddings` extra is then
installed into the image.

The image builds via uv inside the container (see
`docker/Dockerfile`); the lockfile guarantees the deployed venv matches
whatever CI tested. `REQUIRE_MODEL=0` installs `server` + `faiss` —
faiss is required by the vector index in every mode.

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

Prerequisite for semantic search (provider: ollama): a running Ollama
server on the host with the model pulled:

```bash
systemctl enable --now ollama   # or install from https://ollama.com/download
ollama pull bge-m3              # name must match embeddings.model_name
```

The installer prints these reminders when `config/config.yaml` selects
`provider: ollama`; bootstrap also verifies reachability and warns (non-fatal)
when Ollama is down — the service then serves lexical results until Ollama
comes back, no restart needed.

The installer:

- creates system user `docrag`
- installs apt packages: `python3`, `python3-venv`, `python3-dev`,
  `build-essential`, `ca-certificates`, `git`, `rsync`, `antiword`
- syncs the repo into `INSTALL_ROOT` (preserves existing `build/`, `sources/`
  and `models/` - manifest, FAISS index and a delivered model survive upgrades)
- copies `models/bge-m3` from the source checkout when present (legacy
  local-weights branch only)
- bootstraps the venv via `bash scripts/bootstrap.sh` non-interactively (which
  installs uv, runs `uv sync --frozen`, verifies Ollama when the config
  selects that provider, and fetches `models/bge-m3` when the local
  `embeddings` extra is selected and no local copy exists)
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

Two provider branches, selected by `embeddings.provider`:

**Ollama (current production default):**

```yaml
embeddings:
  provider: "ollama"
  api_base: "http://127.0.0.1:11434"
  model_name: "bge-m3"        # as known to Ollama (`ollama pull bge-m3`)
  max_seq_length: 576
  request_timeout_seconds: 120      # bulk index builds
  query_timeout_seconds: 25         # interactive queries; keep < MCP budget (30 s)
  breaker_threshold: 2              # consecutive timeouts that open the breaker
  breaker_cooldown_seconds: 60      # breaker open window -> lexical results
```

A timeout is never retried. After `breaker_threshold` consecutive
timeouts the breaker opens for `breaker_cooldown_seconds`: queries
return lexical results immediately (with a degraded notice) and
`/health` reports `embeddings_degraded: true`, instead of piling up
retries until the MCP layer returns 504.

Requires a running Ollama server with the model pulled. The loader
probes `GET /api/tags` (TTL-cached 30 s): when the server is down the
service reports `ready: false` and `doc_search` degrades to lexical
results instead of hanging. The index fingerprint is derived from the
Ollama manifest digest (`<OLLAMA_MODELS>/manifests/...`, fallbacks
`~/.ollama/models` and `/usr/share/ollama/.ollama/models`), so the
BUG-006 rebuild gate still fires when the model is re-pulled with
different weights. With `provider: ollama` the fingerprint never falls
back to a local-dir hash — a missing manifest yields `None` (fail-closed).

**Local sentence-transformers (legacy, provider unset):**

```yaml
embeddings:
  model_name: "BAAI/bge-m3"   # shipped default: multilingual, 1024-dim
  device: "cpu"               # "cpu" | "cuda" | "auto"
  batch_size: 32
  max_seq_length: 576         # chunk cap 504 + section-prefix budget; bge-m3 defaults to 8192
  normalize: true
  update_check_interval_hours: 72   # 0 disables the HF Hub weight check
  # model_path: "/path/to/models/bge-m3"   # explicit snapshot; disables the check
```

For low-memory machines: `bge-small-en-v1.5` (~120 MB). See
[GPU install](#gpu-install-nvidia--cuda) below for `device: cuda`.

### Offline model delivery (fetch_model.py)

The server never downloads the model at runtime: `model_loader` loads it
with `local_files_only=True` from a local directory (resolution order
below), falling back to an already populated Hugging Face cache by
`embeddings.model_name` only when no local directory resolves. A
`models/bge-m3` snapshot therefore has to exist on every host that runs
semantic search - fetch it once on a connected machine and ship it with
the repository:

```bash
uv run python scripts/fetch_model.py
# optional overrides:
uv run python scripts/fetch_model.py --model BAAI/bge-m3 --dest models/bge-m3 --force
```

The script downloads the HF snapshot into `<dest>.staging`, validates the
sentence-transformers marker files, writes the weights fingerprint
(`WEIGHTS.sha256`), then atomically swaps the staging directory into
place at `models/bge-m3`. Only after a successful download does the
previous copy get set aside (and removed once the swap succeeds), so a
failed download never touches the working model; on a swap failure the
old copy is restored. Exit code is non-zero on any download or
validation failure and the staging directory is always removed.
`--force` only clears a leftover stuck `.staging` directory before the
download - it never touches the working copy.

Where the server looks for the model (in order,
`doc_rag.raglib.model_paths.resolve_model_path`):

1. `embeddings.model_path` from config (if set and valid);
2. `<root>/build/models/bge-m3`;
3. `<root>/models/bge-m3`.

The directory counts as a model only when `config.json` and
`modules.json` are present inside. `bge-m3` is the shipped layout and
what `fetch_model.py` produces by default, regardless of
`embeddings.model_name`; to run a different model, point
`embeddings.model_path` at its snapshot explicitly (this also disables
the background HF Hub update check, since no Hub repo is implied).

The model is loaded at startup (FastAPI lifespan, `local_files_only=True`):
the log shows `Модель эмбеддингов загружена при старте (source=..., path=...)`
or an error. A failed load does not stop the service - `doc_search` returns
lexical results with a notice and `GET /health/ready` returns 503 with reason
`model_missing` in semantic/hybrid mode.

`update_check_interval_hours` (default `72`, `0` disables) controls the
read-only background comparison of the local weights against the HF Hub
- no download, no swap; the result surfaces as
`model_update_available` in `/ui/status` and the Web UI banner.
The first check runs right after startup, then every interval. Only Hub files
with published LFS hashes are compared, so a repo without LFS weight files
never reports an update. The check is also disabled when
`embeddings.model_path` is set, when no model directory resolves, or when
`huggingface_hub` is not installed.

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
- `bge-large-en-v1.5` ≈ 1.5 GB RAM during encoding (the shipped default is
  `bge-m3`; these numbers are for `bge-large-en-v1.5`).
- For low-RAM use `bge-small-en-v1.5`.
- FAISS index is `IndexFlatIP` (exact, no training step).

See [troubleshooting.md](troubleshooting.md) for common install issues.

### Retrieval-quality eval harness

`scripts/eval_retrieval.py` measures recall@10, MRR@10 and nDCG@10 of
the retrieval against a golden query set (`eval/queries_golden.yaml`,
with per-query relevant doc/chunk ids). Use it to turn any
retrieval-quality change into before/after numbers instead of guesswork:

```bash
# Baseline (run BEFORE the change, keep the report):
python scripts/eval_retrieval.py --root . --label baseline

# After the change:
python scripts/eval_retrieval.py --root . --label after
```

Each run prints a human-readable table and writes a JSON report (per
query and aggregate) under `build/eval/`:
`<label>-<date>-<modes>.json` with `--label`, timestamped otherwise.
Modes, depth and query file are configurable (`--modes semantic,hybrid,lexical`,
`--top-k`, `--queries`). Compare two reports by diffing the aggregate
metrics; reports are kept locally under `build/eval/` (gitignored), so
a baseline must be re-taken or shared manually -- a regression is
provable only against a preserved baseline file.
