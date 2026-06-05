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

Creates `.venv`, installs base deps (Docling included — it is the only
PDF backend since v2.0), then prompts (with sensible defaults) for:

- FAISS (`faiss-cpu`) — needed for semantic search
- Server extras (`fastapi`, `uvicorn`, `python-multipart`)
- torch + sentence-transformers (CPU)
- Initial `doc-rag ingest`

Non-interactive mode (for CI / scripts):

```bash
DOC_RAG_BOOTSTRAP_NONINTERACTIVE=1 \
DOC_RAG_BOOTSTRAP_FAISS=Y \
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
pip install -e ".[server,faiss]"
bash scripts/install_torch_cpu.sh        # or install_torch_gpu.sh
pip install sentence-transformers
```

Torch is installed via a separate script on purpose — that lets you pick a CPU- or
CUDA-specific wheel without polluting `requirements.txt`.

---

## GPU install (NVIDIA / CUDA)

`doc-rag` runs fine on CPU; GPU is an optional accelerator for the
embedding step. Measured speedup on a development workstation
(NVIDIA GTX 1650, 4 GB VRAM, WSL2): roughly **3×** for batched encoding
with `bge-small-en-v1.5`. Bigger embeddings models scale better with
GPU; for the default `bge-large-en-v1.5` the speedup is more dramatic
because CPU becomes the bottleneck quickly.

### Requirements

- NVIDIA GPU with at least **4 GB VRAM** for `bge-small-en-v1.5`,
  **6 GB+** for `bge-large-en-v1.5`. With less VRAM, drop the embedding
  `batch_size` in `config/config.yaml` (try 8 or even 4) or switch to
  `bge-small`.
- NVIDIA driver matching CUDA 12.4 or newer (`nvidia-smi` reports the
  driver version; CUDA itself is shipped inside the PyTorch wheel,
  you do not need a system-wide CUDA toolkit).
- WSL2 is supported transparently — the PyTorch CUDA wheel sees the
  Windows-side GPU through the Linux subsystem.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[server,faiss]"
bash scripts/install_torch_gpu.sh        # pulls torch from the CUDA 12.4 wheel index
pip install sentence-transformers
```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce GTX 1650
```

### Tell doc-rag to use the GPU

`config/config.yaml`:

```yaml
embeddings:
  model_name: "BAAI/bge-large-en-v1.5"
  device: "cuda"            # or "auto" — picks cuda if torch.cuda.is_available()
  batch_size: 16            # for 4 GB VRAM cards; raise to 32 on 8 GB+
  normalize: true
```

`auto` is the friendliest default: deploys without a GPU fall back to
CPU silently, deploys with a GPU pick it up without a config change.

### Verify end-to-end

Run the bundled benchmark to confirm the encoder is actually on the GPU:

```bash
python scripts/bench.py --size 1000 --device cuda --model BAAI/bge-large-en-v1.5 --batch-size 16
```

The JSON output's `platform.device` field should read `cuda`, and
`encode_chunks_per_s` should be at least 5-10× higher than on the same
host with `--device cpu`.

---

## GPU install (AMD / ROCm)

`doc-rag` runs identical code on AMD ROCm because PyTorch's HIP layer
translates CUDA calls transparently — `torch.cuda.is_available()`
returns `True` against a ROCm-built `torch`, and `device="cuda"` in
`config/config.yaml` works without changes.

### Requirements

- **Linux only.** ROCm is not supported on WSL2 (CUDA is — this is a
  CUDA/WSL-specific path that has no ROCm equivalent at time of
  writing). If your development host is Windows + WSL2, run ROCm on a
  separate Linux host or VM with PCI passthrough.
- **A ROCm-supported AMD GPU.** Recent releases support RDNA2
  (RX 6000-series), RDNA3 (RX 7000-series), CDNA (Instinct MI200/
  MI300), and modern Vega. Older GCN cards (RX 500-series and older)
  have been dropped from upstream support.
- **Device nodes accessible:** `/dev/kfd` and `/dev/dri/renderD*`
  must exist and be readable by the user running `doc-rag`. In a
  VM this means the hypervisor must pass the GPU through (vfio).
  Inside Docker, the container needs `--device=/dev/kfd
  --device=/dev/dri --group-add video`.
- **ROCm runtime installed system-wide.** AMD publishes apt/dnf
  repositories — use `amdgpu-install --usecase=rocm` on
  Debian/Ubuntu; do not rely on pip alone.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[server,faiss]"

# Pick the ROCm version matching your installed runtime.
# Example for ROCm 6.2:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install sentence-transformers
```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True   <your AMD GPU name>
```

### Tell doc-rag to use the GPU

The same config block as for CUDA — `device: cuda` is correct even
on ROCm; PyTorch maps it to the AMD device internally.

```yaml
embeddings:
  device: "cuda"      # or "auto"
  batch_size: 16
```

### Known limits compared to CUDA

- **`faiss-gpu` is NVIDIA-only.** FAISS stays on `faiss-cpu`. This is
  fine for the project's typical sizes (< 50 K chunks); the
  embedding step is the bottleneck regardless.
- **ONNX Runtime ROCm execution provider is less mature** than CUDA.
  When the v1.5+ Docling backend lands (see `docs/roadmap.md`), some
  of its auxiliary ML models may still run on CPU under ROCm. The
  primary PyTorch path will be GPU-accelerated either way.

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
| Legacy `.doc` | `antiword` (preferred) or `catdoc` | required for binary Word format |
| Scanned PDFs (OCR) | none | Docling runs RapidOCR internally — no separate Tesseract install |
| PDF tables / structure | none | Docling extracts grids, headings, formulas by default |

`scripts/install_server_native.sh` installs `antiword`. For Docker, see `docker/Dockerfile`.

### Docling models on first parse

Docling downloads ~300 MB of ML model weights (TableFormer, DocLayout
detection, RapidOCR) on the first PDF parse. The download is cached
under `~/.cache/docling/` (or `$HOME/.cache/...` for the service user)
and survives restarts. Subsequent parses are offline.

If your install is air-gapped, copy the populated cache directory from
a connected box; the path is the same on all Linux hosts.

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
