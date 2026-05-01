# doc-rag — Universal Local RAG + Cursor MCP

`doc-rag` is a universal local document knowledge base:

PDF / DOCX → Markdown / JSON → Embeddings → FAISS → Cursor MCP tools.

It is designed for engineering documentation, standards, specs, manuals, etc.

Works fully offline after initial model download.

---

## 1. Requirements

- Linux / WSL2 (recommended)
- Python >= 3.10
- (Optional) NVIDIA GPU + CUDA (for faster embeddings)
- Cursor IDE

---

## 2. Installation

## Quickstart (bootstrap)

The easiest way to get a working setup (Cursor-safe by default):

```bash
bash scripts/bootstrap.sh
```

This will create `.venv`, install base deps, optionally install FAISS/PyMuPDF,
optionally install torch + sentence-transformers, and optionally run `ingest`.

### Verify MCP without Cursor (optional)

Requires the HTTP MCP server (in another terminal):

```bash
bash scripts/run_mcp_http.sh
```

Then:

```bash
bash scripts/verify_mcp.sh
```

You should see JSON-RPC responses including `doc_search` in `tools/list`.

**Making `doc_search` available to Cursor Agent**

- Open this project in Cursor (folder that contains `.cursor/mcp.json`).
- Restart Cursor so it loads the MCP server.
- In Chat, check **Available Tools**: `doc_search` should appear under the doc-rag server. Enable it if it’s toggled off.
- Agent uses the tools listed under Available Tools when relevant; you can ask e.g. “Search my docs for X” or “Use doc_search to find …”.

**If the Agent still doesn’t see `doc_search`** (server shows enabled in “Installed MCP Servers” but the Agent says it has no such tool), use **global** MCP config so Cursor always loads doc-rag:

1. Run: `bash scripts/print_global_mcp_config.sh` and copy the printed JSON. (If the script prints corrupted output—e.g. on WSL/Windows with CRLF—run `python3 scripts/write_global_mcp_config.py` instead, or copy `build/mcp_global_example.json` after running it once.)
2. Open or create `~/.cursor/mcp.json` in your home directory. If it already has other servers, add only the `"doc-rag"` entry under `mcpServers`. Otherwise paste the full example.
3. Restart Cursor and open this project (or any folder). Start a **new** chat and ask the Agent to use `doc_search` again.

### 2.1 Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2.2 Install base dependencies

```bash
pip install -r requirements.txt
pip install -e .

# Optional: FAISS (recommended)
pip install faiss-cpu>=1.8.0

# Optional: Better PDF parsing
pip install pymupdf>=1.24.0
```

### 2.3 Install PyTorch (separate on purpose)

GPU (recommended):

```bash
bash scripts/install_torch_gpu.sh
```

CPU only:

```bash
bash scripts/install_torch_cpu.sh
```

We intentionally keep torch out of requirements.txt
to avoid accidental CUDA dependency issues.

---

## 3. Project Layout

```text
doc-rag/
├── sources/
│   ├── incoming/     # Drop new documents here
│   └── archived/     # Processed documents
├── build/
│   ├── docs_md/       # Normalized markdown
│   ├── tables_json/   # Extracted tables
│   ├── chunks_jsonl/  # RAG chunks
│   ├── embeddings/    # Vector files
│   └── index/         # FAISS index
├── config/
│   └── config.yaml
├── src/doc_rag/        # Python package
├── scripts/
├── .cursor/
└── .venv/
```

---

## 4. Configuration

Main config: `config/config.yaml`

Important sections:

### Embeddings

```yaml
embeddings:
  model_name: "BAAI/bge-large-en-v1.5"
  device: "auto"     # auto | cpu | cuda
  batch_size: 32
  normalize: true
```

### Chunking

```yaml
chunking:
  target_tokens: 512
  overlap_tokens: 64
```

Tune for your document size.

---

## 5. Ingest Pipeline

### 5.1 Add documents

Put files into:

```bash
sources/incoming/
```

Supported:

- PDF
- DOCX

### 5.2 Run ingest

```bash
doc-rag ingest
```

Pipeline:

1. Parse documents
2. Normalize text
3. Split by sections
4. Export Markdown
5. Extract tables
6. Chunk text
7. Build embeddings
8. Build FAISS index
9. Archive sources

---

## 6. Rebuild Index

If you changed config / model:

```bash
doc-rag rebuild
```

Reuses existing markdown.

---

## 7. Cursor MCP Integration (Main Feature)

### 7.1 Why wrapper script

Cursor does NOT activate venv.

So MCP server is started via:

```text
scripts/run_mcp_http.sh → uvicorn … doc_rag.server.mcp_http
```

Uses the project `.venv` when present (`run_mcp.sh` forwards here for compatibility).

---

### 7.2 Enable MCP

Point Cursor at Streamable HTTP, e.g. copy `src/doc_rag/server/mcp_cursor_http.json` into `.cursor/mcp.json` (or merge `mcpServers`), or use `~/.cursor/mcp.json` from `scripts/write_global_mcp_config.py`.

Restart Cursor fully after install.

---

### 7.3 Verify MCP

In terminal (server must be running):

```bash
bash scripts/run_mcp_http.sh
```

In another terminal:

```bash
bash scripts/verify_mcp.sh
```

In Cursor Agent Chat:

```
List MCP tools
```

You should see:

```
doc_search
```

---

## 8. Using in Cursor

Example prompt:

```
Find all references to PushSetup in the documentation.
Then summarize its attributes and methods.
```

Cursor will:

1. Call doc_search
2. Retrieve chunks
3. Use them as context
4. Generate code / explanation

---

## 9. HTTP Server (Optional Debug)

For manual testing:

```bash
pip install -e .[server]
doc-rag serve
```

Check:

```bash
curl http://127.0.0.1:3333/health
```

---

## 10. Testing

```bash
pip install -e .[dev,faiss]
pytest
```

---

## 11. Adding New Documents

Workflow:

```bash
cp new.pdf sources/incoming/
doc-rag ingest
```

Only changed/new files are reprocessed.

---

## 12. Performance Notes

- GPU strongly recommended for large corpora
- bge-large ≈ 1.5GB VRAM
- For low VRAM use: bge-small

---

## 13. Troubleshooting

### Torch / CUDA issues

Reinstall:

```bash
pip uninstall torch -y
bash scripts/install_torch_gpu.sh
```

### MCP not visible / Agent doesn’t see doc_search

- **Restart Cursor** and open the project root (the folder that contains `.cursor/`).
- Check **Settings → MCP**: doc-rag should be listed and toggled on; enable **doc_search** in the tools list if shown.
- In **Chat**, check **Available Tools** and ensure doc_search is enabled for the conversation.
- If the Agent still reports it has no `doc_search` tool (known Cursor issue with workspace MCP in some contexts), add doc-rag to **global** config: run `bash scripts/print_global_mcp_config.sh` (or `python3 scripts/write_global_mcp_config.py` if the shell script misbehaves), then add the printed `doc-rag` entry to `~/.cursor/mcp.json` (see “Making doc_search available to Cursor Agent” above).
- Verify: `.cursor/mcp.json` (or global `~/.cursor/mcp.json`) uses `transport: streamableHttp` and `bash scripts/verify_mcp.sh` (with `run_mcp_http.sh` already running) shows `doc_search` in `tools/list`.

### Empty search

Run:

```bash
doc-rag ingest
```

---

## 14. Philosophy

This project is designed to be:

- Offline-first
- Reproducible
- IDE-integrated
- Vendor-independent
- Long-term maintainable

Suitable for standards, specs, research docs, manuals.

---



### Important: install the package (recommended)

To ensure Cursor and scripts can import `doc_rag`, install the project in editable mode:

```bash
pip install -e .[faiss]
```

If you forget this step, set `PYTHONPATH=./src` when running tools, or set `DOC_RAG_ROOT` to the repo root if the package is installed outside the tree; editable install is the cleanest setup.


## Troubleshooting: PEP 668 (externally-managed-environment)

If you see:

`error: externally-managed-environment`

It means pip is trying to install into the **system Python**. This project should install into `.venv`.

Fix:
- Always run `bash scripts/bootstrap.sh` (preferred), or
- Ensure you are using `.venv/bin/python -m pip ...`

The torch install scripts also force using `.venv/bin/python` to avoid this.

## Remote MCP over HTTP (Docker / another machine)

This project includes an MCP server that speaks the **Streamable HTTP** transport on a single endpoint:

- `POST /mcp` (JSON-RPC request/response, `application/json`)
- `GET  /mcp` keeps an **SSE** stream open (`text/event-stream`) and sends a `doc_rag/ready` notification + keepalives.

### Run on a remote machine (example: 192.168.1.118)

On the remote host:

```bash
git clone <your repo>
cd doc-rag
cp .env.example .env
# edit .env if needed
docker compose up -d --build
```

By default the container exposes:

- `http://<HOST>:3333/mcp`

### Configure Cursor to use the remote server

In Cursor global MCP config (recommended for remote servers), add:

```json
{
  "mcpServers": {
    "doc-rag-remote": {
      "transport": "streamableHttp",
      "url": "http://192.168.1.118:3333/mcp"
    }
  }
}
```

If your client sends an `Origin` header, you may need to allow it:

- set `DOC_RAG_ALLOWED_ORIGINS` in `docker-compose.yml` (comma-separated),
  e.g. `http://localhost,http://127.0.0.1`

### Quick manual check (curl)

```bash
curl -sS -X POST http://192.168.1.118:3333/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
```

Then list tools:

```bash
curl -sS -X POST http://192.168.1.118:3333/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
```


## HTTP MCP server (Streamable HTTP + SSE)

This project can run an MCP server over HTTP on port **3333**.

Endpoints:
- `POST /mcp` — MCP Streamable HTTP (JSON-RPC in, JSON out)
- `GET  /mcp` — **SSE stream** (`text/event-stream`) for server→client notifications (Cursor may use this for streaming/notifications)
- `GET  /health` — health check (`{"status":"ok",...}`)
- `GET  /ui` — simple web UI (upload docs + run ingest)

### Run locally

```bash
bash scripts/bootstrap.sh
bash scripts/run_mcp_http.sh
# Server: http://127.0.0.1:3333/mcp
```

If `config/` and `build/` live outside the installed package tree, set **`DOC_RAG_ROOT`** to the repository root (same idea as for `doc-rag ingest` with a non-default layout).

### SSE quick check

```bash
curl -N -H "Accept: text/event-stream" http://127.0.0.1:3333/mcp
```

You should see keepalive comments and a `doc_rag/ready` notification.

### Request logging

HTTP requests are logged to stderr. To also write to a file:

```bash
export DOC_RAG_HTTP_LOG="$PWD/build/http.log"
bash scripts/run_mcp_http.sh
```

### Auth (optional)

In LAN you can run open-by-default. To protect MCP/UI with an API key:

```bash
export DOC_RAG_API_KEY="change-me"
bash scripts/run_mcp_http.sh
```

Clients must send `Authorization: Bearer <key>` (or `X-Api-Key: <key>`).

### Rate limit (optional)

To limit abuse, you can enable a simple per-client token-bucket limiter:

- `DOC_RAG_RATE_LIMIT_RPS` (default: 0 = disabled)
- `DOC_RAG_RATE_LIMIT_BURST` (default: 5)

### Docker

```bash
docker compose up --build
# Server: http://<host>:3333/mcp
# Health: http://<host>:3333/health
```

### Deploy on a server (VM/LXC in LAN)

Recommended: a VM (Ubuntu/Debian) running Docker.

On the server:

```bash
git clone <your repo> doc-rag
cd doc-rag
cp .env.example .env
# edit .env if needed
docker compose up -d --build
```

Health check:

```bash
curl -sS http://<server-ip>:3333/health
```

Web UI (upload + ingest):

- `http://<server-ip>:3333/ui`

MCP config download from UI:

- `http://<server-ip>:3333/ui/mcp/cursor.json`
- `http://<server-ip>:3333/ui/mcp/vscode.json`

### Deploy via archive (recommended for simple copy)

This repo includes a script that builds a deployable tarball from the current git `HEAD`.
By default it **does not** include large documents from `sources/` (so the archive stays small).

Build archive locally:

```bash
bash scripts/make_deploy_archive.sh
# output: doc-rag-deploy-YYYYMMDD-<sha>.tar.gz
```

If you really want to ship the archived PDFs/DOCX too (archive may be huge):

```bash
bash scripts/make_deploy_archive.sh --with-docs
```

Copy to server and unpack:

```bash
scp doc-rag-deploy-*.tar.gz user@<server-ip>:~
ssh user@<server-ip>
tar xzf doc-rag-deploy-*.tar.gz
cd doc-rag
```

Then choose **one** of:

- **Docker deploy**:

```bash
cp .env.example .env
# edit .env if needed
docker compose up -d --build
curl -sS http://127.0.0.1:3333/health
```

- **Native Linux (no Docker) deploy**:

```bash
sudo bash scripts/install_server_native.sh
curl -sS http://127.0.0.1:3333/health
```

### Native Linux (venv + systemd, no Docker)

On Debian/Ubuntu as **root** (installer copies the repo into `/opt/doc-rag-mcp` by default):

```bash
sudo bash scripts/install_server_native.sh
# optional: sudo bash scripts/install_server_native.sh --gpu /opt/doc-rag-mcp   # NVIDIA driver required on host
# optional: sudo bash scripts/install_server_native.sh --minimal            # MCP only, skip torch/embeddings
```

Creates system user **`docrag`**, installs system packages (`python3-venv`, `build-essential`, …), runs **noninteractive** `scripts/bootstrap.sh`, installs **`doc-rag-mcp.service`** (starts on boot), and writes **`/etc/default/doc-rag`** (edit port, origins, optional API key; then `sudo systemctl restart doc-rag-mcp`).

Ingest:

```bash
sudo -u docrag -H bash -lc 'cd /opt/doc-rag-mcp && .venv/bin/doc-rag ingest'
```

(change `/opt/doc-rag-mcp` if you installed elsewhere).

### Web UI

Open:

- `http://<host>:3333/ui`

If `DOC_RAG_API_KEY` is set, use:

- `http://<host>:3333/ui?key=<DOC_RAG_API_KEY>`

The UI can:
- upload `.pdf` / `.docx` into `sources/incoming`
- run `ingest` (async) and show status


## Source archiving after ingest

By default, `doc-rag ingest` moves processed files from `sources/incoming/` to `sources/archived/` (preserving subfolders) and cleans up empty directories.

To disable this behavior, add to `config/config.yaml`:

```yaml
sources:
  archive_after_ingest: false
```


## Ingest archiving & dedup

After a successful `ingest`/`rebuild`, files from `sources/incoming` are handled as follows:

- If `sources/archived` already contains a file with the same relative name **and the SHA256 matches**, the incoming file is deleted (dedup).
- Otherwise the incoming file is moved into `sources/archived`. If the name already exists with a different hash, a `__<hash10>` suffix is added.

To disable archiving:

```yaml
sources:
  archive_after_ingest: false
```


### Incremental ingest

By default, `doc-rag ingest` is **incremental**:
- `build/manifest.json` is updated by appending new document entries.
- `build/chunks_jsonl/chunks.jsonl` is appended with new chunks.
- Files already present (same `sha256`) are skipped but still archived/deduped.

To force a full overwrite behavior:

```yaml
sources:
  incremental_ingest: false
```
