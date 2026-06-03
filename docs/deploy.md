# Deploy

Three deployment styles, in increasing order of operational seriousness:

1. [Docker Compose](#docker-compose) — single-command LAN deploy
2. [Native systemd](#native-systemd) — production server, persists across upgrades
3. [Deploy archive](#deploy-archive) — for air-gapped or `scp`-only environments

## Docker Compose

```bash
cp .env.example .env
docker compose up -d --build
curl -sS http://127.0.0.1:3333/health
```

Volumes (defined in `docker-compose.yml`):

- `./build` — manifest + FAISS index, **persists across `docker compose down`**
- `./config` (ro)
- `./sources`

Environment variables of interest in `docker-compose.yml`:

- `DOC_RAG_HTTP_HOST` / `DOC_RAG_HTTP_PORT`
- `DOC_RAG_ALLOWED_ORIGINS` (CSV; required if your client sends `Origin`)
- `DOC_RAG_API_KEY` (commented out by default; uncomment to require auth)
- `DOC_RAG_HTTP_LOG` (default `/app/build/http.log`)

## Native systemd

For a Linux server you control. Boot-on-startup, single-user `docrag`, repo at
`/opt/doc-rag-mcp/`.

```bash
sudo bash scripts/install_server_native.sh
# flags: --cpu (default) | --gpu | --minimal
# positional: optional INSTALL_ROOT (default /opt/doc-rag-mcp)
```

What it does:

1. `apt install`s base packages: `python3-venv`, `build-essential`, `tesseract-ocr*`, `antiword`.
2. Creates system user `docrag` with `/var/lib/docrag` home.
3. `rsync`s the repo into `INSTALL_ROOT`, protecting any existing `build/`
   directory — manifest and FAISS index survive code upgrades.
4. Runs `scripts/bootstrap.sh` non-interactively under `docrag`.
5. Renders `systemd/doc-rag-mcp.service.in` → `/etc/systemd/system/doc-rag-mcp.service`.
6. Writes `/etc/default/doc-rag` (first install only; subsequent installs don't overwrite).
7. `systemctl enable --now doc-rag-mcp`.

After install:

```bash
systemctl status doc-rag-mcp
journalctl -u doc-rag-mcp -f
sudo -u docrag -H bash -lc 'cd /opt/doc-rag-mcp && .venv/bin/doc-rag ingest'
```

To tune the service: `sudo nano /etc/default/doc-rag && sudo systemctl restart doc-rag-mcp`.

### Restarting from a deploy user

A common pattern: dedicated `deploy` SSH user, restricted sudoers to only restart the
unit. Sample `/etc/sudoers.d/doc-rag-deploy`:

```
deploy ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart doc-rag-mcp, /usr/bin/systemctl stop doc-rag-mcp, /usr/bin/systemctl start doc-rag-mcp
Defaults:deploy !requiretty
Defaults:deploy !use_pty
```

## Deploy archive

Build a tarball from the current `HEAD` (excludes `sources/`, `.venv/`, `build/`):

```bash
bash scripts/make_deploy_archive.sh
# → doc-rag-deploy-YYYYMMDD-<sha>.tar.gz
```

Ship documents too (archive may be hundreds of MB):

```bash
bash scripts/make_deploy_archive.sh --with-docs
```

On the target:

```bash
scp doc-rag-deploy-*.tar.gz user@<server>:~
ssh user@<server>
tar xzf doc-rag-deploy-*.tar.gz
cd doc-rag
# then either:
sudo bash scripts/install_server_native.sh
# or:
cp .env.example .env && docker compose up -d --build
```

## Remote MCP

Once the server is up on `<host>:3333`, configure Cursor/Claude to talk to it. See
[mcp.md](mcp.md#remote-server) for the JSON snippet and CORS notes.

## Upgrades

| Method | Upgrade procedure |
| --- | --- |
| Docker Compose | `git pull && docker compose up -d --build` |
| Native systemd | `git pull && sudo bash scripts/install_server_native.sh` (build/ is preserved) |
| Deploy archive | scp new archive, `tar xzf`, run `install_server_native.sh` |

The FAISS index does **not** need to be rebuilt for code-only upgrades.
Rebuild only if the chunk schema or embedding model changes.

## Backups

The interesting state is:

- `build/manifest.json`
- `build/chunks_jsonl/chunks.jsonl`
- `build/index/faiss.index` + `meta.json`
- `sources/archived/` (originals, if you care about retaining them)

Everything under `build/` can be regenerated from `sources/archived/` via `doc-rag rebuild`
(or full `ingest` if the markdown is gone too), so the original PDFs are the only
truly irreplaceable artefact.
