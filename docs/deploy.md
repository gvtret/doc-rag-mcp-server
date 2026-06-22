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

### UI-managed env (`<root>/.env`)

The Web UI's **Сервис (env)** tab writes a `<INSTALL_ROOT>/.env` file (the
service user can write it; `/etc/default/doc-rag` is root-owned and cannot be
edited from the UI). `scripts/run_mcp_http.sh` sources this file at startup
and it **overrides** `/etc/default/doc-rag`, so a key set in both wins from
`.env`. Override the path with `DOC_RAG_ENV_FILE`. Changes apply on the next
`systemctl restart doc-rag-mcp` (the UI's Restart button, if sudoers allows
it). `DOC_RAG_API_KEY` is never written from the UI — manage the key here or
in `/etc/default/doc-rag`.

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

- `build/manifest.json` (with `schema_version`)
- `build/chunks_jsonl/chunks.jsonl`
- `build/index/faiss.index` + `index_meta.json`
- `build/embeddings/`
- `build/audit.log` (append-only history of destructive operations)
- `sources/archived/` (originals, if you care about retaining them)

Everything under `build/` can be regenerated from `sources/archived/`
via `doc-rag rebuild` (or full `ingest` if the markdown is gone too),
so the original source documents are the only truly irreplaceable
artefact.

### Bundled backup and restore scripts

```bash
# Back up everything but sources/archived (the cheap, fast option):
scripts/backup.sh                              # → ./doc-rag-backup-YYYYMMDD-HHMMSS.tar.gz

# Include the original archived sources (large, may be hundreds of MB):
scripts/backup.sh --with-archived

# Back up a non-default install root:
scripts/backup.sh --root /opt/doc-rag-mcp --output /var/backups/doc-rag

# Restore. Refuses to overwrite a populated build/ without --force:
scripts/restore.sh doc-rag-backup-YYYYMMDD-HHMMSS.tar.gz --root /opt/doc-rag-mcp
```

Both scripts use only `tar` and `sha256sum`. The backup embeds a
`MANIFEST.sha256` so `restore.sh` can verify the archive before
touching the live tree.

### Observability

- Liveness probe: `GET /health/live` — always 200 while the process is up.
- Readiness probe: `GET /health/ready` — 503 if no manifest exists or
  an ingest/rebuild is in flight.
- Prometheus scrape: `GET /metrics`. Requires the `[metrics]` extra
  (`pip install -e .[metrics]`); otherwise the endpoint returns 503
  with a hint.
- Destructive operations are recorded line-by-line in
  `build/audit.log` (append-only JSONL).
- Logs: `journalctl -u doc-rag-mcp -f`. Set `DOC_RAG_LOG_FORMAT=json`
  in `/etc/default/doc-rag` for structured output ingestible by a log
  shipper. Every line carries a `request_id` when emitted during a
  request.

### Graceful shutdown

`scripts/run_mcp_http.sh` passes `--timeout-graceful-shutdown` to
uvicorn (default 30 s, override via `DOC_RAG_SHUTDOWN_TIMEOUT`).
The bundled systemd unit declares `KillSignal=SIGTERM` and
`TimeoutStopSec=60`. On `systemctl stop doc-rag-mcp`, in-flight
requests have up to 30 s to finish before connections are force-closed.

## Scheduled ingest

`doc-rag ingest` is incremental — it only processes files newly
dropped into `sources/incoming/`. On CPU-only servers a full ingest
of a fresh batch can take from minutes to hours depending on the
corpus size, so the typical production pattern is to **run ingest
on a schedule rather than on demand**: operators drop files into
`sources/incoming/` during the day, the actual work happens off-hours.

Two implementations work equally well. Pick whichever fits your
environment.

### Option 1 — `cron`

```cron
# Nightly ingest at 02:00 — runs as the `docrag` system user
0 2 * * * /opt/doc-rag-mcp/.venv/bin/doc-rag --config /opt/doc-rag-mcp/config/config.yaml ingest >> /var/log/doc-rag-ingest.log 2>&1
```

Put this in `/etc/cron.d/doc-rag-ingest` with `docrag` as the user
column. The script logs to stdout/stderr; rotate
`/var/log/doc-rag-ingest.log` with logrotate if you want bounded log
size.

### Option 2 — systemd timer

The unit + timer pair below schedules ingest at 02:00 every day and
gives it up to 6 hours to finish before systemd considers it failed.

`/etc/systemd/system/doc-rag-ingest.service`:

```ini
[Unit]
Description=doc-rag ingest (nightly)
After=network-online.target

[Service]
Type=oneshot
User=docrag
Group=docrag
WorkingDirectory=/opt/doc-rag-mcp
EnvironmentFile=-/etc/default/doc-rag
ExecStart=/opt/doc-rag-mcp/.venv/bin/doc-rag --config /opt/doc-rag-mcp/config/config.yaml ingest
TimeoutStartSec=6h
Nice=10
IOSchedulingClass=idle
```

`/etc/systemd/system/doc-rag-ingest.timer`:

```ini
[Unit]
Description=Nightly ingest for doc-rag

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=10min

[Install]
WantedBy=timers.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now doc-rag-ingest.timer
systemctl list-timers doc-rag-ingest.timer
```

The `Nice=10` + `IOSchedulingClass=idle` lines keep the ingest job
from contending with the live `doc-rag-mcp` service for CPU and
disk when both are running.

### Triggering ingest by accumulated-files threshold

If "nightly" is the wrong cadence — for example, a burst of new
documents needs to be indexed within an hour rather than wait for
02:00 — wrap `doc-rag ingest` in a watcher that counts files in
`sources/incoming/` and fires when the count crosses a threshold.
A 20-line shell script polling every 5 minutes is enough for most
home and small-office deployments; a richer
[inotify](https://man7.org/linux/man-pages/man7/inotify.7.html)-driven
service is overkill for the typical doc-rag corpus.

### What ingest produces

After a scheduled run, the next `GET /health/ready` and the document
table in the Web UI both reflect the new state automatically. The
audit log (`build/audit.log`) records nothing — ingest is not in the
destructive-operations set — but the structured log captures the
duration and outcome. To detect failures programmatically, scrape
`doc_rag_ingest_errors_total` from `/metrics`.
