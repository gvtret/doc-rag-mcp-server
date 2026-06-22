#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# UI-managed runtime env (written by the Web UI's Service editor). Sourced
# here so it overrides systemd's /etc/default/doc-rag, which the service
# user cannot write. Values are single-quoted in the file; `set -a` exports
# them to the uvicorn process. Override the path with DOC_RAG_ENV_FILE.
ENV_FILE="${DOC_RAG_ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# Use project venv if present, otherwise fall back to python3 in PATH.
PY="${DOC_RAG_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY="python3"
  fi
fi

HOST="${DOC_RAG_HTTP_HOST:-0.0.0.0}"
PORT="${DOC_RAG_HTTP_PORT:-3333}"

# Time uvicorn waits for in-flight requests to finish on SIGTERM before
# force-closing connections. systemd's default TimeoutStopSec is 90s, so
# stay safely below that.
SHUTDOWN_TIMEOUT="${DOC_RAG_SHUTDOWN_TIMEOUT:-30}"

exec "$PY" -m uvicorn doc_rag.server.mcp_http:app \
  --host "$HOST" \
  --port "$PORT" \
  --timeout-graceful-shutdown "$SHUTDOWN_TIMEOUT"
