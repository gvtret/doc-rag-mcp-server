#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
