#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PORT="${DOC_RAG_HTTP_PORT:-3333}"
BASE="http://127.0.0.1:${PORT}"

if ! curl -sf "${BASE}/health" >/dev/null; then
  echo "MCP HTTP not reachable at ${BASE}. Start: bash scripts/run_mcp_http.sh" >&2
  exit 1
fi

PY="${DOC_RAG_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY="python3"
  fi
fi

API_KEY="${DOC_RAG_API_KEY:-}"
AUTH_HEADERS=()
if [[ -n "${API_KEY}" ]]; then
  AUTH_HEADERS+=(-H "Authorization: Bearer ${API_KEY}")
fi

curl -fsS -X POST "${BASE}/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "${AUTH_HEADERS[@]}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

echo

curl -fsS -X POST "${BASE}/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "${AUTH_HEADERS[@]}" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

echo

QUERY="${DOC_RAG_VERIFY_QUERY:-conformance testing process}"
TOP_K="${DOC_RAG_VERIFY_TOP_K:-2}"

PAYLOAD="$("$PY" - <<'PY'
import json, os
query = os.environ.get("DOC_RAG_VERIFY_QUERY", "conformance testing process")
top_k_raw = os.environ.get("DOC_RAG_VERIFY_TOP_K", "2")
try:
    top_k = int(top_k_raw)
except Exception:
    top_k = 2
top_k = max(1, min(50, top_k))
print(json.dumps({
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {"name": "doc_search", "arguments": {"query": query, "top_k": top_k}},
}, ensure_ascii=False))
PY
)"

curl -fsS -X POST "${BASE}/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "${AUTH_HEADERS[@]}" \
  -d "${PAYLOAD}"

echo
