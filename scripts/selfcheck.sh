#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV_PY="${ROOT}/.venv/bin/python"
echo "[selfcheck] Root: ${ROOT}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "[selfcheck] .venv not found: ${VENV_PY}"
  echo "[selfcheck] Hint: run: bash scripts/bootstrap.sh"
  exit 2
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

CFG="${1:-config/config.yaml}"

# Extract paths from YAML via Python (no yq dependency).
readarray -t KV < <(
  "${VENV_PY}" - "${CFG}" <<'PY'
import sys, os, yaml
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
paths = cfg.get("paths", {})
print("CHUNKS=" + os.path.join(paths.get("chunks_dir", "build/chunks_jsonl"), "chunks.jsonl"))
print("INDEX=" + os.path.join(paths.get("index_dir", "build/index"), "faiss.index"))
print("EMB_MODEL=" + str(cfg.get("embeddings", {}).get("model_name", "")))
PY
)

CHUNKS="${KV[0]#CHUNKS=}"
INDEX="${KV[1]#INDEX=}"
EMB_MODEL="${KV[2]#EMB_MODEL=}"

echo "[selfcheck] chunks: ${CHUNKS}"
echo "[selfcheck] index : ${INDEX}"

ok=0

if [[ -f "${CHUNKS}" ]]; then
  echo "[selfcheck] ✅ chunks.jsonl found"
else
  echo "[selfcheck] ❌ chunks.jsonl NOT found"
  ok=1
fi

if [[ -f "${INDEX}" ]]; then
  echo "[selfcheck] ✅ faiss.index found"
else
  echo "[selfcheck] ❌ faiss.index NOT found (semantic search disabled)"
fi

# Check python deps presence (best-effort).
"${VENV_PY}" - <<'PY'
import importlib, sys
mods = ["faiss", "sentence_transformers", "torch"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[selfcheck] ✅ python module: {m}")
    except Exception as e:
        print(f"[selfcheck] ❌ python module: {m} ({e.__class__.__name__})")
PY

# Determine retrieval mode MCP will use.
mode="lexical"
if [[ -f "${INDEX}" ]]; then
  # semantic requires faiss + sentence_transformers
  if "${VENV_PY}" - <<'PY' 2>/dev/null
import importlib
importlib.import_module("faiss")
importlib.import_module("sentence_transformers")
print("ok")
PY
  then
    mode="semantic"
  fi
fi

echo "[selfcheck] MCP retrieval mode: ${mode}"
if [[ "${mode}" == "semantic" ]]; then
  echo "[selfcheck] model: ${EMB_MODEL:-<unknown>}"
fi

# Smoke test MCP HTTP: initialize + tools/list (server must listen on DOC_RAG_HTTP_PORT)
echo "[selfcheck] MCP HTTP smoke (needs server: bash scripts/run_mcp_http.sh)"
PORT="${DOC_RAG_HTTP_PORT:-3333}"
BASE="http://127.0.0.1:${PORT}"
if curl -sf "${BASE}/health" >/dev/null; then
  curl -sS -X POST "${BASE}/mcp" -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | head -c 200
  echo
else
  echo "[selfcheck] skip MCP HTTP (${BASE} down)"
fi

if [[ "${ok}" -ne 0 ]]; then
  echo "[selfcheck] NOTE: run ingest to generate chunks:"
  echo "  ${VENV_PY} -m doc_rag.cli ingest"
fi

exit "${ok}"
