#!/usr/bin/env bash
set -euo pipefail

# Universal bootstrap for doc-rag on Linux/WSL (v2.1+ uv-based).
# - Installs uv if not already on PATH (single curl one-liner).
# - Creates .venv via `uv sync` from the committed uv.lock.
# - Pulls base deps + opt-in extras (FAISS, embeddings, server, dev).
# - Optionally runs the initial ingest.
#
# Reproducibility: the committed uv.lock pins every transitive
# dependency. `uv sync --frozen` refuses to update the lock; pass
# DOC_RAG_BOOTSTRAP_FROZEN=0 to relax that (useful when changing
# pyproject.toml locally).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NONINTERACTIVE="${DOC_RAG_BOOTSTRAP_NONINTERACTIVE:-0}"
FROZEN="${DOC_RAG_BOOTSTRAP_FROZEN:-1}"

echo "[doc-rag] Root: ${ROOT}"

# --- Install uv if missing ---------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[doc-rag] uv not found on PATH — installing via Astral's official script…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer prints `Installed uv to <dir>`; bring it onto PATH for
  # this shell so the rest of the script can use it directly.
  if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi
echo "[doc-rag] uv: $(uv --version)"

# --- Decide which extras to install ------------------------------------------
EXTRA_ARGS=("--extra" "server")

if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_faiss="${DOC_RAG_BOOTSTRAP_FAISS:-Y}"
else
  read -r -p "[doc-rag] Install FAISS (recommended)? [Y/n] " ans_faiss || true
  ans_faiss="${ans_faiss:-Y}"
fi
if [[ "${ans_faiss}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  EXTRA_ARGS+=("--extra" "faiss")
fi

if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_dev="${DOC_RAG_BOOTSTRAP_DEV:-N}"
else
  read -r -p "[doc-rag] Install dev deps (pytest, ruff)? [y/N] " ans_dev || true
  ans_dev="${ans_dev:-N}"
fi
if [[ "${ans_dev}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  EXTRA_ARGS+=("--extra" "dev")
fi

# Embeddings (CPU-safe; torch picked from default index). For GPU/ROCm
# wheels, install torch separately after this script.
# NOTE: with the production default `embeddings.provider: ollama` this
# stack is only needed for the legacy local sentence-transformers branch;
# the ollama provider talks HTTP and works without torch. The choice
# below therefore defaults to N when config.yaml selects ollama.
CFG_PROVIDER=""
if [[ -f "${ROOT}/config/config.yaml" ]]; then
  CFG_PROVIDER="$("$ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || echo ""
try:
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml")) or {}
    print(str((cfg.get("embeddings") or {}).get("provider") or "").strip().lower())
except Exception:
    print("")
PY
)"
fi
OLLAMA_MODE=0
if [[ "${CFG_PROVIDER}" == "ollama" ]]; then
  OLLAMA_MODE=1
fi

if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_emb="${DOC_RAG_BOOTSTRAP_EMBEDDINGS:-}"
  if [[ -z "${ans_emb}" ]]; then ans_emb="$([[ ${OLLAMA_MODE} -eq 1 ]] && echo N || echo Y)"; fi
else
  if [[ ${OLLAMA_MODE} -eq 1 ]]; then
    read -r -p "[doc-rag] Install local embeddings stack (torch + sentence-transformers, legacy branch; provider=ollama makes it optional)? [y/N] " ans_emb || true
    ans_emb="${ans_emb:-N}"
  else
    read -r -p "[doc-rag] Install embeddings stack (torch + sentence-transformers, CPU)? [Y/n] " ans_emb || true
    ans_emb="${ans_emb:-Y}"
  fi
fi
if [[ "${ans_emb}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  EXTRA_ARGS+=("--extra" "embeddings")
fi

if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_metrics="${DOC_RAG_BOOTSTRAP_METRICS:-N}"
else
  read -r -p "[doc-rag] Install Prometheus /metrics extra? [y/N] " ans_metrics || true
  ans_metrics="${ans_metrics:-N}"
fi
if [[ "${ans_metrics}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  EXTRA_ARGS+=("--extra" "metrics")
fi

# Unstructured — second-tier PDF fallback for quality cascade
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_unstructured="${DOC_RAG_BOOTSTRAP_UNSTRUCTURED:-N}"
else
  read -r -p "[doc-rag] Install Unstructured (PDF quality cascade fallback)? [y/N] " ans_unstructured || true
  ans_unstructured="${ans_unstructured:-N}"
fi
if [[ "${ans_unstructured}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  EXTRA_ARGS+=("--extra" "unstructured")
fi

# --- Sync ---------------------------------------------------------------------
SYNC_ARGS=()
if [[ "${FROZEN}" == "1" ]]; then
  SYNC_ARGS+=("--frozen")
fi
SYNC_ARGS+=("${EXTRA_ARGS[@]}")

echo "[doc-rag] uv sync ${SYNC_ARGS[*]}"
uv sync "${SYNC_ARGS[@]}"
echo "[doc-rag] venv ready: ${ROOT}/.venv"

# --- Fix rapidocr missing arch_config.yaml (packaging bug in 3.8.x) ---------
ARCH_CFG_SRC="${ROOT}/scripts/rapidocr/arch_config.yaml"
ARCH_CFG_DST="${ROOT}/.venv/lib/python3.*/site-packages/rapidocr/inference_engine/pytorch/networks/arch_config.yaml"
if [[ -f "$ARCH_CFG_SRC" ]]; then
  for dst in $ARCH_CFG_DST; do
    if [[ -d "$(dirname "$dst")" ]] && [[ ! -f "$dst" ]]; then
      cp "$ARCH_CFG_SRC" "$dst"
      echo "[doc-rag] patched rapidocr: installed arch_config.yaml"
    fi
  done
fi

# --- Ensure onnxruntime is installed (rapidocr default engine) ----------------
if ! "$ROOT/.venv/bin/python" -c "import onnxruntime" 2>/dev/null; then
  echo "[doc-rag] onnxruntime missing — installing (rapidocr default engine)…"
  uv pip install --python "$ROOT/.venv/bin/python" onnxruntime
fi

# --- Offline embedding model ---------------------------------------------------
# The model is gitignored and may be absent (fresh clone, air-gapped install).
# Only fetched when the local embeddings stack was selected above AND the
# config does not use the ollama provider (whose weights live on the Ollama
# server, not in models/); DOC_RAG_BOOTSTRAP_FETCH_MODEL=Y/N overrides.
# Failure is non-fatal so air-gapped installs still bootstrap (the legacy
# branch will fail later until the model is provided).
MODEL_OK=0
if [[ -f "${ROOT}/models/bge-m3/config.json" && -f "${ROOT}/models/bge-m3/modules.json" ]]; then
  MODEL_OK=1
fi
FETCH_MODEL="${DOC_RAG_BOOTSTRAP_FETCH_MODEL:-}"
if [[ -z "${FETCH_MODEL}" ]]; then
  if [[ ${OLLAMA_MODE} -eq 1 ]]; then
    FETCH_MODEL="N"
  elif [[ "${ans_emb}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
    FETCH_MODEL="Y"
  else
    FETCH_MODEL="N"
  fi
fi
if [[ ${OLLAMA_MODE} -eq 1 ]]; then
  # provider=ollama: verify the Ollama server is reachable and has the model.
  API_BASE="$("$ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || echo "http://127.0.0.1:11434"
try:
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml")) or {}
    print(str((cfg.get("embeddings") or {}).get("api_base") or "http://127.0.0.1:11434"))
except Exception:
    print("http://127.0.0.1:11434")
PY
)"
  MODEL_NAME="$("$ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || echo "bge-m3"
try:
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml")) or {}
    print(str((cfg.get("embeddings") or {}).get("model_name") or "bge-m3"))
except Exception:
    print("bge-m3")
PY
)"
  if curl -fsS -m 3 "${API_BASE%/}/api/tags" >/dev/null 2>&1; then
    # /api/tags returns canonical names with tags ("bge-m3:latest"); the
    # config may reference the bare name, which Ollama resolves to :latest.
    MODEL_BASE="${MODEL_NAME%%:*}"
    if curl -fsS -m 3 "${API_BASE%/}/api/tags" | grep -Eq "\"${MODEL_BASE}(:[^\"]*)?\""; then
      echo "[doc-rag] Ollama OK: ${API_BASE} has model ${MODEL_NAME}"
    else
      echo "[doc-rag] WARNING: Ollama at ${API_BASE} is reachable but model '${MODEL_NAME}' is missing." >&2
      echo "[doc-rag] WARNING: run: ollama pull ${MODEL_NAME}" >&2
    fi
  else
    echo "[doc-rag] WARNING: Ollama is not reachable at ${API_BASE}." >&2
    echo "[doc-rag] WARNING: install+start Ollama, then: ollama pull ${MODEL_NAME}" >&2
    echo "[doc-rag] WARNING: until then doc-rag degrades to lexical search (no crash)." >&2
  fi
fi
if [[ "${MODEL_OK}" -eq 0 && "${FETCH_MODEL}" == "Y" ]]; then
  echo "[doc-rag] offline model models/bge-m3 not found -- fetching via scripts/fetch_model.py..."
  if "$ROOT/.venv/bin/python" scripts/fetch_model.py; then
    echo "[doc-rag] offline model ready: models/bge-m3"
  else
    echo "[doc-rag] WARNING: fetch_model.py failed (no network?). Bootstrap continues," >&2
    echo "[doc-rag] WARNING: but embeddings are unavailable offline. Fetch the model manually" >&2
    echo "[doc-rag] WARNING: later: ${ROOT}/.venv/bin/python scripts/fetch_model.py" >&2
  fi
fi

# --- Optional initial ingest --------------------------------------------------
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_ing="${DOC_RAG_BOOTSTRAP_INGEST:-N}"
else
  read -r -p "[doc-rag] Run initial ingest now? [y/N] " ans_ing || true
  ans_ing="${ans_ing:-N}"
fi
if [[ "${ans_ing}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  echo "[doc-rag] Running doc-rag ingest…"
  uv run doc-rag ingest
fi

echo "[doc-rag] Bootstrap complete."

# --- Svelte UI build (required) ----------------------------------------------
# The Svelte + Vite frontend under `ui/` is the only web UI doc-rag ships —
# there is no inline HTML fallback. `/ui` and `/ui-next/` 404 without a
# built `ui/dist`, so a missing Node/npm is a hard failure, not a skip.
if [[ -f "${ROOT}/ui/package.json" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "[doc-rag] ERROR: Node/npm not found on PATH." >&2
    echo "[doc-rag] The Web UI (Svelte, ui/) has no fallback and requires Node >= 20." >&2
    echo "[doc-rag] Install Node >= 20, then re-run this script (or run" >&2
    echo "[doc-rag] 'cd ui && npm ci && npm run build' manually and re-run bootstrap)." >&2
    exit 1
  fi
  echo "[doc-rag] Building Svelte UI bundle (ui/)…"
  pushd "${ROOT}/ui" >/dev/null
  npm ci --no-audit --no-fund
  npm run build
  popd >/dev/null
  echo "[doc-rag] ui/dist ready."
fi
