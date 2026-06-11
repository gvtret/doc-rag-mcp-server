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
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_emb="${DOC_RAG_BOOTSTRAP_EMBEDDINGS:-Y}"
else
  read -r -p "[doc-rag] Install embeddings stack (torch + sentence-transformers, CPU)? [Y/n] " ans_emb || true
  ans_emb="${ans_emb:-Y}"
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

# --- Sync ---------------------------------------------------------------------
SYNC_ARGS=()
if [[ "${FROZEN}" == "1" ]]; then
  SYNC_ARGS+=("--frozen")
fi
SYNC_ARGS+=("${EXTRA_ARGS[@]}")

echo "[doc-rag] uv sync ${SYNC_ARGS[*]}"
uv sync "${SYNC_ARGS[@]}"
echo "[doc-rag] venv ready: ${ROOT}/.venv"

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
