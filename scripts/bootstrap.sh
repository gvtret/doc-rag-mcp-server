#!/usr/bin/env bash
set -euo pipefail

# Universal bootstrap for doc-rag on Linux/WSL.
# - Creates venv if missing
# - Installs base deps (incl. Docling, the only PDF backend since v2.0)
# - Optionally installs FAISS
# - Optionally installs embeddings stack (torch + sentence-transformers)
# - Optionally runs ingest

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="${ROOT}/${VENV_DIR}/bin/python"
PIP="${VENV_PY} -m pip"
NONINTERACTIVE="${DOC_RAG_BOOTSTRAP_NONINTERACTIVE:-0}"

echo "[doc-rag] Root: ${ROOT}"
echo "[doc-rag] Python: ${PYTHON_BIN}"
echo "[doc-rag] Venv: ${VENV_DIR}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "[doc-rag] Creating venv..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "[doc-rag] Upgrading pip..."
${PIP} install -U pip

echo "[doc-rag] Installing base deps (Cursor-safe)..."
${PIP} install -r requirements.txt
${PIP} install -e .

# Optional: Dev deps (tests)
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_dev="${DOC_RAG_BOOTSTRAP_DEV:-N}"
else
  read -r -p "[doc-rag] Install dev deps (pytest)? [y/N] " ans_dev || true
  ans_dev="${ans_dev:-N}"
fi
if [[ "${ans_dev}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  echo "[doc-rag] Installing dev deps..."
  ${PIP} install -e ".[dev]"
fi

# Optional: FAISS
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_faiss="${DOC_RAG_BOOTSTRAP_FAISS:-Y}"
else
  read -r -p "[doc-rag] Install FAISS (recommended)? [Y/n] " ans_faiss || true
  ans_faiss="${ans_faiss:-Y}"
fi
if [[ "${ans_faiss}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  echo "[doc-rag] Installing faiss-cpu..."
  ${PIP} install "faiss-cpu>=1.8.0"
fi

# Docling ships as a base dep since v2.0 — `pip install -e .` above
# already pulled it in. OCR for scanned PDFs is handled internally by
# Docling via RapidOCR; no separate Tesseract install is required.

# Optional: Embeddings stack
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_torch="${DOC_RAG_BOOTSTRAP_TORCH:-3}"
else
  echo ""
  echo "[doc-rag] Embeddings are optional but required for building vectors/index."
  echo "[doc-rag] Choose torch variant:"
  echo "  1) GPU (CUDA cu124 via PyTorch index)"
  echo "  2) CPU (via PyTorch index)"
  echo "  3) Skip (only parsing/MD/JSON, no embeddings/index)"
  read -r -p "[doc-rag] Select [1/2/3] (default 3): " ans_torch || true
  ans_torch="${ans_torch:-3}"
fi

if [[ "${ans_torch}" == "1" ]]; then
  echo "[doc-rag] Installing torch GPU (cu124)..."
  VENV_PY="${VENV_PY}" bash scripts/install_torch_gpu.sh
  echo "[doc-rag] Installing sentence-transformers..."
  ${PIP} install -r requirements-embed.txt
elif [[ "${ans_torch}" == "2" ]]; then
  echo "[doc-rag] Installing torch CPU..."
  VENV_PY="${VENV_PY}" bash scripts/install_torch_cpu.sh
  echo "[doc-rag] Installing sentence-transformers..."
  ${PIP} install -r requirements-embed.txt
else
  echo "[doc-rag] Skipping embeddings stack."
fi

# Optional: Server deps (debug)
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_srv="${DOC_RAG_BOOTSTRAP_SERVER:-N}"
else
  read -r -p "[doc-rag] Install HTTP debug server deps (fastapi/uvicorn)? [y/N] " ans_srv || true
  ans_srv="${ans_srv:-N}"
fi
if [[ "${ans_srv}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  echo "[doc-rag] Installing server deps..."
  ${PIP} install -e ".[server]"
fi

# Optional: Run ingest
if [[ "${NONINTERACTIVE}" == "1" ]]; then
  ans_ing="${DOC_RAG_BOOTSTRAP_INGEST:-N}"
else
  read -r -p "[doc-rag] Run ingest now? [y/N] " ans_ing || true
  ans_ing="${ans_ing:-N}"
fi
if [[ "${ans_ing}" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
  echo "[doc-rag] Running: doc-rag ingest"
  "${VENV_PY}" -m doc_rag.cli ingest
  echo "[doc-rag] Done."
else
  echo "[doc-rag] Bootstrap complete."
  echo "Next steps:"
  echo "  - Put docs into sources/incoming/"
  echo "  - Run: ${VENV_PY} -m doc_rag.cli ingest"
  echo "  - Restart Cursor (MCP reads .cursor/mcp.json)"
fi
