#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${VENV_PY:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: venv python not found at ${VENV_PY}"
  echo "Create venv first (recommended): python3 -m venv .venv"
  exit 1
fi

"${VENV_PY}" -m pip install -U pip
"${VENV_PY}" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
