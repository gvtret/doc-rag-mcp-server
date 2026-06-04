#!/usr/bin/env bash
# Print MCP config for doc-rag to paste into ~/.cursor/mcp.json
# so Agent gets doc_search even when workspace MCP isn't loaded.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"
exec "$PYTHON" "${ROOT}/scripts/write_global_mcp_config.py"
