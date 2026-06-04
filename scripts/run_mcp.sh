#!/usr/bin/env bash
# Legacy entry: identical to HTTP MCP (stdio transport is not supported).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/run_mcp_http.sh"
