#!/usr/bin/env python3
"""Write doc-rag MCP Streamable HTTP config to build/mcp_global_example.json (for ~/.cursor/mcp.json)."""

import json
import os
import sys

DEFAULT_MCP_URL = os.environ.get("DOC_RAG_MCP_URL", "http://127.0.0.1:3333/mcp")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(root, "build", "mcp_global_example.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    config = {
        "mcpServers": {
            "doc-rag": {
                "transport": "streamableHttp",
                "url": DEFAULT_MCP_URL,
            }
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Config written to:", out_path)
    print(
        "Copy that file to ~/.cursor/mcp.json (or merge its mcpServers into your existing ~/.cursor/mcp.json)."
    )
    print(
        "Server must be running (e.g. bash scripts/run_mcp_http.sh). Override URL: DOC_RAG_MCP_URL=..."
    )
    print()
    with open(out_path, encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
