from __future__ import annotations

"""MCP tool registry."""

from typing import Any

from .base import BaseTool, ToolSpec
from .search import DocSearchTool


class ToolRegistry:
    """Simple in-process registry.

    Keep this tiny and deterministic. No dynamic imports, no plugin discovery.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.register(DocSearchTool())

    def register(self, tool: BaseTool) -> None:
        spec = tool.spec()
        self._tools[spec.name] = tool

    def list_specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        tool = self._tools.get(name)
        if tool is None:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]
        return tool.call(arguments)
