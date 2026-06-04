from __future__ import annotations

"""Tool base classes for MCP.

The HTTP MCP layer should stay transport-only.
All business logic for tools lives under :mod:`doc_rag.tools`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Tool metadata exposed via MCP `tools/list`."""

    name: str
    description: str
    input_schema: dict[str, Any]


class BaseTool:
    """Base class for all MCP tools."""

    def spec(self) -> ToolSpec:
        raise NotImplementedError

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute tool.

        Returns MCP `content` array, e.g. ``[{"type":"text","text":"..."}]``.
        """
        raise NotImplementedError
