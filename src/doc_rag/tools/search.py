from __future__ import annotations

"""`doc_search` tool implementation.

This module reuses the same retrieval logic as the HTTP MCP server to avoid drift.
"""

from typing import Any

from doc_rag.server.retrieval import doc_search

from .base import BaseTool, ToolSpec


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No results."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("score", None)
        score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"

        doc_id = str(r.get("doc_id", "") or "")
        chunk_id = str(r.get("chunk_id", "") or "")
        source_file = str(r.get("source_file", "") or "")

        header_bits = [f"{i}. ({score_s})"]
        if doc_id:
            header_bits.append(f"doc_id={doc_id}")
        if chunk_id:
            header_bits.append(f"chunk_id={chunk_id}")
        if source_file:
            header_bits.append(f"source={source_file}")

        text = str(r.get("text", "")).strip()
        if text:
            lines.append(" ".join(header_bits) + "\n" + text)
        else:
            lines.append(" ".join(header_bits))

    return "\n\n".join(lines)


class DocSearchTool(BaseTool):
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="doc_search",
            description="Search the document knowledge base (semantic if FAISS+embeddings are available).",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 6},
                },
                "required": ["query"],
            },
        )

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return [{"type": "text", "text": "Empty query. Provide `query` string."}]

        top_k_raw = arguments.get("top_k", 6)
        try:
            top_k = int(top_k_raw)
        except Exception:
            top_k = 6
        top_k = max(1, min(50, top_k))

        results = doc_search(query=query, top_k=top_k)
        return [{"type": "text", "text": _format_results(results)}]
