from __future__ import annotations

"""MCP doc_generate tool — retrieve context + call LLM for answer with citations."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def doc_generate_tool(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute `doc_generate` tool and return MCP `content` array.

    Args:
        arguments: JSON object with fields:
            - query: str (required)
            - top_k: int (optional, default 5)
            - namespace: str (optional, default "default")
            - max_tokens: int (optional, override LLM max_tokens)

    Returns:
        MCP content array with answer + structured citations.
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return [{"type": "text", "text": "Empty query. Provide `query` string."}]

    top_k_raw = arguments.get("top_k", 5)
    try:
        top_k = int(top_k_raw)
    except Exception:
        top_k = 5

    namespace = str(arguments.get("namespace", "default")).strip() or "default"
    max_tokens_override = arguments.get("max_tokens")

    from doc_rag.raglib.rag_generate import rag_generate
    from doc_rag.server.retrieval import load_config

    cfg = load_config()

    kwargs: dict[str, Any] = {
        "cfg": cfg,
        "query": query,
        "top_k": max(1, min(20, top_k)),
        "namespace": namespace,
    }
    if max_tokens_override is not None:
        try:
            kwargs["max_tokens"] = int(max_tokens_override)
        except (ValueError, TypeError):
            pass

    max_ctx = arguments.get("max_context_tokens")
    if max_ctx is not None:
        try:
            kwargs["max_context_tokens"] = int(max_ctx)
        except (ValueError, TypeError):
            pass

    result = rag_generate(**kwargs)

    content: list[dict[str, Any]] = []

    if result.get("error"):
        content.append({"type": "text", "text": f"Error: {result['error']}"})
        if result.get("sources"):
            citations = _format_citations(result["sources"])
            content.append({"type": "text", "text": citations})
        return content

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    model = result.get("model", "")

    if answer:
        header = f"Model: {model}\n\n" if model else ""
        content.append({"type": "text", "text": header + answer})

    if sources:
        citations = _format_citations(sources)
        content.append({"type": "text", "text": citations})

    return content


def _format_citations(sources: list[dict[str, Any]]) -> str:
    """Format sources as a readable citations block."""
    lines = ["\n---\nSources:"]
    for s in sources:
        idx = s.get("index", "?")
        source_file = s.get("source_file", "unknown")
        section = s.get("section_path", "")
        chunk_id = s.get("chunk_id", "")
        score = s.get("score")
        score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"

        parts = [f"[{idx}] {source_file}"]
        if section:
            parts.append(f"§{section}")
        if chunk_id:
            parts.append(f"({chunk_id})")
        parts.append(f"score={score_s}")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)
