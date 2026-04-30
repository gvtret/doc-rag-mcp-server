from __future__ import annotations

"""HTTP MCP tool helpers."""

from typing import Any, Dict, List

from doc_rag.server.retrieval import doc_search


def doc_search_tool(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute `doc_search` tool and return MCP `content` array.

    Args:
        arguments: JSON object with fields:
            - query: str (required)
            - top_k: int (optional)

    Returns:
        MCP content array (list of {type,text} objects).
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return [{"type": "text", "text": "Empty query. Provide `query` string."}]

    top_k_raw = arguments.get("top_k", 6)
    try:
        top_k = int(top_k_raw)
    except Exception:
        top_k = 6

    results = doc_search(query=query, top_k=top_k)

    if not results:
        return [{"type": "text", "text": "No results."}]

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("score", None)
        score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        doc_id = r.get("doc_id", "")
        chunk_id = r.get("chunk_id", "")
        source_file = r.get("source_file", "")

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

    return [{"type": "text", "text": "\n\n".join(lines)}]
