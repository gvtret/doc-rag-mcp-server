from __future__ import annotations

"""HTTP MCP tool helpers."""

from typing import Any

from doc_rag.server.retrieval import doc_search, indexed_catalog, load_config

_FALLBACK_NOTICE = (
    "⚠ Семантический поиск недоступен (FAISS-индекс отсутствует или ещё не построен). "
    "Ниже — результаты лексического поиска: они часто менее релевантны и могут "
    "пропустить семантически близкие формулировки. "
    "Откройте /ui и нажмите «Rebuild индекса» (займёт ~30–60 мин), затем повторите запрос."
)


def doc_search_tool(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute `doc_search` tool and return MCP `content` array.

    Args:
        arguments: JSON object with fields:
            - query: str (required)
            - top_k: int (optional)
            - namespace: str (optional, default "default")

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

    # namespace is reserved for future multi-collection support
    _namespace = str(arguments.get("namespace", "default")).strip() or "default"

    # Detect whether semantic mode is configured but the FAISS index isn't
    # available — in that case doc_search() silently falls back to lexical
    # search and the caller should know the quality is degraded.
    fallback_active = False
    try:
        cfg = load_config()
        mode = str((cfg.get("mcp", {}) or {}).get("retrieval_mode", "semantic")).lower()
        if mode == "semantic":
            cat = indexed_catalog()
            if not cat.get("semantic_search_ready"):
                fallback_active = True
    except Exception:
        pass

    results = doc_search(query=query, top_k=top_k, namespace=_namespace)

    content: list[dict[str, Any]] = []
    if fallback_active:
        content.append({"type": "text", "text": _FALLBACK_NOTICE})

    if not results:
        content.append({"type": "text", "text": "No results."})
        return content

    lines: list[str] = []
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

    content.append({"type": "text", "text": "\n\n".join(lines)})
    return content
