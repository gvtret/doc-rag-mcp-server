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

# Модель не найдена или не загрузилась (отличать от отсутствия индекса):
# rebuild тут не поможет, нужны сами веса -- scripts/fetch_model.py.
_MODEL_NOTICE = (
    "Семантический поиск недоступен: модель эмбеддингов не найдена и не загружена "
    "(offline-режим, весов нет ни в models/, ни в кэше). "
    "Ниже - результаты лексического поиска: они часто менее релевантны и могут "
    "пропустить семантически близкие формулировки. "
    "Выполните scripts/fetch_model.py, чтобы скачать модель, затем перезапустите сервис "
    "(или переключите mcp.retrieval_mode на lexical)."
)

# Индекс построен, но веса модели изменились (фингерпринты различаются):
# старый индекс не соответствует текущим весам.
_STALE_NOTICE = (
    "Семантический поиск недоступен: веса модели изменились после построения индекса "
    "(фингерпринты различаются), текущий FAISS-индекс устарел. "
    "Ниже - результаты лексического поиска. "
    "Выполните scripts/fetch_model.py (при необходимости вернуться к прежним весам), "
    "перезапустите сервис и запустите doc-rag rebuild, чтобы переиндексировать корпус."
)


def _build_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build structured citations array from search results."""
    citations: list[dict[str, Any]] = []
    for i, r in enumerate(results, 1):
        citations.append(
            {
                "index": i,
                "chunk_id": r.get("chunk_id", ""),
                "source_file": r.get("source_file", ""),
                "section_path": r.get("section_path", ""),
                "doc_id": r.get("doc_id", ""),
                "score": r.get("score"),
                "text_preview": str(r.get("text", ""))[:200],
            }
        )
    return citations


def doc_search_tool(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute `doc_search` tool and return MCP `content` array.

    Args:
        arguments: JSON object with fields:
            - query: str (required)
            - top_k: int (optional)
            - namespace: str (optional, default "default")
            - doc_id: str (optional, filter by document)
            - section_path: str (optional, filter by section)
            - tables_only: bool (optional, only table chunks)

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

    namespace = str(arguments.get("namespace", "default")).strip() or "default"

    filters: dict[str, Any] | None = None
    if "doc_id" in arguments:
        filters = filters or {}
        filters["doc_id"] = str(arguments["doc_id"])
    if "section_path" in arguments:
        filters = filters or {}
        filters["section_path"] = str(arguments["section_path"])
    if arguments.get("tables_only"):
        filters = filters or {}
        filters["is_table"] = True

    fallback_active = False
    notice: str | None = None
    try:
        cfg = load_config()
        mode = str((cfg.get("mcp", {}) or {}).get("retrieval_mode", "semantic")).lower()
        if mode in ("semantic", "hybrid"):
            cat = indexed_catalog()
            if not cat.get("semantic_search_ready"):
                fallback_active = True
                if cat.get("index_stale"):
                    # The index exists but was built with different weights.
                    notice = _STALE_NOTICE
                elif not cat.get("model_ready"):
                    # No loadable model: rebuilding the index would not help.
                    notice = _MODEL_NOTICE
                else:
                    # Model is fine; the index itself is missing/unbuilt.
                    notice = _FALLBACK_NOTICE
    except Exception:
        pass

    results = doc_search(query=query, top_k=top_k, namespace=namespace, filters=filters)

    content: list[dict[str, Any]] = []
    if fallback_active and notice:
        content.append({"type": "text", "text": notice})

    if not results:
        content.append({"type": "text", "text": "No results."})
        return content

    citations = _build_citations(results)

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("score", None)
        score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        source_file = r.get("source_file", "")
        section = r.get("section_path", "")

        header_bits = [f"[{i}]"]
        if source_file:
            header_bits.append(f"({source_file})")
        if section:
            header_bits.append(f"§{section}")
        header_bits.append(f"score={score_s}")

        text = str(r.get("text", "")).strip()
        if text:
            lines.append(" ".join(header_bits) + "\n" + text)
        else:
            lines.append(" ".join(header_bits))

    content.append({"type": "text", "text": "\n\n".join(lines)})

    content.append(
        {
            "type": "text",
            "text": "\n---\nCitations: " + str(citations),
        }
    )

    return content
