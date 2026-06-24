"""RAG generate endpoint — retrieve chunks + call LLM for answer with citations.

Supports OpenAI-compatible APIs (Ollama, llama.cpp, vLLM, OpenAI).
Config: mcp.rag_generate (base_url, model, api_key, max_tokens, temperature).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question based ONLY on "
    "the provided context. Cite sources using [1], [2], etc. matching the "
    "source numbers. If the context does not contain enough information to "
    "answer, say so explicitly. Be concise."
)


def _get_rag_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Extract RAG generate config from the main config."""
    rag = cfg.get("mcp", {}).get("rag_generate") or {}
    base_url = rag.get("base_url") or os.environ.get("DOC_RAG_LLM_BASE_URL", "")
    model = rag.get("model") or os.environ.get("DOC_RAG_LLM_MODEL", "")
    if not base_url or not model:
        return None
    return {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key": rag.get("api_key") or os.environ.get("DOC_RAG_LLM_API_KEY", ""),
        "max_tokens": rag.get("max_tokens", 1024),
        "temperature": rag.get("temperature", 0.3),
    }


def _format_context(
    results: list[dict[str, Any]],
    max_context_tokens: int = 6000,
) -> tuple[str, list[dict[str, Any]]]:
    """Format search results into numbered context block.

    Truncates chunks to fit within max_context_tokens budget (~4 chars/token).
    """
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    budget = max_context_tokens * 4  # rough char estimate
    used = 0
    for i, r in enumerate(results, 1):
        text = str(r.get("text", "")).strip()
        source_file = str(r.get("source_file", "unknown"))
        chunk_id = str(r.get("chunk_id", ""))
        section = str(r.get("section_path", ""))
        header = f"[{i}] ({source_file}, {chunk_id})\n"
        header_len = len(header)
        remaining = budget - used - header_len
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0] + "..."
        used += header_len + len(text) + 2  # +2 for \n\n
        lines.append(header + text)
        sources.append(
            {
                "index": i,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "section_path": section,
                "score": r.get("score"),
            }
        )
    return "\n\n".join(lines), sources


def _call_llm(
    rag_cfg: dict[str, Any],
    system: str,
    user: str,
) -> str | None:
    """Call an OpenAI-compatible chat completion endpoint."""
    import urllib.error
    import urllib.request

    url = f"{rag_cfg['base_url']}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if rag_cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {rag_cfg['api_key']}"

    payload = json.dumps(
        {
            "model": rag_cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": rag_cfg.get("max_tokens", 1024),
            "temperature": rag_cfg.get("temperature", 0.3),
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {})
            return msg.get("content", "")
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.error("LLM call failed: %s", exc)
        return None


def rag_generate(
    cfg: dict[str, Any],
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    max_context_tokens: int = 6000,
) -> dict[str, Any]:
    """Full RAG pipeline: search → format context → LLM generate.

    Returns dict with keys: answer, sources, query, model, error.
    """
    from doc_rag.server.retrieval import doc_search

    rag_cfg = _get_rag_config(cfg)
    if rag_cfg is None:
        return {
            "answer": None,
            "sources": [],
            "query": query,
            "model": None,
            "error": "LLM not configured. Set mcp.rag_generate.base_url and "
            "mcp.rag_generate.model in config.yaml, or DOC_RAG_LLM_BASE_URL "
            "and DOC_RAG_LLM_MODEL env vars.",
        }

    # Retrieve
    results = doc_search(query, top_k, namespace=namespace)
    if not results:
        return {
            "answer": "No relevant documents found for this query.",
            "sources": [],
            "query": query,
            "model": rag_cfg["model"],
            "error": None,
        }

    context, sources = _format_context(results, max_context_tokens)

    # Generate
    user_msg = f"Context:\n\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
    answer = _call_llm(rag_cfg, _SYSTEM_PROMPT, user_msg)

    if answer is None:
        return {
            "answer": None,
            "sources": sources,
            "query": query,
            "model": rag_cfg["model"],
            "error": "LLM request failed. Check base_url and model configuration.",
        }

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
        "model": rag_cfg["model"],
        "error": None,
    }
