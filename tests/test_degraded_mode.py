"""Contract tests for degraded-mode signalling.

When semantic mode is configured but the FAISS index is not on disk,
two things must hold:

1. `semantic_search(...)` returns `None` (it must never try to build the
   index inline — that was the FAISS-rebuild-in-handler incident that
   pinned the box for hours).
2. `doc_search_tool(...)` returns a content array whose first element
   is the configured warning text, regardless of whether lexical results
   come back.

Both invariants together are the "degraded-mode contract" advertised in
the README and in the published Habr article.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _config_with_no_index(tmp_corpus_root: Path) -> Dict[str, Any]:
    """Load the test config and confirm there is no FAISS index on disk."""
    from doc_rag.server.retrieval import load_config

    cfg = load_config()  # honours DOC_RAG_ROOT
    index_dir = tmp_corpus_root / "build" / "index"
    assert not (index_dir / "faiss.index").exists(), "test setup expects no index"
    return cfg


def test_semantic_search_returns_none_when_index_missing(tmp_corpus_root):
    """The headline guard against the 'inline rebuild' regression."""
    from doc_rag.server.retrieval import semantic_search

    cfg = _config_with_no_index(tmp_corpus_root)
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "alpha", "source_file": "a"},
        {"chunk_id": "c2", "doc_id": "d2", "text": "beta", "source_file": "b"},
    ]

    out = semantic_search(cfg, chunks, "anything", top_k=2)

    assert out is None


def test_doc_search_tool_prepends_degraded_notice_when_index_missing(
    tmp_corpus_root,
):
    """`doc_search_tool` must alert the MCP client when semantic is down."""
    from doc_rag.server.search_tool import _FALLBACK_NOTICE, doc_search_tool

    _config_with_no_index(tmp_corpus_root)

    # Drop a tiny chunks.jsonl so lexical search has something to work with.
    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps({
            "chunk_id": "doc-1:0000",
            "doc_id": "doc-1",
            "text": "the unique marker word zeppelin appears here",
            "source_file": "doc-1.txt",
        }) + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "zeppelin", "top_k": 3})

    # First content item is the degraded-mode notice.
    assert content, "expected at least one content item"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _FALLBACK_NOTICE


def test_doc_search_tool_empty_query_short_circuits(tmp_corpus_root):
    """Empty query must not even reach the search path."""
    from doc_rag.server.search_tool import doc_search_tool

    _config_with_no_index(tmp_corpus_root)

    content = doc_search_tool({"query": "   "})

    assert len(content) == 1
    assert "Empty query" in content[0]["text"]


def test_doc_search_tool_no_notice_when_semantic_disabled(
    tmp_corpus_root, monkeypatch
):
    """If the operator disables semantic mode, there's nothing to warn about.

    We rewrite config to retrieval_mode: lexical and verify the notice is
    not added even though there is no FAISS index either.
    """
    import yaml

    from doc_rag.server.search_tool import _FALLBACK_NOTICE, doc_search_tool

    config_path = tmp_corpus_root / "config" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("mcp", {})["retrieval_mode"] = "lexical"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Cache invalidation: retrieval module caches load_config()'s result.
    from doc_rag.server import retrieval as r

    monkeypatch.setattr(r, "_CONFIG_CACHE", {}, raising=False)

    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps({
            "chunk_id": "doc-1:0000",
            "doc_id": "doc-1",
            "text": "no marker here",
            "source_file": "doc-1.txt",
        }) + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "marker", "top_k": 3})

    assert content
    # No degraded notice as the first item.
    assert content[0]["text"] != _FALLBACK_NOTICE
