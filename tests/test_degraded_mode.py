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
from typing import Any


def _config_with_no_index(tmp_corpus_root: Path) -> dict[str, Any]:
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
    from doc_rag.server.search_tool import _MODEL_NOTICE, doc_search_tool

    _config_with_no_index(tmp_corpus_root)

    # Drop a tiny chunks.jsonl so lexical search has something to work with.
    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "text": "the unique marker word zeppelin appears here",
                "source_file": "doc-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "zeppelin", "top_k": 3})

    # First content item is the degraded-mode notice. With no FAISS index
    # and no loadable model the branch is the model-specific notice, not
    # the generic rebuild hint.
    assert content, "expected at least one content item"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _MODEL_NOTICE


def test_doc_search_tool_fallback_notice_when_model_ready_but_no_index(
    tmp_corpus_root, monkeypatch
):
    """Model loads fine, but the FAISS index was never built."""
    from doc_rag.server.search_tool import _FALLBACK_NOTICE, doc_search_tool

    _config_with_no_index(tmp_corpus_root)
    _patch_model_status(monkeypatch, ready=True, fingerprint="fp")

    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "text": "the unique marker word zeppelin appears here",
                "source_file": "doc-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "zeppelin", "top_k": 3})

    assert content, "expected at least one content item"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _FALLBACK_NOTICE


def test_doc_search_tool_stale_notice_when_fingerprint_mismatch(tmp_corpus_root, monkeypatch):
    """Index exists and model loads, but weights changed since the build."""
    from doc_rag.server.search_tool import _STALE_NOTICE, doc_search_tool

    _config_with_no_index(tmp_corpus_root)
    _patch_model_status(monkeypatch, ready=True, fingerprint="new")

    # An index whose fingerprint does not match the (patched) model's.
    index_dir = tmp_corpus_root / "build" / "index"
    (index_dir / "faiss.index").write_bytes(b"stub")
    (index_dir / "index_meta.json").write_text(
        json.dumps({"weights_sha256": "old"}), encoding="utf-8"
    )

    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "text": "the unique marker word zeppelin appears here",
                "source_file": "doc-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "zeppelin", "top_k": 3})

    assert content, "expected at least one content item"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == _STALE_NOTICE


def test_doc_search_tool_degraded_notice_when_embeddings_unavailable(tmp_corpus_root, monkeypatch):
    """An open embedding breaker must be announced, not silently swallowed."""
    from doc_rag.server.search_tool import DEGRADED_NOTICE, doc_search_tool

    _config_with_no_index(tmp_corpus_root)
    _patch_model_status(monkeypatch, ready=True, fingerprint="fp", degraded=True)

    index_dir = tmp_corpus_root / "build" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "faiss.index").write_bytes(b"stub")
    (index_dir / "index_meta.json").write_text(
        json.dumps({"weights_sha256": "fp"}), encoding="utf-8"
    )
    (tmp_corpus_root / "build" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "documents": []}), encoding="utf-8"
    )

    chunks_path = tmp_corpus_root / "build" / "chunks_jsonl" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "text": "the unique marker word zeppelin appears here",
                "source_file": "doc-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "zeppelin", "top_k": 3})

    assert content, "expected at least one content item"
    assert content[0]["type"] == "text"
    assert content[0]["text"] == DEGRADED_NOTICE


def test_doc_search_tool_degraded_notice_with_otherwise_healthy_index(tmp_corpus_root, monkeypatch):
    """Only the breaker is open: index and model are fine.

    Isolates the `embeddings_degraded` branch from the generic
    "semantic_search_ready is False" branch.
    """
    import doc_rag.server.search_tool as search_tool_module

    _config_with_no_index(tmp_corpus_root)
    monkeypatch.setattr(
        search_tool_module,
        "indexed_catalog",
        lambda: {
            "semantic_search_ready": True,
            "model_ready": True,
            "index_stale": False,
            "embeddings_degraded": True,
        },
    )
    monkeypatch.setattr(search_tool_module, "doc_search", lambda *a, **k: [])

    content = search_tool_module.doc_search_tool({"query": "anything"})

    assert content[0]["text"] == search_tool_module.DEGRADED_NOTICE


def _patch_model_status(
    monkeypatch,
    *,
    ready: bool,
    fingerprint: str | None = None,
    degraded: bool = False,
):
    """Pin model_loader.status() for notice-branch tests."""
    from doc_rag.raglib import model_loader as model_loader

    monkeypatch.setattr(
        model_loader,
        "status",
        lambda: {
            "ready": ready,
            "path": "/models/bge-m3" if ready else None,
            "fingerprint": fingerprint,
            "error": None if ready else "embedder construction failed",
            "source": "local" if ready else None,
            "degraded": degraded,
        },
    )


def test_semantic_search_fails_fast_without_legacy_retry(tmp_corpus_root, monkeypatch):
    """A wedged embedding backend must end the query on the first attempt.

    Regression guard for the 2026-11-25 incident: after an
    EmbeddingUnavailable from the vector path, the legacy FAISS fallback
    used to embed the query again, doubling the query timeout (2 x 20 s)
    and blowing the 30 s MCP tool budget.
    """
    import doc_rag.raglib.vector_index as vector_index
    from doc_rag.raglib import model_loader
    from doc_rag.raglib.ollama_embedder import EmbeddingUnavailable
    from doc_rag.server import retrieval

    cfg = _config_with_no_index(tmp_corpus_root)
    # Make the legacy branch reachable: without a faiss.index file it would
    # bail out before loading the embedder and the guard would be vacuous.
    (tmp_corpus_root / "build" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_corpus_root / "build" / "index" / "faiss.index").write_bytes(b"stub")

    class _WedgedVectorIndex:
        def __init__(self, *args, **kwargs):
            pass

        def is_available(self):
            return True

        def search(self, query, top_k):
            raise EmbeddingUnavailable("Ollama embed timed out after 20s")

        def search_with_filter(self, query, top_k, filters, chunk_meta=None):
            raise EmbeddingUnavailable("Ollama embed timed out after 20s")

    monkeypatch.setattr(vector_index, "VectorIndex", _WedgedVectorIndex)
    monkeypatch.setattr(
        model_loader,
        "load_embedder",
        lambda cfg: (_ for _ in ()).throw(
            AssertionError("legacy path embedded the query a second time")
        ),
    )
    retrieval.invalidate_index_cache()

    assert retrieval.semantic_search(cfg, [], "query", 2) is None


def test_doc_search_tool_empty_query_short_circuits(tmp_corpus_root):
    """Empty query must not even reach the search path."""
    from doc_rag.server.search_tool import doc_search_tool

    _config_with_no_index(tmp_corpus_root)

    content = doc_search_tool({"query": "   "})

    assert len(content) == 1
    assert "Empty query" in content[0]["text"]


def test_doc_search_tool_no_notice_when_semantic_disabled(tmp_corpus_root, monkeypatch):
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
        json.dumps(
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "text": "no marker here",
                "source_file": "doc-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    content = doc_search_tool({"query": "marker", "top_k": 3})

    assert content
    # No degraded notice as the first item.
    assert content[0]["text"] != _FALLBACK_NOTICE
