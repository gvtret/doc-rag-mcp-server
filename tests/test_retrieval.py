from __future__ import annotations

import json

import pytest

from doc_rag.server import retrieval
from doc_rag.server.retrieval import (
    annotation_from_markdown,
    document_preview,
    indexed_catalog,
    lexical_search,
    semantic_search,
)


def test_lexical_search_finds_match_and_scores() -> None:
    chunks = [
        {"doc_id": "a", "chunk_id": "a:0", "text": "hello world"},
        {"doc_id": "b", "chunk_id": "b:0", "text": "something else"},
        {"doc_id": "c", "chunk_id": "c:0", "text": "world world world"},
    ]
    out = lexical_search(chunks, query="world", top_k=2)
    assert len(out) == 2
    assert out[0]["doc_id"] in {"a", "c"}
    assert isinstance(out[0].get("score"), float)


def test_indexed_catalog_keys() -> None:
    c = indexed_catalog()
    assert "documents" in c
    assert "document_count" in c
    assert isinstance(c["documents"], list)
    assert "lexical_search_ready" in c
    assert "semantic_search_ready" in c


def test_annotation_from_markdown_uses_first_heading() -> None:
    md = "# My Title\n\nFirst paragraph here.\n\nMore text."
    title, preview = annotation_from_markdown(md)
    assert title == "My Title"
    assert "First paragraph" in preview


def test_annotation_from_markdown_empty() -> None:
    title, preview = annotation_from_markdown("")
    assert title == ""
    assert "Пустой" in preview


def test_document_preview_rejects_empty_doc_id() -> None:
    out = document_preview("")
    assert out.get("ok") is False
    assert "empty" in str(out.get("error", "")).lower()


def test_load_config_rejects_invalid_index_backend(tmp_path, monkeypatch) -> None:
    """load_config raises ValueError for invalid index.backend."""
    import yaml

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"index": {"backend": "mongodb"}}))

    monkeypatch.setattr("doc_rag.server.retrieval.project_root", lambda: str(tmp_path))
    # Clear cache
    monkeypatch.setattr("doc_rag.server.retrieval._CFG_CACHE", {})

    import pytest as _pytest

    with _pytest.raises(ValueError, match="Invalid index.backend"):
        from doc_rag.server.retrieval import load_config as _lc

        _lc()


def test_load_config_rejects_invalid_retrieval_mode(tmp_path, monkeypatch) -> None:
    """load_config raises ValueError for invalid mcp.retrieval_mode."""
    import yaml

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"mcp": {"retrieval_mode": "vector_only"}}))

    monkeypatch.setattr("doc_rag.server.retrieval.project_root", lambda: str(tmp_path))
    monkeypatch.setattr("doc_rag.server.retrieval._CFG_CACHE", {})

    import pytest as _pytest

    with _pytest.raises(ValueError, match="Invalid mcp.retrieval_mode"):
        from doc_rag.server.retrieval import load_config as _lc

        _lc()


def test_legacy_failing_embedder_does_not_reread_index(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy FAISS path: a persistently failing embedder must not cause
    faiss.read_index on every request (regression for review finding 1)."""
    faiss = pytest.importorskip("faiss")
    import numpy as np

    # Minimal legacy layout: chunks.jsonl + faiss.index under tmp root.
    root = tmp_path
    chunks = [{"chunk_id": "d:0", "doc_id": "d", "text": "hello world", "source_file": "d.pdf"}]
    chunks_dir = root / "build" / "chunks_jsonl"
    chunks_dir.mkdir(parents=True)
    chunks_path = chunks_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(chunks[0], ensure_ascii=False) + "\n")

    vec = np.ones((1, 8), dtype="float32")
    index = faiss.IndexFlatIP(8)
    index.add(vec)
    index_dir = root / "build" / "index"
    index_dir.mkdir(parents=True)
    index_file = index_dir / "faiss.index"
    faiss.write_index(index, str(index_file))

    cfg = {
        "_root": str(root),
        "paths": {"chunks_dir": "build/chunks_jsonl", "index_dir": "build/index"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }

    # The VectorIndex (pluggable-backend) path must fall through to the
    # legacy branch: make its constructor raise so the outer try in
    # semantic_search passes to the fallback. Patching is_available=False
    # is NOT enough - semantic_search returns None at that point without
    # reaching the legacy code under test.
    monkeypatch.setattr(
        "doc_rag.raglib.vector_index.VectorIndex.__init__",
        lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("no vector backend")),
    )
    # Embedder load fails persistently. The legacy branch does
    # `from doc_rag.raglib.model_loader import load_embedder` inside the
    # function, so patch the source module attribute.
    monkeypatch.setattr("doc_rag.raglib.model_loader.load_embedder", lambda cfg: None)

    reads = []
    real_read_index = faiss.read_index

    def counting_read_index(path):
        reads.append(path)
        return real_read_index(path)

    monkeypatch.setattr(faiss, "read_index", counting_read_index)

    retrieval._SEMANTIC_CACHE["index"] = None
    retrieval._SEMANTIC_CACHE["index_path"] = None
    retrieval._SEMANTIC_CACHE["index_mtime"] = None
    retrieval._SEMANTIC_CACHE["model_key"] = None

    for _ in range(3):
        out = semantic_search(cfg, chunks, "hello", top_k=2)
        assert out is None  # embedder unavailable -> degrade, no results

    assert reads == []  # zero faiss.read_index calls across 3 requests


def test_legacy_working_embedder_reads_index_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion check: with a working embedder the legacy path returns
    results and reads the index only on the first request."""
    faiss = pytest.importorskip("faiss")
    import numpy as np

    root = tmp_path
    chunks = [{"chunk_id": "d:0", "doc_id": "d", "text": "hello world", "source_file": "d.pdf"}]
    chunks_dir = root / "build" / "chunks_jsonl"
    chunks_dir.mkdir(parents=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(chunks[0], ensure_ascii=False) + "\n")

    vec = np.ones((1, 8), dtype="float32")
    index = faiss.IndexFlatIP(8)
    index.add(vec)
    index_dir = root / "build" / "index"
    index_dir.mkdir(parents=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    cfg = {
        "_root": str(root),
        "paths": {"chunks_dir": "build/chunks_jsonl", "index_dir": "build/index"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.ones((1, 8), dtype="float32")

    monkeypatch.setattr(
        "doc_rag.raglib.vector_index.VectorIndex.__init__",
        lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("no vector backend")),
    )
    monkeypatch.setattr("doc_rag.raglib.model_loader.load_embedder", lambda cfg: FakeModel())

    reads = []
    real_read_index = faiss.read_index

    def counting_read_index(path):
        reads.append(path)
        return real_read_index(path)

    monkeypatch.setattr(faiss, "read_index", counting_read_index)

    retrieval._SEMANTIC_CACHE["index"] = None
    retrieval._SEMANTIC_CACHE["index_path"] = None
    retrieval._SEMANTIC_CACHE["index_mtime"] = None
    retrieval._SEMANTIC_CACHE["model_key"] = None

    for _ in range(3):
        out = semantic_search(cfg, chunks, "hello", top_k=2)
        assert out is not None and len(out) == 1

    assert reads == [str(index_dir / "faiss.index")]  # read exactly once
