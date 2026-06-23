"""Tests for the high-level VectorIndex class."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from doc_rag.raglib.vector_index import VectorIndex


def _make_cfg(tmp_path: Path) -> dict:
    return {
        "_root": str(tmp_path),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {"index_dir": "build/index", "chunks_dir": "build/chunks_jsonl"},
        "embeddings": {
            "model_name": "test-model",
            "device": "cpu",
            "batch_size": 4,
            "normalize": True,
        },
        "pipeline_version": "1.0.0",
    }


def _make_chunks(n: int = 5) -> list[dict]:
    return [
        {"doc_id": f"doc{i}", "chunk_id": f"doc{i}:0", "text": f"chunk text {i}"} for i in range(n)
    ]


class _StubEmbedder:
    """Deterministic embedder that doesn't require torch."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        n = len(texts)
        vecs = np.random.RandomState(42).randn(n, self._dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
        return vecs


class TestVectorIndex:
    def test_build_and_search(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        chunks = _make_chunks(5)
        ok = idx.build(chunks)
        assert ok is True
        assert idx.size == 5

        dists, ids = idx.search("query", top_k=3)
        assert len(dists) == 3
        assert all(isinstance(cid, str) for cid in ids)

    def test_incremental_update(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        idx.build(_make_chunks(3))
        assert idx.size == 3

        more = [
            {"doc_id": "doc3", "chunk_id": "doc3:0", "text": "new chunk"},
            {"doc_id": "doc4", "chunk_id": "doc4:0", "text": "another chunk"},
        ]
        idx.build(more)
        assert idx.size == 5

    def test_delete(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        idx.build(_make_chunks(5))
        removed = idx.delete({"doc2"})
        assert removed == 1
        assert idx.size == 4

    def test_empty_chunks(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        ok = idx.build([])
        assert ok is False
        assert idx.size == 0

    def test_no_chunk_ids(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        ok = idx.build([{"text": "no id here"}])
        assert ok is False

    def test_reset(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        idx.build(_make_chunks(3))
        assert idx.size == 3
        idx.reset()
        assert idx.size == 0

    def test_search_empty_index(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        dists, ids = idx.search("query", top_k=5)
        assert len(dists) == 0
        assert len(ids) == 0
