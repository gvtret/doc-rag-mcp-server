"""Tests for the high-level VectorIndex class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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

    def test_encode_retries_with_shrinking_batch_size(self, tmp_path: Path, monkeypatch):
        """A failure (e.g. OOM) should not be retried with the same batch
        size — that just reproduces the same failure. Regression for a
        retry loop that kept `batch_size` constant across all attempts.

        Encoding now runs in explicit batches, so each attempt issues several
        encode() calls; we assert the batch size strictly shrinks per attempt.
        """
        cfg = _make_cfg(tmp_path)  # batch_size: 4
        idx = VectorIndex(cfg)

        seen_batch_sizes: list[int] = []
        attempts = {"n": 0}

        class _FlakyEmbedder:
            def encode(
                self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
            ):
                seen_batch_sizes.append(batch_size)
                # Fail the whole first attempt; succeed afterwards.
                if attempts["n"] == 0:
                    raise RuntimeError("CUDA out of memory")
                return np.ones((len(texts), 8), dtype=np.float32)

        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _FlakyEmbedder()
        )

        def _fake_sleep(*_):
            attempts["n"] += 1  # advance past the failing attempt

        monkeypatch.setattr("time.sleep", _fake_sleep)

        vecs = idx._encode(["a", "b"])

        # first attempt at batch 4, retry at batch 2 — strictly smaller
        assert seen_batch_sizes[0] == 4
        assert min(seen_batch_sizes) < 4
        assert vecs.shape == (2, 8)

    def test_encode_raises_after_exhausting_retries(self, tmp_path: Path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)

        class _AlwaysFailEmbedder:
            def encode(self, *a, **kw):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _AlwaysFailEmbedder()
        )
        monkeypatch.setattr("time.sleep", lambda *_: None)

        with pytest.raises(RuntimeError, match="Embedding failed after 3 attempts"):
            idx._encode(["a"])


class TestModelChangeForcesRebuild:
    """BUG-006: an incremental build must not mix vectors from two models."""

    def test_changed_model_name_rebuilds_from_scratch(self, tmp_path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )
        idx.build(_make_chunks(3))
        assert idx.size == 3

        # Same chunks (so incremental would find no *new* ids), but the
        # embedding model changed in config -> must rebuild, not "update nothing".
        import json

        meta_path = tmp_path / "build" / "index" / "index_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["model_name"] == "test-model"

        cfg2 = _make_cfg(tmp_path)
        cfg2["embeddings"]["model_name"] = "other-model"
        idx2 = VectorIndex(cfg2)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )
        idx2.build(_make_chunks(3))

        meta2 = json.loads(meta_path.read_text())
        assert meta2["model_name"] == "other-model"
        assert idx2.size == 3

    def test_changed_model_and_dim_rebuilds_from_scratch(self, tmp_path, monkeypatch):
        """A model swap that also changes dim must rebuild cleanly, not crash."""
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder(dim=8)
        )
        idx.build(_make_chunks(3))

        cfg2 = _make_cfg(tmp_path)
        cfg2["embeddings"]["model_name"] = "other-model"  # new model -> full rebuild
        idx2 = VectorIndex(cfg2)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder(dim=16)
        )
        idx2.build(_make_chunks(3))
        assert idx2.size == 3
        assert idx2.dim == 16

    def test_same_model_still_incremental(self, tmp_path, monkeypatch):
        """Guard: the rebuild trigger must not defeat normal incremental adds."""
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )
        idx.build(_make_chunks(3))

        idx2 = VectorIndex(cfg)  # same config, fresh instance
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )
        idx2.build(_make_chunks(5))
        assert idx2.size == 5


class TestProgressCallback:
    def test_build_emits_embed_and_done_progress(self, tmp_path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )
        events = []
        idx.set_progress_callback(lambda phase, done, total: events.append((phase, done, total)))

        idx.build(_make_chunks(5))

        assert ("embed", 0, 5) in events
        assert ("embed", 5, 5) in events
        assert events[-1] == ("done", 5, 5)

    def test_progress_callback_exception_is_swallowed(self, tmp_path, monkeypatch):
        cfg = _make_cfg(tmp_path)
        idx = VectorIndex(cfg)
        monkeypatch.setattr(
            "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder()
        )

        def boom(phase, done, total):
            raise RuntimeError("ui blew up")

        idx.set_progress_callback(boom)
        assert idx.build(_make_chunks(3)) is True  # must not propagate
