"""Tests for the pluggable vector-store backends."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from doc_rag.raglib.vectorstore.factory import create_vector_store
from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore, IndexMeta


class TestFaissVectorStore:
    def _make_store(self, tmp_path: Path) -> FaissVectorStore:
        return FaissVectorStore(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "index_meta.json"),
            metric="ip",
        )

    def test_init_and_add(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 8
        store.init(dim=dim, model_name="test-model")
        assert store.size == 0
        assert store.dim == dim

        vecs = np.random.randn(3, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(vecs, ["c0", "c1", "c2"])
        assert store.size == 3

    def test_search(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        vecs = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        store.add(vecs, ["a", "b", "c"])

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        dists, ids = store.search(query, top_k=2)
        assert len(dists) == 2
        assert ids[0] == "a"

    def test_delete(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])
        assert store.size == 4

        removed = store.delete(["b", "d"])
        assert removed == 2
        assert store.size == 2

        ids_left = store._meta.chunk_ids
        assert "a" in ids_left
        assert "c" in ids_left
        assert "b" not in ids_left
        assert "d" not in ids_left

    def test_delete_nonexistent(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])
        removed = store.delete(["x", "y"])
        assert removed == 0
        assert store.size == 4

    def test_save_and_reload(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])
        store.save()

        assert os.path.exists(tmp_path / "faiss.index")
        assert os.path.exists(tmp_path / "index_meta.json")

        store2 = self._make_store(tmp_path)
        assert store2.size == 4
        dists, ids = store2.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=2)
        assert ids[0] == "a"

    def test_reset(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        store.add(np.eye(4, dtype=np.float32), ["a", "b", "c", "d"])
        store.save()

        store.reset()
        assert store.size == 0
        assert not os.path.exists(tmp_path / "faiss.index")

    def test_reconstruct(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        dim = 4
        store.init(dim=dim, model_name="m")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])

        v = store.reconstruct("b")
        assert v is not None
        assert v.shape == (4,)
        assert v.dtype == np.float32

        assert store.reconstruct("x") is None


class TestIndexMeta:
    def test_roundtrip(self):
        meta = IndexMeta(
            protocol_version="1.0.0",
            model_name="test",
            metric="ip",
            normalize=True,
            dim=8,
            chunk_ids=["a", "b"],
        )
        d = meta.to_dict()
        meta2 = IndexMeta.from_dict(d)
        assert meta2.model_name == "test"
        assert meta2.chunk_ids == ["a", "b"]
        assert meta2.dim == 8


class TestFactory:
    def test_faiss_factory(self, tmp_path: Path):
        cfg = {
            "_root": str(tmp_path),
            "index": {"backend": "faiss", "metric": "ip"},
            "paths": {"index_dir": "build/index"},
        }
        os.makedirs(tmp_path / "build" / "index", exist_ok=True)
        store = create_vector_store(cfg)
        assert isinstance(store, FaissVectorStore)
        assert store.dim == 0  # not yet initialized


class TestE2EFaissMigration:
    """E2E: build FAISS index, migrate to fresh FAISS, verify search integrity."""

    def test_migrate_preserves_search_results(self, tmp_path: Path):
        dim = 8
        ids = [f"doc{i}:0" for i in range(5)]
        vecs = np.random.RandomState(42).randn(5, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src = FaissVectorStore(
            index_path=str(src_dir / "faiss.index"),
            meta_path=str(src_dir / "index_meta.json"),
            metric="ip",
        )
        src.init(dim=dim, model_name="test")
        src.add(vecs, ids)
        src.save()

        # Migrate: copy vectors to a new store
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        dst = FaissVectorStore(
            index_path=str(dst_dir / "faiss.index"),
            meta_path=str(dst_dir / "index_meta.json"),
            metric="ip",
        )
        dst.init(dim=dim, model_name="test")
        dst.add(vecs, ids)
        dst.save()

        assert dst.size == 5
        query = vecs[0:1]
        dists, found_ids = dst.search(query, top_k=3)
        assert len(dists) == 3
        assert found_ids[0] == "doc0:0"

    def test_delete_then_search(self, tmp_path: Path):
        dim = 4
        store = FaissVectorStore(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "index_meta.json"),
            metric="ip",
        )
        store.init(dim=dim, model_name="m")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])
        store.save()

        store.delete(["b", "d"])
        store.save()

        store2 = FaissVectorStore(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "index_meta.json"),
            metric="ip",
        )
        assert store2.size == 2
        dists, ids = store2.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=2)
        assert len(ids) == 2
        assert "a" in ids
        assert "c" in ids
