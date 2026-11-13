"""Weights-fingerprint in index metadata and revision-mixing guard (BUG-006).

Covers:
- IndexMeta roundtrip keeps weights_sha256; legacy JSON without the field
  loads as an empty string.
- VectorIndex.build refuses incremental updates when the index fingerprint
  differs from the current model weights (full rebuild instead).
- VectorIndex.build refuses incremental updates on a legacy index that has
  no fingerprint at all (WARN + full rebuild).
- indexed_catalog() reports model_ready / weights_sha256 / index_stale /
  stale_reason / model_update_available by reading index_meta.json directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from doc_rag.raglib.vector_index import VectorIndex
from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore, IndexMeta

FP_PATCH_TARGET = "doc_rag.raglib.model_paths.fingerprint_for_cfg"


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
    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vecs = np.random.RandomState(42).randn(len(texts), self._dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
        return vecs


def _patch_fingerprint(monkeypatch, value: str | None) -> None:
    monkeypatch.setattr(FP_PATCH_TARGET, lambda cfg: value)


def _patch_embedder(monkeypatch) -> None:
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder())


def _meta_path(tmp_path: Path) -> Path:
    return tmp_path / "build" / "index" / "index_meta.json"


# ---------------------------------------------------------------------------
# IndexMeta roundtrip
# ---------------------------------------------------------------------------


def test_meta_roundtrip_keeps_fingerprint():
    meta = IndexMeta(
        protocol_version="1",
        model_name="BAAI/bge-m3",
        metric="ip",
        normalize=True,
        dim=1024,
        chunk_ids=[],
        weights_sha256="abc",
    )
    assert IndexMeta.from_dict(meta.to_dict()).weights_sha256 == "abc"


def test_old_meta_without_field_loads_empty():
    data = {
        "protocol_version": "1",
        "model_name": "BAAI/bge-m3",
        "metric": "ip",
        "normalize": True,
        "dim": 1024,
        "chunk_ids": [],
    }
    assert IndexMeta.from_dict(data).weights_sha256 == ""


def test_init_persists_fingerprint_to_meta_json(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "index_meta.json"
    store = FaissVectorStore(str(index_path), str(meta_path), metric="ip")

    import faiss  # noqa: F401  (init imports faiss itself; skip if missing)

    store.init(dim=8, model_name="m", weights_sha256="deadbeef")
    meta = store.get_meta()
    assert meta is not None and meta.weights_sha256 == "deadbeef"


# ---------------------------------------------------------------------------
# Incremental update refused on fingerprint mismatch
# ---------------------------------------------------------------------------


def test_incremental_refused_on_fingerprint_mismatch(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, "old")

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)
    assert idx.size == 3
    assert json.loads(_meta_path(tmp_path).read_text())["weights_sha256"] == "old"

    # The model revision changed: same chunk ids, but the fingerprints now
    # differ -> must fully rebuild, not treat everything as "up to date".
    _patch_fingerprint(monkeypatch, "new")
    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    idx2.build(_make_chunks(3), log=logs2.append)

    joined = "\n".join(logs2)
    assert "forcing full rebuild" in joined
    assert "Incremental update" not in joined
    assert json.loads(_meta_path(tmp_path).read_text())["weights_sha256"] == "new"
    assert idx2.size == 3


# ---------------------------------------------------------------------------
# Incremental update refused on legacy index without a fingerprint
# ---------------------------------------------------------------------------


def test_incremental_refused_on_legacy_meta(tmp_path, monkeypatch, caplog):
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, None)

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)
    assert idx.size == 3
    # Built before fingerprints existed: meta has no fingerprint.
    assert json.loads(_meta_path(tmp_path).read_text())["weights_sha256"] == ""

    # First ingest after the upgrade: the fingerprint becomes known, the
    # on-disk index has none -> WARN + full rebuild (no silent mixing).
    _patch_fingerprint(monkeypatch, "new")
    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    with caplog.at_level(logging.WARNING, logger="doc_rag.raglib.vector_index"):
        idx2.build(_make_chunks(3), log=logs2.append)

    assert any("index has no weights fingerprint" in rec.message for rec in caplog.records)
    joined = "\n".join(logs2)
    assert "forcing full rebuild" in joined
    assert "Incremental update" not in joined
    assert idx2.size == 3
    assert json.loads(_meta_path(tmp_path).read_text())["weights_sha256"] == "new"


def test_matching_fingerprint_keeps_incremental(tmp_path, monkeypatch):
    """Guard: identical fingerprints must not defeat normal incremental adds."""
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, "same")

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)

    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    idx2.build(_make_chunks(5), log=logs2.append)

    assert "Incremental update" in "\n".join(logs2)
    assert idx2.size == 5


# ---------------------------------------------------------------------------
# indexed_catalog staleness report
# ---------------------------------------------------------------------------


def _write_index_meta(root: Path, weights_sha256: str) -> None:
    meta = {
        "protocol_version": "1",
        "model_name": "test-stub",
        "metric": "ip",
        "normalize": True,
        "dim": 8,
        "chunk_ids": ["doc-aaa:0000"],
        "weights_sha256": weights_sha256,
    }
    (root / "build" / "index" / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _patch_model_status(monkeypatch, ready: bool, fingerprint: str | None) -> None:
    import doc_rag.raglib.model_loader as ml

    monkeypatch.setattr(
        ml,
        "status",
        lambda: {
            "ready": ready,
            "path": "/models/bge-m3",
            "fingerprint": fingerprint,
            "error": None,
            "source": "local",
        },
    )


def test_catalog_marks_stale_on_fingerprint_mismatch(tmp_corpus_root, monkeypatch):
    from doc_rag.server.retrieval import indexed_catalog

    _write_index_meta(tmp_corpus_root, "old")
    _patch_model_status(monkeypatch, ready=True, fingerprint="new")

    cat = indexed_catalog()
    assert cat["index_stale"] is True
    assert cat["stale_reason"] == "weights fingerprint mismatch"
    assert cat["model_ready"] is True
    assert cat["weights_sha256"] == "old"
    assert cat["model_update_available"] is False


def test_catalog_not_stale_when_fingerprints_match(tmp_corpus_root, monkeypatch):
    from doc_rag.server.retrieval import indexed_catalog

    _write_index_meta(tmp_corpus_root, "same")
    _patch_model_status(monkeypatch, ready=True, fingerprint="same")

    cat = indexed_catalog()
    assert cat["index_stale"] is False
    assert cat["stale_reason"] is None


def test_catalog_legacy_index_without_fingerprint_not_stale(tmp_corpus_root, monkeypatch):
    """Old index (no fingerprint) is reported as-is: not 'stale', no reason."""
    from doc_rag.server.retrieval import indexed_catalog

    _write_index_meta(tmp_corpus_root, "")
    _patch_model_status(monkeypatch, ready=True, fingerprint="new")

    cat = indexed_catalog()
    assert cat["index_stale"] is False
    assert cat["stale_reason"] is None
    assert cat["weights_sha256"] is None


def test_catalog_model_not_ready_means_not_stale(tmp_corpus_root, monkeypatch):
    from doc_rag.server.retrieval import indexed_catalog

    _write_index_meta(tmp_corpus_root, "old")
    _patch_model_status(monkeypatch, ready=False, fingerprint=None)

    cat = indexed_catalog()
    assert cat["model_ready"] is False
    assert cat["index_stale"] is False
    assert cat["stale_reason"] is None
