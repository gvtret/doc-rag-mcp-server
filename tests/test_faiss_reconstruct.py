"""Regression test for the FAISS reconstruct-on-delete path.

The optimisation in `pipeline._rebuild_faiss_after_delete` is the
difference between a 30-second delete and a 3-hour rebuild on a
CPU-only machine. Its correctness invariant: after the prune, the
remaining vectors retrieved by `index.reconstruct(i)` must equal the
originals byte-for-byte (within float32 representation).

This test bypasses `delete_documents` and calls the prune function
directly so that the assertion is about the prune itself, not about
the surrounding manifest/chunks bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _faiss_or_skip():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        pytest.skip("faiss not installed")


def _build_index(root: Path, doc_chunks, vectors):
    """Persist a FAISS index plus matching index_meta.json under `root`."""
    faiss = _faiss_or_skip()

    index_dir = root / "build" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    faiss.write_index(idx, str(index_dir / "faiss.index"))

    chunk_ids = []
    for doc_id, n in doc_chunks:
        for i in range(n):
            chunk_ids.append(f"{doc_id}:{i:04d}")

    (index_dir / "index_meta.json").write_text(
        json.dumps({"chunk_ids": chunk_ids, "dim": int(dim), "metric": "ip"}),
        encoding="utf-8",
    )
    return chunk_ids


def test_reconstruct_preserves_kept_vectors_byte_for_byte(tmp_corpus_root, synthetic_embeddings):
    """The headline invariant: prune drops vectors, doesn't mutate the rest."""
    import numpy as np

    from doc_rag.raglib.pipeline import _rebuild_faiss_after_delete, load_config

    # Three documents with sizes 4, 3, 2 — total 9 vectors.
    doc_chunks = [("doc-a", 4), ("doc-b", 3), ("doc-c", 2)]
    total = sum(n for _, n in doc_chunks)
    vectors = synthetic_embeddings(n=total, dim=16)

    chunk_ids = _build_index(tmp_corpus_root, doc_chunks, vectors)
    # Map chunk_id -> original vector (immutable expectation).
    original = {cid: vectors[i].copy() for i, cid in enumerate(chunk_ids)}

    cfg = load_config(str(tmp_corpus_root / "config" / "config.yaml"))
    stats = _rebuild_faiss_after_delete(cfg, {"doc-b"})

    assert stats["had_index"] is True
    assert stats["removed_vectors"] == 3
    assert stats["kept_vectors"] == total - 3 == 6

    # Read the new index back and check every kept vector is byte-identical.
    faiss = _faiss_or_skip()
    new_idx = faiss.read_index(str(tmp_corpus_root / "build" / "index" / "faiss.index"))
    new_meta = json.loads((tmp_corpus_root / "build" / "index" / "index_meta.json").read_text())

    assert new_meta["dim"] == 16
    assert new_meta["metric"] == "ip"
    assert len(new_meta["chunk_ids"]) == 6
    assert all(not cid.startswith("doc-b:") for cid in new_meta["chunk_ids"])

    for i, cid in enumerate(new_meta["chunk_ids"]):
        kept = new_idx.reconstruct(i)
        assert np.array_equal(kept, original[cid]), f"vector for {cid} drifted during prune"


def test_reconstruct_deletes_all_when_every_doc_targeted(tmp_corpus_root, synthetic_embeddings):
    """When the prune empties the index, the on-disk files are removed."""
    from doc_rag.raglib.pipeline import _rebuild_faiss_after_delete, load_config

    doc_chunks = [("doc-x", 2), ("doc-y", 3)]
    vectors = synthetic_embeddings(n=5, dim=8)
    _build_index(tmp_corpus_root, doc_chunks, vectors)

    cfg = load_config(str(tmp_corpus_root / "config" / "config.yaml"))
    stats = _rebuild_faiss_after_delete(cfg, {"doc-x", "doc-y"})

    assert stats["had_index"] is True
    assert stats["removed_vectors"] == 5
    assert stats["kept_vectors"] == 0

    index_dir = tmp_corpus_root / "build" / "index"
    assert not (index_dir / "faiss.index").exists()
    assert not (index_dir / "index_meta.json").exists()


def test_reconstruct_noop_when_no_deletion_target_matches(tmp_corpus_root, synthetic_embeddings):
    """If the target set has nothing in the index, nothing must move."""
    import numpy as np

    from doc_rag.raglib.pipeline import _rebuild_faiss_after_delete, load_config

    doc_chunks = [("doc-a", 3), ("doc-b", 2)]
    vectors = synthetic_embeddings(n=5, dim=8)
    chunk_ids = _build_index(tmp_corpus_root, doc_chunks, vectors)
    original = vectors.copy()

    cfg = load_config(str(tmp_corpus_root / "config" / "config.yaml"))
    stats = _rebuild_faiss_after_delete(cfg, {"doc-not-in-index"})

    assert stats["removed_vectors"] == 0
    assert stats["kept_vectors"] == len(chunk_ids)
    assert stats["had_index"] is True

    # Index must still match the originals.
    faiss = _faiss_or_skip()
    new_idx = faiss.read_index(str(tmp_corpus_root / "build" / "index" / "faiss.index"))
    for i in range(len(chunk_ids)):
        assert np.array_equal(new_idx.reconstruct(i), original[i])


def test_reconstruct_silent_when_index_missing(tmp_corpus_root):
    """No index → returns stats with had_index=False and does not raise."""
    from doc_rag.raglib.pipeline import _rebuild_faiss_after_delete, load_config

    cfg = load_config(str(tmp_corpus_root / "config" / "config.yaml"))
    stats = _rebuild_faiss_after_delete(cfg, {"anything"})

    assert stats == {"removed_vectors": 0, "kept_vectors": 0, "had_index": False}
