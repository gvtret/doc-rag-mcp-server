"""Weights-fingerprint in index metadata and revision-mixing guard (BUG-006).

Covers:
- IndexMeta roundtrip keeps weights_sha256; legacy JSON without the field
  loads as an empty string.
- VectorIndex.build refuses incremental updates when the index fingerprint
  differs from the current model weights (full rebuild instead).
- VectorIndex.build refuses incremental updates on a legacy index that has
  no fingerprint at all (WARN + full rebuild).
- Same guard for the chunking/text config fingerprint (config_fingerprint):
  mismatch -> full rebuild; legacy meta without the field -> WARN + rebuild.
- chunking_meta.config_fingerprint(): stable over identical inputs, changes
  when the tokenizer.json content or any chunking parameter changes, and
  tolerates a missing tokenizer (hash excluded, fp still computed).
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
CFG_FP_PATCH_TARGET = "doc_rag.raglib.vector_index.config_fingerprint"


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


def _patch_cfg_fingerprint(monkeypatch, value: str | None) -> None:
    monkeypatch.setattr(CFG_FP_PATCH_TARGET, lambda cfg: value)


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
    """init() + save() roundtrip: the fingerprint survives a disk reload."""
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "index_meta.json"
    store = FaissVectorStore(str(index_path), str(meta_path), metric="ip")

    import faiss  # noqa: F401  (init imports faiss itself; skip if missing)

    store.init(dim=8, model_name="m", weights_sha256="deadbeef")
    store.save()

    reloaded = FaissVectorStore(str(index_path), str(meta_path), metric="ip")
    reloaded._ensure_loaded()
    meta = reloaded.get_meta()
    assert meta is not None and meta.weights_sha256 == "deadbeef"


def test_migrate_stamps_fingerprint_into_target_meta(tmp_path: Path, monkeypatch):
    """Migration re-encodes with current weights: the target index meta
    must carry their fingerprint (review finding 1)."""
    import faiss  # noqa: F401
    import numpy as np

    from doc_rag.raglib.migrate_vectors import migrate_vectors

    root = tmp_path
    (root / "build" / "index").mkdir(parents=True)
    (root / "build" / "chunks_jsonl").mkdir(parents=True)

    chunks = [{"chunk_id": f"doc:000{i}", "doc_id": "doc", "text": f"chunk {i}"} for i in range(3)]
    with (root / "build" / "chunks_jsonl" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    vecs = np.random.RandomState(7).randn(3, 8).astype(np.float32)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    faiss.write_index(index, str(root / "build" / "index" / "faiss.index"))
    meta = {
        "protocol_version": "1.0.0",
        "model_name": "m",
        "metric": "ip",
        "normalize": True,
        "dim": 8,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "weights_sha256": "src-fp",
    }
    (root / "build" / "index" / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    cfg = {
        "_root": str(root),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {
            "index_dir": "build/index",
            "chunks_dir": "build/chunks_jsonl",
        },
        "embeddings": {"model_name": "test-model", "device": "cpu", "normalize": True},
    }
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda c: _StubEmbedder())
    monkeypatch.setattr(
        "doc_rag.raglib.migrate_vectors.model_paths.fingerprint_for_cfg", lambda c: "new-fp"
    )

    stats = migrate_vectors(cfg, source_backend="faiss", target_backend="faiss")
    assert stats["success"] is True
    assert stats["vectors_migrated"] == 3

    stored = json.loads((root / "build" / "index" / "index_meta.json").read_text())
    assert stored["weights_sha256"] == "new-fp"


def test_meta_roundtrip_keeps_config_fingerprint():
    meta = IndexMeta(
        protocol_version="1",
        model_name="BAAI/bge-m3",
        metric="ip",
        normalize=True,
        dim=1024,
        chunk_ids=[],
        weights_sha256="abc",
        config_fingerprint="def",
    )
    d = meta.to_dict()
    assert d["config_fingerprint"] == "def"
    assert IndexMeta.from_dict(d).config_fingerprint == "def"


def test_old_meta_without_config_fingerprint_loads_empty():
    data = {
        "protocol_version": "1",
        "model_name": "BAAI/bge-m3",
        "metric": "ip",
        "normalize": True,
        "dim": 1024,
        "chunk_ids": [],
    }
    assert IndexMeta.from_dict(data).config_fingerprint == ""


def test_init_persists_config_fingerprint_to_meta_json(tmp_path: Path):
    """init() + save() roundtrip: the config fingerprint survives a reload."""
    index_path = tmp_path / "faiss.index"
    meta_path = tmp_path / "index_meta.json"
    store = FaissVectorStore(str(index_path), str(meta_path), metric="ip")

    import faiss  # noqa: F401  (init imports faiss itself; skip if missing)

    store.init(dim=8, model_name="m", config_fingerprint="cafebabe")
    store.save()

    reloaded = FaissVectorStore(str(index_path), str(meta_path), metric="ip")
    reloaded._ensure_loaded()
    meta = reloaded.get_meta()
    assert meta is not None and meta.config_fingerprint == "cafebabe"


def test_migrate_stamps_config_fingerprint_into_target_meta(tmp_path: Path, monkeypatch):
    """Migration re-encodes with current weights: the target index meta must
    also carry the chunking/text config fingerprint."""
    import faiss  # noqa: F401
    import numpy as np

    from doc_rag.raglib.migrate_vectors import migrate_vectors

    root = tmp_path
    (root / "build" / "index").mkdir(parents=True)
    (root / "build" / "chunks_jsonl").mkdir(parents=True)

    chunks = [{"chunk_id": f"doc:000{i}", "doc_id": "doc", "text": f"chunk {i}"} for i in range(3)]
    with (root / "build" / "chunks_jsonl" / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    vecs = np.random.RandomState(7).randn(3, 8).astype(np.float32)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    faiss.write_index(index, str(root / "build" / "index" / "faiss.index"))
    meta = {
        "protocol_version": "1.0.0",
        "model_name": "m",
        "metric": "ip",
        "normalize": True,
        "dim": 8,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "weights_sha256": "src-fp",
    }
    (root / "build" / "index" / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    cfg = {
        "_root": str(root),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {
            "index_dir": "build/index",
            "chunks_dir": "build/chunks_jsonl",
        },
        "embeddings": {"model_name": "test-model", "device": "cpu", "normalize": True},
    }
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda c: _StubEmbedder())
    monkeypatch.setattr(
        "doc_rag.raglib.migrate_vectors.model_paths.fingerprint_for_cfg", lambda c: "new-fp"
    )
    monkeypatch.setattr("doc_rag.raglib.migrate_vectors.config_fingerprint", lambda c: "new-cfg-fp")

    stats = migrate_vectors(cfg, source_backend="faiss", target_backend="faiss")
    assert stats["success"] is True
    assert stats["vectors_migrated"] == 3

    stored = json.loads((root / "build" / "index" / "index_meta.json").read_text())
    assert stored["config_fingerprint"] == "new-cfg-fp"


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
# Config fingerprint (chunking / tokenizer / embedding-text) guard
# ---------------------------------------------------------------------------


def test_incremental_refused_on_config_fingerprint_mismatch(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, "w")
    _patch_cfg_fingerprint(monkeypatch, "abc")

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)
    assert idx.size == 3
    assert json.loads(_meta_path(tmp_path).read_text())["config_fingerprint"] == "abc"

    # The chunking config changed: same chunk ids, but the fingerprint now
    # differs -> must fully rebuild, not treat everything as "up to date".
    _patch_cfg_fingerprint(monkeypatch, "def")
    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    idx2.build(_make_chunks(3), log=logs2.append)

    joined = "\n".join(logs2)
    assert "forcing full rebuild" in joined
    assert "Incremental update" not in joined
    assert json.loads(_meta_path(tmp_path).read_text())["config_fingerprint"] == "def"
    assert idx2.size == 3


def test_incremental_refused_on_legacy_meta_without_config_fingerprint(
    tmp_path, monkeypatch, caplog
):
    """Index built before config fingerprints existed: the field is absent
    on disk -> the first build after the upgrade must fully rebuild once."""
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, "w")
    # First build runs with the fingerprint patched out (legacy meta).
    _patch_cfg_fingerprint(monkeypatch, None)

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)
    assert idx.size == 3
    assert json.loads(_meta_path(tmp_path).read_text())["config_fingerprint"] == ""

    # First ingest after the upgrade: the fingerprint becomes known, the
    # on-disk index has none -> WARN + full rebuild (no silent mixing).
    _patch_cfg_fingerprint(monkeypatch, "def")
    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    with caplog.at_level(logging.WARNING, logger="doc_rag.raglib.vector_index"):
        idx2.build(_make_chunks(3), log=logs2.append)

    assert any("index predates config fingerprint" in rec.message for rec in caplog.records)
    joined = "\n".join(logs2)
    assert "forcing full rebuild" in joined
    assert "Incremental update" not in joined
    assert idx2.size == 3
    assert json.loads(_meta_path(tmp_path).read_text())["config_fingerprint"] == "def"


def test_matching_config_fingerprint_keeps_incremental(tmp_path, monkeypatch):
    """Guard: identical config fingerprints must not defeat normal adds."""
    cfg = _make_cfg(tmp_path)
    _patch_embedder(monkeypatch)
    _patch_fingerprint(monkeypatch, "w")
    _patch_cfg_fingerprint(monkeypatch, "same")

    idx = VectorIndex(cfg)
    logs: list[str] = []
    idx.build(_make_chunks(3), log=logs.append)

    idx2 = VectorIndex(cfg)
    logs2: list[str] = []
    idx2.build(_make_chunks(5), log=logs2.append)

    assert "Incremental update" in "\n".join(logs2)
    assert idx2.size == 5


# ---------------------------------------------------------------------------
# chunking_meta.config_fingerprint
# ---------------------------------------------------------------------------


def _write_tokenizer(tmp_path: Path, payload: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tok = tmp_path / "tokenizer.json"
    tok.write_text(payload, encoding="utf-8")
    return tok


def test_config_fingerprint_stable_and_field_sensitive(tmp_path):
    from doc_rag.raglib.chunking_meta import config_fingerprint as fp

    tok = _write_tokenizer(tmp_path, '{"model": "bge"}')
    cfg = {"chunking": {"strategy": "recursive", "target_tokens": 512, "overlap_tokens": 64}}

    a = fp(cfg, tok)
    b = fp(cfg, tok)
    assert a == b
    assert len(a) == 64

    # Any chunking parameter change must change the fingerprint.
    changed = dict(cfg, chunking={"strategy": "fixed", "target_tokens": 512, "overlap_tokens": 64})
    assert fp(changed, tok) != a
    changed = dict(
        cfg, chunking={"strategy": "recursive", "target_tokens": 256, "overlap_tokens": 64}
    )
    assert fp(changed, tok) != a
    changed = dict(
        cfg, chunking={"strategy": "recursive", "target_tokens": 512, "overlap_tokens": 32}
    )
    assert fp(changed, tok) != a


def test_config_fingerprint_changes_on_tokenizer_change(tmp_path):
    from doc_rag.raglib.chunking_meta import config_fingerprint as fp

    tok = _write_tokenizer(tmp_path, '{"model": "v1"}')
    cfg = {"chunking": {"strategy": "recursive", "target_tokens": 512, "overlap_tokens": 64}}
    fp1 = fp(cfg, tok)

    tok.write_text('{"model": "v2"}', encoding="utf-8")
    fp2 = fp(cfg, tok)
    assert fp1 != fp2


def test_config_fingerprint_without_tokenizer_still_computes(tmp_path, caplog):
    """Missing tokenizer.json: its hash is excluded, the fingerprint is
    still deterministic (and changes when the tokenizer appears)."""
    from doc_rag.raglib.chunking_meta import config_fingerprint as fp

    cfg = {"chunking": {"strategy": "recursive", "target_tokens": 512, "overlap_tokens": 64}}
    missing = tmp_path / "nope" / "tokenizer.json"

    with caplog.at_level(logging.WARNING, logger="doc_rag.raglib.chunking_meta"):
        import doc_rag.raglib.chunking_meta as cm

        # Reset the once-latch: an earlier test in this module may have
        # already triggered the WARN for a missing tokenizer.
        cm._warned_no_tokenizer = False
        fp1 = fp(cfg, missing)
        fp2 = fp(cfg, missing)
    assert fp1 == fp2
    assert len(fp1) == 64
    assert any("tokenizer" in rec.message for rec in caplog.records)
    # The WARN is logged once, not on every call.
    tok_warnings = [r for r in caplog.records if "tokenizer.json not found" in r.message]
    assert len(tok_warnings) == 1

    tok = _write_tokenizer(tmp_path / "nope", '{"model": "bge"}')
    assert fp(cfg, tok) != fp1


def test_config_fingerprint_bad_chunking_values_fall_back(caplog):
    """Non-int target_tokens/overlap_tokens: fallback 512/64 with a WARN
    (logged once), no TypeError."""
    from doc_rag.raglib.chunking_meta import config_fingerprint as fp

    good = {"chunking": {"strategy": "recursive", "target_tokens": 512, "overlap_tokens": 64}}
    bad = {"chunking": {"strategy": "recursive", "target_tokens": "many", "overlap_tokens": None}}

    with caplog.at_level(logging.WARNING, logger="doc_rag.raglib.chunking_meta"):
        import doc_rag.raglib.chunking_meta as cm

        cm._warned_bad_chunking_values = False
        fp_bad = fp(bad)
        fp_bad2 = fp(bad)
    fp_good = fp(good)
    assert fp_bad == fp_good
    assert fp_bad == fp_bad2
    fallback_warnings = [r for r in caplog.records if "falling back to defaults" in r.message]
    assert len(fallback_warnings) == 1


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
