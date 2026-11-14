"""Task 7: metadata filters must actually work in every retrieval mode.

Before the fix, VectorIndex.search_with_filter silently ignored filters on
FAISS (plain top_k was returned), lexical_search had no filters parameter at
all, and doc_search's lexical fallback dropped them. These tests pin the
correct behavior: filtered results only, empty list for filters matching
nothing, and a loud ValueError instead of a silent ignore when metadata is
unavailable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from doc_rag.raglib.vector_index import VectorIndex
from doc_rag.server import retrieval


def _make_cfg(tmp_path) -> dict:
    return {
        "_root": str(tmp_path),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {"index_dir": "build/index", "chunks_dir": "build/chunks_jsonl"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }


class _StubEmbedder:
    """Query -> unit vector along axis int(text); chunk text is a digit str."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros((1, self._dim), dtype="float32")
            v[0, int(t) % self._dim] = 1.0
            out.append(v)
        return np.vstack(out)


def _write_index(tmp_path, chunk_ids: list[str]) -> None:
    """Write faiss.index + index_meta.json: chunk "N" -> unit vector on axis N."""
    import faiss as _faiss

    index_dir = tmp_path / "build" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    dim = max(8, len(chunk_ids))
    vecs = np.zeros((len(chunk_ids), dim), dtype="float32")
    for i, _cid in enumerate(chunk_ids):
        vecs[i, i % dim] = 1.0
    index = _faiss.IndexFlatIP(dim)
    index.add(vecs)
    _faiss.write_index(index, str(index_dir / "faiss.index"))
    meta = {
        "model_name": "stub",
        "dim": dim,
        "metric": "ip",
        "chunk_ids": chunk_ids,
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _chunks_for(chunk_ids: list[str]) -> list[dict]:
    """Chunk dicts keyed by chunk_id: text "N", doc from prefix, alternating
    section paths, every third chunk a table."""
    out = []
    for i, cid in enumerate(chunk_ids):
        doc_id = cid.split(":")[0]
        out.append(
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "text": cid.split(":")[1],
                "section_path": f"{doc_id}/sec{i % 2}",
                "is_table": (i % 3 == 0),
            }
        )
    return out


@pytest.fixture()
def _clean_caches(monkeypatch):
    retrieval._VECTOR_INDEX_CACHE.clear()
    retrieval._SEMANTIC_CACHE.update(
        {"index": None, "index_path": None, "index_mtime": None, "model_key": None}
    )
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubEmbedder())
    yield


# --- (a) semantic: filter doc_id keeps only that doc -----------------------


def test_semantic_filter_doc_id(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ["doc-a:0", "doc-b:0", "doc-c:0"])
    chunks = _chunks_for(["doc-a:0", "doc-b:0", "doc-c:0"])

    out = retrieval.semantic_search(cfg, chunks, "0", top_k=3, filters={"doc_id": "doc-b"})
    assert out is not None
    assert [r["chunk_id"] for r in out] == ["doc-b:0"]

    vi = VectorIndex(cfg)
    dists, ids, _payloads = vi.search_with_filter(
        "0", 3, {"doc_id": "doc-b"}, chunk_meta=_chunk_meta(chunks)
    )
    assert ids == ["doc-b:0"]


def _chunk_meta(chunks: list[dict]) -> dict[str, dict]:
    return {
        c["chunk_id"]: {
            "doc_id": c.get("doc_id", c["chunk_id"].split(":")[0]),
            "section_path": c.get("section_path", ""),
            "is_table": bool(c.get("is_table")),
        }
        for c in chunks
    }


# --- (b) semantic: tables_only ---------------------------------------------


def test_semantic_filter_tables_only(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    ids = ["doc-a:0", "doc-a:1", "doc-a:2", "doc-a:3"]
    _write_index(tmp_path, ids)
    chunks = _chunks_for(ids)  # is_table on positions 0 and 3

    out = retrieval.semantic_search(cfg, chunks, "0", top_k=10, filters={"is_table": True})
    assert out is not None
    got = {r["chunk_id"] for r in out}
    assert got == {"doc-a:0", "doc-a:3"}
    assert all(r["is_table"] for r in out)


# --- (c) semantic: section_path prefix match -------------------------------


def test_semantic_filter_section_path_prefix(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    ids = ["doc-a:0", "doc-a:1", "doc-b:0", "doc-b:1"]
    _write_index(tmp_path, ids)
    chunks = _chunks_for(ids)
    # sections alternate sec0/sec1 per position -> prefix "doc-a/sec0" matches
    # doc-a:0 only (positions: 0 -> doc-a/sec0, 1 -> doc-a/sec1, ...).

    out = retrieval.semantic_search(
        cfg, chunks, "0", top_k=10, filters={"section_path": "doc-a/sec0"}
    )
    assert out is not None
    assert [r["chunk_id"] for r in out] == ["doc-a:0"]

    # Prefix must also match a longer path that starts with the filter.
    out = retrieval.semantic_search(cfg, chunks, "0", top_k=10, filters={"section_path": "doc-a"})
    assert out is not None
    assert {r["chunk_id"] for r in out} == {"doc-a:0", "doc-a:1"}


# --- (d) nonexistent doc_id -> empty result, never unfiltered --------------


def test_semantic_filter_nonexistent_doc_id_empty(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    ids = ["doc-a:0", "doc-b:0"]
    _write_index(tmp_path, ids)
    chunks = _chunks_for(ids)

    out = retrieval.semantic_search(cfg, chunks, "0", top_k=5, filters={"doc_id": "doc-zzz"})
    # Empty filter match must stay empty (no unfiltered fallback), so the
    # tool layer reports "No results." instead of wrong documents.
    assert out == []


# --- (e) lexical and hybrid honor the same filter --------------------------


def test_lexical_filter_doc_id(tmp_path, _clean_caches):
    ids = ["doc-a:0", "doc-b:0", "doc-b:1"]
    chunks = _chunks_for(ids)
    # All texts are digits ("0"/"1"), so lexical matches everything; the
    # filter alone must decide what survives.
    out = retrieval.lexical_search(chunks, query="0", top_k=5, filters={"doc_id": "doc-b"})
    assert out
    assert {r["doc_id"] for r in out} == {"doc-b"}

    out = retrieval.lexical_search(chunks, query="0", top_k=5, filters={"doc_id": "doc-zzz"})
    assert out == []


def test_lexical_filter_tables_and_section(tmp_path, _clean_caches):
    ids = ["doc-a:0", "doc-a:1", "doc-a:2"]
    chunks = _chunks_for(ids)
    # Make every chunk lexically match the query so the filter alone decides.
    for c in chunks:
        c["text"] = f"common needle {c['text']}"
    out = retrieval.lexical_search(chunks, query="needle", top_k=5, filters={"is_table": True})
    assert {r["chunk_id"] for r in out} == {"doc-a:0"}
    out = retrieval.lexical_search(
        chunks, query="needle", top_k=5, filters={"section_path": "doc-a/sec1"}
    )
    assert {r["chunk_id"] for r in out} == {"doc-a:1"}


def test_hybrid_filter_both_channels(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    ids = ["doc-a:0", "doc-b:0", "doc-b:1"]
    _write_index(tmp_path, ids)
    chunks = _chunks_for(ids)

    out = retrieval.hybrid_search(cfg, chunks, "0", top_k=5, filters={"doc_id": "doc-b"})
    assert out
    assert {r["doc_id"] for r in out} == {"doc-b"}


def test_doc_search_lexical_mode_passes_filters(tmp_path, _clean_caches, monkeypatch):
    """retrieval_mode=lexical goes through doc_search's lexical path with the
    MCP filters intact (the fallback previously dropped them)."""
    monkeypatch.setattr(retrieval, "load_config", lambda: _make_cfg(tmp_path))
    ids = ["doc-a:0", "doc-b:0", "doc-b:1"]
    monkeypatch.setattr(retrieval, "load_chunks", lambda cfg: _chunks_for(ids))
    cfg = _make_cfg(tmp_path)
    cfg["mcp"] = {"retrieval_mode": "lexical"}

    out = retrieval.doc_search("0", top_k=5, filters={"doc_id": "doc-b"})
    assert out
    assert {r["doc_id"] for r in out} == {"doc-b"}


def test_doc_search_lexical_fallback_keeps_filters(tmp_path, _clean_caches, monkeypatch):
    """Semantic mode, no index on disk: doc_search falls back to lexical and
    must apply the filters there too (previously silently dropped)."""
    cfg = _make_cfg(tmp_path)
    cfg["mcp"] = {"retrieval_mode": "semantic"}
    monkeypatch.setattr(retrieval, "load_config", lambda: cfg)
    ids = ["doc-a:0", "doc-b:0", "doc-b:1"]
    monkeypatch.setattr(retrieval, "load_chunks", lambda cfg: _chunks_for(ids))

    out = retrieval.doc_search("0", top_k=5, filters={"doc_id": "doc-b"})
    assert out
    assert {r["doc_id"] for r in out} == {"doc-b"}


def test_doc_search_empty_filter_match_is_empty(tmp_path, _clean_caches, monkeypatch):
    """A filter matching nothing must not degrade to unfiltered lexical
    results in doc_search (the caller believes the filter applied)."""
    cfg = _make_cfg(tmp_path)
    cfg["mcp"] = {"retrieval_mode": "lexical"}
    monkeypatch.setattr(retrieval, "load_config", lambda: cfg)
    ids = ["doc-a:0", "doc-b:0"]
    monkeypatch.setattr(retrieval, "load_chunks", lambda cfg: _chunks_for(ids))

    out = retrieval.doc_search("0", top_k=5, filters={"doc_id": "doc-zzz"})
    assert out == []


# --- (f) top_k larger than the number of allowed chunks --------------------


def test_filter_top_k_exceeds_allowed_returns_all_allowed(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    ids = ["doc-a:0", "doc-a:1", "doc-b:0"]
    _write_index(tmp_path, ids)
    chunks = _chunks_for(ids)

    out = retrieval.semantic_search(cfg, chunks, "0", top_k=10, filters={"doc_id": "doc-b"})
    assert out is not None
    assert [r["chunk_id"] for r in out] == ["doc-b:0"]


# --- (g) filters without metadata -> ValueError, not silent ignore ---------


def test_search_with_filter_requires_chunk_meta(tmp_path, _clean_caches):
    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ["doc-a:0", "doc-b:0"])
    vi = VectorIndex(cfg)
    vi.is_available()

    with pytest.raises(ValueError, match="filters require chunk metadata"):
        vi.search_with_filter("0", 3, {"doc_id": "doc-b"}, chunk_meta=None)

    # Qdrant-style store path: with metadata provided the filter applies.
    dists, ids, payloads = vi.search_with_filter(
        "0", 3, {"doc_id": "doc-a"}, chunk_meta={"doc-a:0": {"doc_id": "doc-a"}}
    )
    assert ids == ["doc-a:0"]


def test_no_filters_keeps_old_behavior(tmp_path, _clean_caches):
    """filters=None must keep the plain-search contract (2-tuple via search,
    3-tuple with synthetic payloads via search_with_filter)."""
    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ["doc-a:0"])
    vi = VectorIndex(cfg)
    dists, ids = vi.search("0", top_k=1)
    assert ids == ["doc-a:0"]
    dists, ids, payloads = vi.search_with_filter("0", 1, None, chunk_meta=None)
    assert ids == ["doc-a:0"]
    assert payloads == [{"chunk_id": "doc-a:0"}]


# --- review fixes -----------------------------------------------------------


def test_l2_index_filter_order_matches_plain_search(tmp_path, _clean_caches):
    """On an L2 index the filtered scan must rank by L2 distance (lower is
    better) and return the same distances as the unfiltered store.search:
    an IP-style descending dot-scan would invert the order (review finding)."""
    import faiss as _faiss

    cfg = _make_cfg(tmp_path)
    cfg["index"]["metric"] = "l2"
    index_dir = tmp_path / "build" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    ids = ["doc-a:0", "doc-a:1", "doc-a:2"]
    dim = 8
    # Deliberately NOT normalized: query close to v1, far from v0.
    vecs = np.array(
        [[1.0, 0.1, 0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0, 0, 0]],
        dtype="float32",
    )
    index = _faiss.IndexFlatL2(dim)
    index.add(vecs)
    _faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "index_meta.json").write_text(
        json.dumps({"model_name": "stub", "dim": dim, "metric": "l2", "chunk_ids": ids}),
        encoding="utf-8",
    )

    chunks = _chunks_for(ids)
    vi = VectorIndex(cfg)

    # Plain L2 search ranks by distance ascending: v0 (0.01), v1 (1.0),
    # v2 (16.0). An IP-style descending dot-scan would return the exact
    # reverse order (q.v2 = 5 > q.v1 = 2 > q.v0 = 1) -- so this asserts the
    # metric is actually honored, not just that filtering happens.
    d_plain, ids_plain = vi.search("0", top_k=3)
    assert ids_plain == ["doc-a:0", "doc-a:1", "doc-a:2"]

    d_f, ids_f, _p = vi.search_with_filter(
        "0", 3, {"doc_id": "doc-a"}, chunk_meta=_chunk_meta(chunks)
    )
    # Same order as the unfiltered search, and L2 distances (not IP scores).
    assert ids_f == ids_plain == ["doc-a:0", "doc-a:1", "doc-a:2"]
    np.testing.assert_allclose(d_f, d_plain, rtol=1e-5)
    assert d_f[0] <= d_f[1] <= d_f[2]  # ascending distances, lower = better


def test_empty_chunk_meta_raises(tmp_path, _clean_caches):
    """chunk_meta={} is as useless as None: fail loud, not an empty result."""
    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ["doc-a:0"])
    vi = VectorIndex(cfg)
    vi.is_available()
    with pytest.raises(ValueError, match="filters require chunk metadata"):
        vi.search_with_filter("0", 3, {"doc_id": "doc-a"}, chunk_meta={})


def test_is_table_false_keeps_non_table(tmp_path, _clean_caches):
    """is_table=False is honored symmetrically: only non-table chunks."""
    ids = ["doc-a:0", "doc-a:1", "doc-a:2"]  # is_table on position 0 only
    chunks = _chunks_for(ids)
    # Make every chunk lexically match the query so the filter alone decides.
    for c in chunks:
        c["text"] = f"common needle {c['text']}"
    out = retrieval.lexical_search(chunks, query="needle", top_k=5, filters={"is_table": False})
    assert {r["chunk_id"] for r in out} == {"doc-a:1", "doc-a:2"}

    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ids)
    out = retrieval.semantic_search(cfg, chunks, "0", top_k=10, filters={"is_table": False})
    assert out is not None
    assert {r["chunk_id"] for r in out} == {"doc-a:1", "doc-a:2"}


def test_unknown_filter_key_raises(tmp_path, _clean_caches):
    """A typo'd key (e.g. table_only) must fail loud, not filter nothing."""
    cfg = _make_cfg(tmp_path)
    _write_index(tmp_path, ["doc-a:0"])
    vi = VectorIndex(cfg)
    vi.is_available()
    with pytest.raises(ValueError, match="unsupported filter keys"):
        vi.search_with_filter(
            "0", 3, {"table_only": True}, chunk_meta={"doc-a:0": {"doc_id": "doc-a"}}
        )

    # Same policy in the lexical path.
    with pytest.raises(ValueError, match="unsupported filter keys"):
        retrieval.lexical_search(_chunks_for(["doc-a:0"]), "0", 3, filters={"table_only": True})
