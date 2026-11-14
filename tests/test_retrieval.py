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


def test_legacy_corrupt_meta_blocks_positional_mapping(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If index_meta.json exists but is corrupt (unparseable JSON), the
    legacy positional-mapping path must NOT run: semantic_search returns
    None and doc_search degrades to lexical (task 5 fail-closed gate)."""
    faiss = pytest.importorskip("faiss")
    import numpy as np

    root = tmp_path
    # Two chunks in an order DIFFERENT from index positions: if the legacy
    # positional mapping ran, the query would return the wrong chunk.
    chunks = [
        {"chunk_id": "d:0", "doc_id": "d", "text": "alpha document", "source_file": "d.pdf"},
        {"chunk_id": "d:1", "doc_id": "d", "text": "zeppelin marker", "source_file": "d.pdf"},
    ]
    chunks_dir = root / "build" / "chunks_jsonl"
    chunks_dir.mkdir(parents=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Index position 0 = "zeppelin marker" vector, position 1 = "alpha".
    v_alpha = np.zeros((1, 8), dtype="float32")
    v_alpha[0, 0] = 1.0
    v_zepp = np.zeros((1, 8), dtype="float32")
    v_zepp[0, 1] = 1.0
    index = faiss.IndexFlatIP(8)
    index.add(np.vstack([v_zepp, v_alpha]))
    index_dir = root / "build" / "index"
    index_dir.mkdir(parents=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    # Corrupt meta: present on disk but unparseable.
    (index_dir / "index_meta.json").write_text("{not json", encoding="utf-8")

    cfg = {
        "_root": str(root),
        "paths": {"chunks_dir": "build/chunks_jsonl", "index_dir": "build/index"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return v_zepp.copy()

    # VectorIndex path is unavailable (as in the legacy-path tests above).
    monkeypatch.setattr(
        "doc_rag.raglib.vector_index.VectorIndex.__init__",
        lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("no vector backend")),
    )
    monkeypatch.setattr("doc_rag.raglib.model_loader.load_embedder", lambda cfg: FakeModel())

    retrieval._SEMANTIC_CACHE["index"] = None
    retrieval._SEMANTIC_CACHE["index_path"] = None
    retrieval._SEMANTIC_CACHE["index_mtime"] = None
    retrieval._SEMANTIC_CACHE["model_key"] = None

    out = semantic_search(cfg, chunks, "zeppelin", top_k=2)
    assert out is None  # corrupt meta -> fail closed, no positional mapping

    # And the tool-level fallback serves lexical results instead. This is
    # also the positional-mapping proof: index position 0 holds the
    # "zeppelin marker" vector, so a legacy positional hit would return
    # chunks[0] = "d:0"; lexical returns the true match "d:1".
    monkeypatch.setattr(retrieval, "load_config", lambda: cfg)
    monkeypatch.setattr(retrieval, "_CHUNKS_CACHE", {"path": None, "mtime": None, "chunks": None})
    results = retrieval.doc_search("zeppelin", top_k=2)
    assert [r["chunk_id"] for r in results] == ["d:1"]


def test_legacy_meta_len_mismatch_blocks_positional_mapping(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parseable meta whose chunk_ids length disagrees with the index
    ntotal must also block the legacy path (positional mapping unsafe)."""
    faiss = pytest.importorskip("faiss")
    import numpy as np

    root = tmp_path
    chunks = [
        {"chunk_id": "d:0", "doc_id": "d", "text": "alpha document", "source_file": "d.pdf"},
        {"chunk_id": "d:1", "doc_id": "d", "text": "zeppelin marker", "source_file": "d.pdf"},
    ]
    chunks_dir = root / "build" / "chunks_jsonl"
    chunks_dir.mkdir(parents=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    v_zepp = np.zeros((1, 8), dtype="float32")
    v_zepp[0, 1] = 1.0
    index = faiss.IndexFlatIP(8)
    index.add(v_zepp)  # ntotal=1, meta below lists 2 ids
    index_dir = root / "build" / "index"
    index_dir.mkdir(parents=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "index_meta.json").write_text(
        json.dumps({"chunk_ids": ["d:0", "d:1"]}), encoding="utf-8"
    )

    cfg = {
        "_root": str(root),
        "paths": {"chunks_dir": "build/chunks_jsonl", "index_dir": "build/index"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return v_zepp.copy()

    monkeypatch.setattr(
        "doc_rag.raglib.vector_index.VectorIndex.__init__",
        lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError("no vector backend")),
    )
    monkeypatch.setattr("doc_rag.raglib.model_loader.load_embedder", lambda cfg: FakeModel())

    retrieval._SEMANTIC_CACHE["index"] = None
    retrieval._SEMANTIC_CACHE["index_path"] = None
    retrieval._SEMANTIC_CACHE["index_mtime"] = None
    retrieval._SEMANTIC_CACHE["model_key"] = None

    out = semantic_search(cfg, chunks, "zeppelin", top_k=2)
    assert out is None


def _make_vector_cfg(tmp_path) -> dict:
    """Minimal cfg for the VectorIndex (pluggable-backend) path of
    semantic_search: faiss backend, stubbed embedder, tmp index dir."""
    return {
        "_root": str(tmp_path),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {"index_dir": "build/index", "chunks_dir": "build/chunks_jsonl"},
        "embeddings": {"model_name": "stub", "device": "cpu"},
    }


def _write_vector_index(
    root, texts: list[str], chunk_ids: list[str] | None = None, namespace: str = "default"
) -> None:
    """Write faiss.index + index_meta.json directly (bypasses VectorIndex.build
    and its embedder): each text gets its own deterministic vector."""
    import faiss as _faiss
    import numpy as np

    ids = chunk_ids if chunk_ids is not None else [f"d{i}:0" for i in range(len(texts))]
    index_dir = (
        root / "build" / "index" / namespace if namespace != "default" else root / "build" / "index"
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    dim = 8
    vecs = np.linspace(0.1, 0.9, len(texts) * dim, dtype="float32").reshape(len(texts), dim)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    index = _faiss.IndexFlatIP(dim)
    index.add(vecs)
    _faiss.write_index(index, str(index_dir / "faiss.index"))
    meta = {
        "model_name": "stub",
        "dim": dim,
        "metric": "ip",
        "chunk_ids": ids,
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")


class _StubQueryEmbedder:
    """Encode each query as a unit vector along a distinct axis."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        import numpy as np

        out = []
        for t in texts:
            v = np.zeros((1, self._dim), dtype="float32")
            v[0, int(t) % self._dim] = 1.0
            out.append(v)
        return np.vstack(out)


def test_vectorindex_cache_sees_rebuilt_index(tmp_path, monkeypatch) -> None:
    """After an in-process rebuild overwrites faiss.index + index_meta.json,
    the cached VectorIndex must be recreated: a search then returns the new
    chunk ids without a restart (mtime validation, task 3)."""

    monkeypatch.setattr(
        "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubQueryEmbedder()
    )
    retrieval._VECTOR_INDEX_CACHE.clear()

    chunks_a = [{"chunk_id": "d0:0", "doc_id": "d0", "text": "old content", "source_file": "a.pdf"}]
    chunks_b = [{"chunk_id": "d9:0", "doc_id": "d9", "text": "new content", "source_file": "b.pdf"}]
    cfg = _make_vector_cfg(tmp_path)

    _write_vector_index(tmp_path, ["0"], chunk_ids=["d0:0"])
    out = semantic_search(cfg, chunks_a, "0", top_k=2)
    assert out is not None and [c["chunk_id"] for c in out] == ["d0:0"]

    # Simulate the rebuild: rewrite both files with a different index set.
    import time

    time.sleep(0.02)  # ensure st_mtime strictly increases (ns resolution)
    old_ids = set(retrieval._VECTOR_INDEX_CACHE)
    _write_vector_index(tmp_path, ["1"], chunk_ids=["d9:0"])

    out = semantic_search(cfg, chunks_b, "1", top_k=2)
    assert out is not None and [c["chunk_id"] for c in out] == ["d9:0"]
    # The stale entry was replaced, not reused.
    assert set(retrieval._VECTOR_INDEX_CACHE) == old_ids


def test_vectorindex_cache_per_namespace_and_invalidation(tmp_path, monkeypatch) -> None:
    """Entries are keyed by "{root}:{namespace}"; invalidate_index_cache()
    drops them all; the LRU cap evicts the oldest entry beyond 8."""
    monkeypatch.setattr(
        "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubQueryEmbedder()
    )
    retrieval._VECTOR_INDEX_CACHE.clear()

    chunks = [{"chunk_id": "d0:0", "doc_id": "d0", "text": "content", "source_file": "a.pdf"}]
    cfg = _make_vector_cfg(tmp_path)
    _write_vector_index(tmp_path, ["0"], namespace="ns1")

    out = semantic_search(cfg, chunks, "0", top_k=2, namespace="ns1")
    assert out is not None and [c["chunk_id"] for c in out] == ["d0:0"]
    assert f"{tmp_path}:ns1" in retrieval._VECTOR_INDEX_CACHE

    retrieval.invalidate_index_cache()
    assert len(retrieval._VECTOR_INDEX_CACHE) == 0

    # LRU cap: fill 9 namespaces, the first must be evicted.
    for i in range(9):
        ns = f"ns{i}"
        _write_vector_index(tmp_path, ["0"], namespace=ns)
        out = semantic_search(cfg, chunks, "0", top_k=2, namespace=ns)
        assert out is not None
    assert len(retrieval._VECTOR_INDEX_CACHE) == 8
    assert f"{tmp_path}:ns0" not in retrieval._VECTOR_INDEX_CACHE
    assert f"{tmp_path}:ns8" in retrieval._VECTOR_INDEX_CACHE


def test_vectorindex_cache_missing_files_recreated(tmp_path, monkeypatch) -> None:
    """If index files disappear after caching, the next search recreates the
    VectorIndex (which then reports unavailable -> None, legacy fallback)."""
    monkeypatch.setattr(
        "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _StubQueryEmbedder()
    )
    retrieval._VECTOR_INDEX_CACHE.clear()

    chunks = [{"chunk_id": "d0:0", "doc_id": "d0", "text": "content", "source_file": "a.pdf"}]
    cfg = _make_vector_cfg(tmp_path)
    _write_vector_index(tmp_path, ["0"])

    out = semantic_search(cfg, chunks, "0", top_k=2)
    assert out is not None

    (tmp_path / "build" / "index" / "faiss.index").unlink()
    (tmp_path / "build" / "index" / "index_meta.json").unlink()
    out = semantic_search(cfg, chunks, "0", top_k=2)
    # Files gone: new VectorIndex finds no index -> falls through to legacy
    # path, which also finds nothing -> None (no stale results served).
    assert out is None
