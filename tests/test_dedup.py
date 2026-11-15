"""Regression tests for document/chunk deduplication logic.

Each test covers a specific gap that was found and fixed:
- rebuild(): docs dedup by doc_id, chunks dedup by chunk_id
- rebuild_docs(): same dedup after merge
- VectorIndex.build(): input chunk dedup
- _build_documents_list(): pending dedup by basename, indexed dedup by doc_id
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

# ── VectorIndex.build() input dedup ────────────────────────────────────


def test_vector_index_build_dedup_input_chunks(tmp_path: Path) -> None:
    """Duplicate chunk_ids in input should not create ghost vectors."""
    from doc_rag.raglib.vector_index import VectorIndex

    root = str(tmp_path / "dedup_vi")
    os.makedirs(os.path.join(root, "build/index"), exist_ok=True)
    cfg = {
        "_root": root,
        "embeddings": {"model_name": "BAAI/bge-small-en-v1.5", "device": "cpu", "batch_size": 8},
        "index": {"backend": "faiss"},
        "paths": {"index_dir": "build/index"},
    }
    vi = VectorIndex(cfg)
    chunks = [
        {"chunk_id": "doc1:0", "text": "first version"},
        {"chunk_id": "doc1:0", "text": "second version"},
        {"chunk_id": "doc1:1", "text": "unique chunk"},
    ]
    vi.build(chunks, force=True, log=lambda *a: None)
    assert vi.size == 2


def test_vector_index_build_incremental_dedup_input(tmp_path: Path) -> None:
    """Incremental build should also dedup within the input batch."""
    from doc_rag.raglib.vector_index import VectorIndex

    root = str(tmp_path / "dedup_vi_inc")
    os.makedirs(os.path.join(root, "build/index"), exist_ok=True)
    cfg = {
        "_root": root,
        "embeddings": {"model_name": "BAAI/bge-small-en-v1.5", "device": "cpu", "batch_size": 8},
        "index": {"backend": "faiss"},
        "paths": {"index_dir": "build/index"},
    }
    vi = VectorIndex(cfg)
    vi.build([{"chunk_id": "a:0", "text": "alpha"}], force=True, log=lambda *a: None)
    assert vi.size == 1
    vi.build(
        [
            {"chunk_id": "b:0", "text": "beta"},
            {"chunk_id": "b:0", "text": "beta dup"},
        ],
        force=False,
        log=lambda *a: None,
    )
    assert vi.size == 2


# ── helpers ────────────────────────────────────────────────────────────


def _make_root(tmp_path: Path) -> Path:
    """Create standard directory layout and config for rebuild tests.

    load_config computes root = dirname(config_path), so the config
    must live at <root>/config/config.yaml for paths to resolve correctly.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for d in [
        "build/docs_md",
        "build/chunks",
        "build/blocks",
        "build/index",
        "sources/archived",
        "sources/incoming",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(
        textwrap.dedent("""\
        paths:
          sources_archived: sources/archived
          sources_incoming: sources/incoming
          docs_md_dir: build/docs_md
          chunks_dir: build/chunks
          manifest_path: build/manifest.json
          blocks_dir: build/blocks
          index_dir: build/index
        parsing:
          pdf_backend: docling
        chunking:
          strategy: fixed
          target_tokens: 512
          overlap_tokens: 64
        embeddings:
          model_name: BAAI/bge-small-en-v1.5
          device: cpu
    """)
    )
    return tmp_path / "config" / "config.yaml"


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ── rebuild() doc + chunk dedup ────────────────────────────────────────


def test_rebuild_deduplicates_docs_by_doc_id(tmp_path: Path) -> None:
    """rebuild() should drop duplicate doc_id entries from existing manifest."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    docs_md = tmp_path / "build" / "docs_md"

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "dup_x",
                "source_file": "a.txt",
                "sha256": "aaa",
                "md_path": "build/docs_md/dup_x.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "dup_x",
                "source_file": "a.txt",
                "sha256": "aaa",
                "md_path": "build/docs_md/dup_x.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "unique_y",
                "source_file": "b.txt",
                "sha256": "bbb",
                "md_path": "build/docs_md/unique_y.md",
                "chunk_count": 1,
            },
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    (docs_md / "dup_x.md").write_text("dup content")
    (docs_md / "unique_y.md").write_text("unique content")

    rebuild(str(config_path), force=False)

    with open(manifest_path) as f:
        m = json.load(f)
    doc_ids = [d["doc_id"] for d in m["documents"]]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc_ids remain: {doc_ids}"
    assert len(doc_ids) == 2


def test_rebuild_deduplicates_chunks_by_chunk_id(tmp_path: Path) -> None:
    """rebuild() should drop duplicate chunk_ids from the merged chunks list."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    docs_md = tmp_path / "build" / "docs_md"

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "doc_a",
                "source_file": "a.txt",
                "sha256": "aaa",
                "md_path": "build/docs_md/doc_a.md",
                "chunk_count": 2,
            },
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    _write_chunks(
        chunks_path,
        [
            {
                "doc_id": "doc_a",
                "chunk_id": "doc_a:0",
                "text": "chunk zero with enough text to survive postprocessing",
            },
            {
                "doc_id": "doc_a",
                "chunk_id": "doc_a:0",
                "text": "chunk zero duplicate entry with different body content",
            },
            {
                "doc_id": "doc_a",
                "chunk_id": "doc_a:1",
                "text": "chunk one with enough text to survive postprocessing too",
            },
        ],
    )

    (docs_md / "doc_a.md").write_text("content a")

    rebuild(str(config_path), force=False)

    loaded = []
    with open(chunks_path) as f:
        for line in f:
            loaded.append(json.loads(line))
    cids = [c["chunk_id"] for c in loaded]
    assert len(cids) == len(set(cids)), f"Duplicate chunk_ids remain: {cids}"
    assert len(cids) == 2


# ── _build_documents_list() dedup ──────────────────────────────────────


def test_build_documents_list_dedup_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same filename in both incoming/ and archived/ should appear once in pending."""
    from doc_rag.server import mcp_http

    incoming = tmp_path / "sources" / "incoming"
    archived = tmp_path / "sources" / "archived"
    incoming.mkdir(parents=True)
    archived.mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)

    (incoming / "dup.txt").write_text("from incoming")
    (archived / "dup.txt").write_text("from archived")
    (tmp_path / "build" / "manifest.json").write_text(json.dumps({"documents": []}))

    (tmp_path / "config.yaml").write_text(
        textwrap.dedent("""\
        paths:
          sources_archived: sources/archived
          sources_incoming: sources/incoming
          manifest_path: build/manifest.json
    """)
    )

    monkeypatch.setattr(mcp_http, "_root_dir", lambda: tmp_path)
    monkeypatch.setattr(mcp_http, "_config_path", lambda: tmp_path / "config.yaml")

    result = mcp_http._build_documents_list()
    pending_names = [p["basename"] for p in result["pending"]]
    assert pending_names.count("dup.txt") == 1
    assert result["pending_count"] == 1


def test_build_documents_list_dedup_indexed_by_doc_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate doc_id entries in manifest should appear once in indexed."""
    from doc_rag.server import mcp_http

    archived = tmp_path / "sources" / "archived"
    archived.mkdir(parents=True)
    (tmp_path / "sources" / "incoming").mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)

    (tmp_path / "build" / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": "dup_x", "source_file": "x.txt", "sha256": "aaa", "chunk_count": 1},
                    {"doc_id": "dup_x", "source_file": "x.txt", "sha256": "aaa", "chunk_count": 1},
                    {"doc_id": "uniq_y", "source_file": "y.txt", "sha256": "bbb", "chunk_count": 1},
                ]
            }
        )
    )

    (archived / "x.txt").write_text("x")
    (archived / "y.txt").write_text("y")

    (tmp_path / "config.yaml").write_text(
        textwrap.dedent("""\
        paths:
          sources_archived: sources/archived
          sources_incoming: sources/incoming
          manifest_path: build/manifest.json
    """)
    )

    monkeypatch.setattr(mcp_http, "_root_dir", lambda: tmp_path)
    monkeypatch.setattr(mcp_http, "_config_path", lambda: tmp_path / "config.yaml")

    result = mcp_http._build_documents_list()
    assert result["indexed_count"] == 2


def test_build_documents_list_includes_coverage_for_ocr_badge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `/ui/documents` response must carry `coverage.ocr` per document —
    the Svelte Documents page reads it to render the OCR badge. Regression
    for a gap where this endpoint dropped `coverage` (only `/ui/doc-stats`
    had it), so the badge had no data to render from.
    """
    from doc_rag.server import mcp_http

    archived = tmp_path / "sources" / "archived"
    archived.mkdir(parents=True)
    (tmp_path / "sources" / "incoming").mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)

    (tmp_path / "build" / "manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "scanned",
                        "source_file": "scanned.pdf",
                        "sha256": "aaa",
                        "chunk_count": 1,
                        "coverage": {
                            "ocr": {
                                "applied": True,
                                "pages_recognized": 3,
                                "confidence": 0.97,
                            }
                        },
                    },
                ]
            }
        )
    )
    (archived / "scanned.pdf").write_text("x")

    (tmp_path / "config.yaml").write_text(
        textwrap.dedent("""\
        paths:
          sources_archived: sources/archived
          sources_incoming: sources/incoming
          manifest_path: build/manifest.json
    """)
    )

    monkeypatch.setattr(mcp_http, "_root_dir", lambda: tmp_path)
    monkeypatch.setattr(mcp_http, "_config_path", lambda: tmp_path / "config.yaml")

    result = mcp_http._build_documents_list()
    doc = result["indexed"][0]
    assert doc["coverage"]["ocr"]["applied"] is True
    assert doc["coverage"]["ocr"]["pages_recognized"] == 3
    assert doc["coverage"]["ocr"]["confidence"] == 0.97


# ── rebuild_docs() dedup ───────────────────────────────────────────────


def test_rebuild_docs_deduplicates_chunks(tmp_path: Path) -> None:
    """rebuild_docs() should dedup chunks after merge."""
    from doc_rag.raglib.pipeline import rebuild_docs

    config_path = _make_root(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    docs_md = tmp_path / "build" / "docs_md"

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "keep_me",
                "source_file": "keep.txt",
                "sha256": "111",
                "md_path": "build/docs_md/keep_me.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "dup_target",
                "source_file": "target.txt",
                "sha256": "222",
                "md_path": "build/docs_md/dup_target.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "dup_target",
                "source_file": "target.txt",
                "sha256": "222",
                "md_path": "build/docs_md/dup_target.md",
                "chunk_count": 1,
            },
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    _write_chunks(
        chunks_path,
        [
            {"doc_id": "keep_me", "chunk_id": "keep_me:0", "text": "keep"},
            {"doc_id": "dup_target", "chunk_id": "dup_target:0", "text": "old target"},
            {"doc_id": "dup_target", "chunk_id": "dup_target:0", "text": "old target dup"},
        ],
    )

    (docs_md / "keep_me.md").write_text("keep content")

    # Create the source file so rebuild_docs can find it
    (tmp_path / "sources" / "archived" / "target.txt").write_text("target content")

    rebuild_docs(str(config_path), doc_ids=["dup_target"], filenames=[])

    with open(manifest_path) as f:
        m = json.load(f)
    doc_ids = [d["doc_id"] for d in m["documents"]]
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc_ids in manifest: {doc_ids}"

    loaded = []
    with open(chunks_path) as f:
        for line in f:
            loaded.append(json.loads(line))
    cids = [c["chunk_id"] for c in loaded]
    assert len(cids) == len(set(cids)), f"Duplicate chunk_ids: {cids}"


def test_rebuild_docs_drops_orphaned_chunks_when_content_hash_changes(tmp_path: Path) -> None:
    """rebuild_docs() regenerates doc_id from the file's content hash, so a
    document whose content actually changed gets a *new* doc_id. The old
    doc_id's chunks must not survive in chunks.jsonl — otherwise stale
    content keeps showing up in search results forever.
    """
    from doc_rag.raglib.pipeline import rebuild_docs

    config_path = _make_root(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    docs_md = tmp_path / "build" / "docs_md"

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "keep_me",
                "source_file": "keep.txt",
                "sha256": "111",
                "md_path": "build/docs_md/keep_me.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "target",
                "source_file": "target.txt",
                "sha256": "222",
                "md_path": "build/docs_md/target.md",
                "chunk_count": 1,
            },
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    _write_chunks(
        chunks_path,
        [
            {"doc_id": "keep_me", "chunk_id": "keep_me:0", "text": "keep chunk text"},
            {"doc_id": "target", "chunk_id": "target:0", "text": "stale old text"},
        ],
    )
    (docs_md / "keep_me.md").write_text("keep content")
    # Content differs from what produced sha256 "222", so rebuild_docs will
    # compute a brand-new doc_id for this file.
    (tmp_path / "sources" / "archived" / "target.txt").write_text("updated target content")

    rebuild_docs(str(config_path), doc_ids=["target"], filenames=[])

    with open(manifest_path) as f:
        m = json.load(f)
    final_doc_ids = {d["doc_id"] for d in m["documents"]}
    assert "target" not in final_doc_ids, "old doc_id should be gone from the manifest"

    loaded = []
    with open(chunks_path) as f:
        for line in f:
            loaded.append(json.loads(line))
    chunk_doc_ids = {c["doc_id"] for c in loaded}
    assert chunk_doc_ids == final_doc_ids, (
        f"chunks.jsonl references doc_ids not in the manifest (orphans): "
        f"{chunk_doc_ids - final_doc_ids}"
    )
    assert not any(c["text"] == "stale old text" for c in loaded), "old chunk text must be gone"


def test_rebuild_docs_only_reembeds_target_document(tmp_path: Path, monkeypatch) -> None:
    """rebuild_docs() must not re-embed chunks belonging to documents that
    weren't targeted — only the rebuilt doc's (new) chunks should hit the
    embedder. Regression for a full-corpus re-embed on every rebuild_docs
    call (`vi.reset()` + `vi.build(..., force=True)` on all chunks).
    """
    import doc_rag.raglib.pipeline as pipeline_mod
    from doc_rag.raglib.pipeline import rebuild_docs

    pipeline_mod._VECTOR_INDEX = None  # this test's tmp_path root is unique; start clean

    config_path = _make_root(tmp_path)
    manifest_path = tmp_path / "build" / "manifest.json"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    docs_md = tmp_path / "build" / "docs_md"

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "keep_me",
                "source_file": "keep.txt",
                "sha256": "111",
                "md_path": "build/docs_md/keep_me.md",
                "chunk_count": 1,
            },
            {
                "doc_id": "target",
                "source_file": "target.txt",
                "sha256": "222",
                "md_path": "build/docs_md/target.md",
                "chunk_count": 1,
            },
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    _write_chunks(
        chunks_path,
        [
            {"doc_id": "keep_me", "chunk_id": "keep_me:0", "text": "keep chunk text"},
            {"doc_id": "target", "chunk_id": "target:0", "text": "old target text"},
        ],
    )
    (docs_md / "keep_me.md").write_text("keep content")
    (tmp_path / "sources" / "archived" / "target.txt").write_text("new target content")

    encoded_texts: list[str] = []

    class _RecordingEmbedder:
        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
            import numpy as np

            encoded_texts.extend(texts)
            return np.random.RandomState(0).randn(len(texts), 8).astype(np.float32)

    monkeypatch.setattr(
        "doc_rag.raglib.vector_index._load_embedder", lambda cfg: _RecordingEmbedder()
    )

    # Pre-populate the index with both docs' current chunks, as if a prior
    # ingest/rebuild had already embedded the whole corpus.
    from doc_rag.raglib.pipeline import load_config

    cfg = load_config(str(config_path))
    vi = pipeline_mod._get_vector_index(cfg)
    vi.build(
        [
            {"doc_id": "keep_me", "chunk_id": "keep_me:0", "text": "keep chunk text"},
            {"doc_id": "target", "chunk_id": "target:0", "text": "old target text"},
        ],
        force=True,
        log=lambda *a: None,
    )
    assert vi.size == 2
    encoded_texts.clear()  # only care about what rebuild_docs() triggers below

    rebuild_docs(str(config_path), doc_ids=["target"], filenames=[])

    assert vi.size == 2, "index should still hold exactly one chunk per document"
    assert not any("keep" in t for t in encoded_texts), (
        f"keep_me's chunk was re-embedded, expected only target's: {encoded_texts}"
    )
    assert len(encoded_texts) == 1, f"expected exactly target's one new chunk, got {encoded_texts}"
