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
            {"doc_id": "doc_a", "chunk_id": "doc_a:0", "text": "chunk zero"},
            {"doc_id": "doc_a", "chunk_id": "doc_a:0", "text": "chunk zero dup"},
            {"doc_id": "doc_a", "chunk_id": "doc_a:1", "text": "chunk one"},
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
