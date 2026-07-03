"""Regression tests for incremental per-file persistence.

Verifies that chunks are written to chunks.jsonl after each file is parsed,
so that a crash mid-rebuild doesn't lose already-processed documents.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch


def _make_root(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for d in [
        "build/docs_md", "build/chunks", "build/blocks", "build/index",
        "sources/archived", "sources/incoming",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(textwrap.dedent("""\
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
    """))
    return tmp_path / "config" / "config.yaml"


def _create_txt(path: Path, content: str) -> None:
    path.write_text(content)


def _count_chunks(chunks_path: Path) -> int:
    if not chunks_path.exists():
        return 0
    count = 0
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        return []
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


# ── Test 1: Chunks appear in chunks.jsonl after each file ──────────────

def test_rebuild_per_file_chunk_persist(tmp_path: Path) -> None:
    """Each parsed file's chunks should be in chunks.jsonl immediately after parsing."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"

    _create_txt(archived / "a.txt", "First document content. " * 20)
    _create_txt(archived / "b.txt", "Second document content. " * 20)
    _create_txt(archived / "c.txt", "Third document content. " * 20)

    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"

    call_count = 0
    original_append = None

    def counting_append(path, chunks):
        nonlocal call_count
        call_count += 1
        original_append(path, chunks)

    from doc_rag.raglib import pipeline
    original_append = pipeline._append_chunks
    with patch.object(pipeline, "_append_chunks", side_effect=counting_append):
        rebuild(str(config_path), force=True)

    assert call_count == 3, f"Expected 3 _append_chunks calls, got {call_count}"
    assert _count_chunks(chunks_path) > 0


# ── Test 2: Crash recovery — chunks exist for completed files ──────────

def test_rebuild_crash_recovery_chunks_exist(tmp_path: Path) -> None:
    """If _ingest_sources fails on file #3, files #1-2 should have chunks in chunks.jsonl."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"

    _create_txt(archived / "a.txt", "Document A content. " * 20)
    _create_txt(archived / "b.txt", "Document B content. " * 20)
    _create_txt(archived / "c.txt", "Document C content. " * 20)

    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"

    parse_call = 0
    original_parse = None

    def failing_parse(cfg, src):
        nonlocal parse_call
        parse_call += 1
        if parse_call == 3:
            raise RuntimeError("Simulated crash on file #3")
        return original_parse(cfg, src)

    from doc_rag.raglib import pipeline
    original_parse = pipeline.parse_document
    with patch.object(pipeline, "parse_document", side_effect=failing_parse):
        rebuild(str(config_path), force=True)

    # Files #1 and #2 should have chunks
    chunks = _read_chunks(chunks_path)
    doc_ids = {c["doc_id"].split("_")[0] for c in chunks}
    assert len(chunks) > 0, "No chunks found after partial rebuild"
    # At least 2 documents should have been processed
    assert len(doc_ids) >= 2, f"Expected at least 2 doc_ids, got {doc_ids}"


# ── Test 3: rebuild_docs per-file persist ──────────────────────────────

def test_rebuild_docs_per_file_chunk_persist(tmp_path: Path) -> None:
    """rebuild_docs should also persist chunks per-file."""
    from doc_rag.raglib.pipeline import rebuild_docs

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"
    docs_md = tmp_path / "build" / "docs_md"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    manifest_path = tmp_path / "build" / "manifest.json"

    # Create a manifest with one existing doc
    _create_txt(archived / "existing.txt", "Existing document. " * 20)
    (docs_md / "existing.md").write_text("existing content")
    manifest = {
        "schema_version": 1,
        "documents": [{
            "doc_id": "existing_abc123",
            "source_file": "sources/archived/existing.txt",
            "sha256": "aaa",
            "md_path": "build/docs_md/existing.md",
            "chunk_count": 1,
        }],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Write existing chunks
    with open(chunks_path, "w") as f:
        f.write(json.dumps({"doc_id": "existing_abc123", "chunk_id": "existing_abc123:0", "text": "existing chunk"}) + "\n")

    # New file to rebuild
    _create_txt(archived / "new.txt", "New document content. " * 20)

    call_count = 0
    original_append = None

    def counting_append(path, chunks):
        nonlocal call_count
        call_count += 1
        original_append(path, chunks)

    from doc_rag.raglib import pipeline
    original_append = pipeline._append_chunks
    with patch.object(pipeline, "_append_chunks", side_effect=counting_append):
        rebuild_docs(str(config_path), doc_ids=[], filenames=["new.txt"])

    assert call_count >= 1, "Expected at least 1 _append_chunks call"
    chunks = _read_chunks(chunks_path)
    assert len(chunks) >= 2, f"Expected at least 2 chunks (1 existing + 1 new), got {len(chunks)}"


# ── Test 4: force=True triggers full index rebuild ─────────────────────

def test_rebuild_force_index_full_rebuild(tmp_path: Path) -> None:
    """force=True should rebuild the vector index from scratch."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"

    _create_txt(archived / "a.txt", "Document A content. " * 20)

    build_calls = []
    original_build = None

    def tracking_build(self, chunks, force=False, log=print):
        build_calls.append({"force": force, "chunk_count": len(chunks)})
        return original_build(self, chunks, force=force, log=log)

    from doc_rag.raglib import vector_index
    original_build = vector_index.VectorIndex.build
    with patch.object(vector_index.VectorIndex, "build", tracking_build):
        rebuild(str(config_path), force=True)

    assert len(build_calls) >= 1
    assert any(c["force"] for c in build_calls), "Expected at least one force=True build"


# ── Test 5: force=False triggers incremental index update ──────────────

def test_rebuild_incremental_index_update(tmp_path: Path) -> None:
    """force=False with new chunks should do incremental index update."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"

    _create_txt(archived / "a.txt", "Document A content. " * 20)

    build_calls = []
    original_build = None

    def tracking_build(self, chunks, force=False, log=print):
        build_calls.append({"force": force, "chunk_count": len(chunks)})
        return original_build(self, chunks, force=force, log=log)

    from doc_rag.raglib import vector_index
    original_build = vector_index.VectorIndex.build
    with patch.object(vector_index.VectorIndex, "build", tracking_build):
        rebuild(str(config_path), force=False)

    assert len(build_calls) >= 1
    # rebuild() now always does force=True to ensure index sync with chunks.jsonl


# ── Test 6: ingest incremental per-file index update ───────────────────

def test_ingest_per_file_index_update(tmp_path: Path) -> None:
    """ingest() incremental should update the vector index per-file."""
    from doc_rag.raglib.pipeline import ingest

    config_path = _make_root(tmp_path)
    incoming = tmp_path / "sources" / "incoming"

    _create_txt(incoming / "a.txt", "Document A content. " * 20)
    _create_txt(incoming / "b.txt", "Document B content. " * 20)

    build_calls = []
    original_build = None

    def tracking_build(self, chunks, force=False, log=print):
        build_calls.append({"force": force, "chunk_count": len(chunks)})
        return original_build(self, chunks, force=force, log=log)

    from doc_rag.raglib import vector_index
    original_build = vector_index.VectorIndex.build
    with patch.object(vector_index.VectorIndex, "build", tracking_build):
        ingest(str(config_path))

    # Should have at least 2 incremental builds (one per file)
    assert len(build_calls) >= 2, f"Expected >=2 build calls, got {len(build_calls)}"
    assert all(not c["force"] for c in build_calls), "All builds should be incremental"


# ── Test 7: Skip logic after crash — .md exists → file skipped ─────────

def test_rebuild_skip_logic_after_crash(tmp_path: Path) -> None:
    """If .md exists from a prior partial rebuild, the file should be skipped."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"
    docs_md = tmp_path / "build" / "docs_md"
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    manifest_path = tmp_path / "build" / "manifest.json"

    # Create a file with .md already on disk (simulating prior partial rebuild)
    _create_txt(archived / "a.txt", "Document A content. " * 20)
    file_hash = __import__("hashlib").sha256(open(archived / "a.txt", "rb").read()).hexdigest()
    doc_id = "a_" + file_hash[:10]
    (docs_md / f"{doc_id}.md").write_text("existing content")

    # Manifest with this doc
    manifest = {
        "schema_version": 1,
        "documents": [{
            "doc_id": doc_id,
            "source_file": "sources/archived/a.txt",
            "sha256": file_hash,
            "md_path": f"build/docs_md/{doc_id}.md",
            "chunk_count": 0,
        }],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Write some chunks for this doc
    with open(chunks_path, "w") as f:
        f.write(json.dumps({"doc_id": doc_id, "chunk_id": f"{doc_id}:0", "text": "existing chunk"}) + "\n")

    parse_count = 0
    original_parse = None

    def counting_parse(cfg, src):
        nonlocal parse_count
        parse_count += 1
        return original_parse(cfg, src)

    from doc_rag.raglib import pipeline
    original_parse = pipeline.parse_document
    with patch.object(pipeline, "parse_document", counting_parse):
        rebuild(str(config_path), force=False)

    # File should have been skipped (not re-parsed)
    assert parse_count == 0, f"Expected 0 parses (file skipped), got {parse_count}"


# ── Test 8: manifest written at end, chunks per-file ───────────────────

def test_manifest_written_at_end_chunks_per_file(tmp_path: Path) -> None:
    """Manifest should be written after all files, chunks after each file."""
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    archived = tmp_path / "sources" / "archived"

    _create_txt(archived / "a.txt", "Document A content. " * 20)
    _create_txt(archived / "b.txt", "Document B content. " * 20)

    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    manifest_path = tmp_path / "build" / "manifest.json"

    # Track chunk counts after each append
    chunk_counts_after_append = []
    original_append = None

    def tracking_append(path, chunks):
        original_append(path, chunks)
        chunk_counts_after_append.append(_count_chunks(chunks_path))

    from doc_rag.raglib import pipeline
    original_append = pipeline._append_chunks
    with patch.object(pipeline, "_append_chunks", side_effect=tracking_append):
        rebuild(str(config_path), force=True)

    # Each append should increase the chunk count
    assert len(chunk_counts_after_append) == 2
    assert chunk_counts_after_append[0] > 0, "First append should add chunks"
    assert chunk_counts_after_append[1] > chunk_counts_after_append[0], "Second append should add more chunks"

    # Manifest should exist and have 2 docs
    assert manifest_path.exists()
    with open(manifest_path) as f:
        m = json.load(f)
    assert len(m["documents"]) == 2
