"""Consistency tests for chunk postprocessing (A7/A8).

- A7: the rebuild path must run the same _postprocess_chunks over the FULL
  merged chunk set as ingest does, so the same document produces the same
  chunk text regardless of which path built it.
- A8: whitespace cleanup preserves single newlines (markdown structure --
  tables, lists) while collapsing runs of 3+ newlines to paragraph breaks.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def _make_root(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    config_dir = base / "config"
    config_dir.mkdir()
    for d in [
        "build/docs_md",
        "build/chunks",
        "build/blocks",
        "build/index",
        "sources/archived",
        "sources/incoming",
    ]:
        (base / d).mkdir(parents=True, exist_ok=True)
    (base / "config" / "config.yaml").write_text(
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
        sources:
          incremental_ingest: false
    """)
    )
    return base / "config" / "config.yaml"


def _read_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        return []
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


# -- A7: ingest vs rebuild produce identical chunk text -----------------


def test_ingest_and_rebuild_produce_identical_chunk_text(tmp_path: Path) -> None:
    """The same document must yield the same chunk text through ingest()
    (non-incremental path, which runs _postprocess_chunks) and through
    rebuild(), which now postprocesses the full merged chunk set too.
    """
    from doc_rag.raglib.pipeline import ingest, rebuild

    doc_text = (
        "Document about semantic search pipelines. " * 20
        + "\n\n"
        + "Second section with additional detail. " * 20
    )

    # Path 1: ingest
    root_a = _make_root(tmp_path / "a")
    (tmp_path / "a" / "sources" / "incoming" / "doc.txt").write_text(doc_text)
    ingest(str(root_a))
    chunks_a = _read_chunks(tmp_path / "a" / "build" / "chunks" / "chunks.jsonl")
    assert chunks_a, "ingest produced no chunks"

    # Path 2: rebuild over the same source
    root_b = _make_root(tmp_path / "b")
    (tmp_path / "b" / "sources" / "archived" / "doc.txt").write_text(doc_text)
    rebuild(str(root_b), force=True, skip_index=True)
    chunks_b = _read_chunks(tmp_path / "b" / "build" / "chunks" / "chunks.jsonl")
    assert chunks_b, "rebuild produced no chunks"

    texts_a = [c["text"] for c in chunks_a if c["doc_id"].startswith("doc_")]
    texts_b = [c["text"] for c in chunks_b if c["doc_id"].startswith("doc_")]
    assert texts_a == texts_b, "chunk text differs between ingest and rebuild for the same document"


def test_rebuild_postprocess_runs_on_full_merged_set(tmp_path: Path) -> None:
    """rebuild() must postprocess the FULL merged chunk set (existing + new)
    BEFORE dedup: duplicate header detection is corpus-wide, and merged
    chunks loaded from chunks.jsonl must be cleaned too, not only freshly
    parsed ones.
    """
    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_root(tmp_path)
    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    docs_md = tmp_path / "build" / "docs_md"

    # Existing doc in the manifest. The short header text appears as the
    # ENTIRE text of 3 chunks (header detection counts whole chunk texts,
    # 3-100 chars), and also as a leading line inside body chunks -- after
    # corpus-wide postprocessing the header must be gone from everywhere.
    header = "ISO 13239 Header Line"
    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "doc_id": "doc_a",
                "source_file": "a.txt",
                "sha256": "aaa",
                "md_path": "build/docs_md/doc_a.md",
                "chunk_count": 5,
            },
        ],
    }
    (tmp_path / "build" / "manifest.json").write_text(json.dumps(manifest))

    raw_chunks = [
        {"doc_id": "doc_a", "chunk_id": "doc_a:0", "text": header},
        {"doc_id": "doc_a", "chunk_id": "doc_a:1", "text": header},
        {"doc_id": "doc_a", "chunk_id": "doc_a:2", "text": header},
        {
            "doc_id": "doc_a",
            "chunk_id": "doc_a:3",
            "text": f"{header}\nFirst body with enough meaningful content here.",
        },
        {
            "doc_id": "doc_a",
            "chunk_id": "doc_a:4",
            "text": f"{header}\nSecond body with enough meaningful content here.",
        },
    ]
    with open(chunks_path, "w") as f:
        for c in raw_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    (docs_md / "doc_a.md").write_text("content a")

    rebuild(str(config_path), force=False, skip_index=True)

    loaded = _read_chunks(chunks_path)
    texts = [c["text"] for c in loaded]
    # The header repeats in 3+ chunks corpus-wide, so it must be stripped
    # everywhere by the rebuild-path postprocessing.
    assert texts, "no chunks survived rebuild"
    assert all(header not in t for t in texts), (
        f"rebuild did not strip corpus-wide duplicate headers: {texts}"
    )
    # The header-only chunks collapse to nothing and are dropped; the body
    # chunks keep their (header-stripped) text. Gaps in chunk_idx numbering
    # are acceptable by design (chunk_ids stay stable).
    assert len(texts) == 2, f"expected 2 body chunks to survive, got {len(texts)}"


# -- A8: markdown structure survives whitespace cleanup -----------------


def test_postprocess_preserves_single_newlines_markdown_table():
    """A markdown table's row separators must survive cleanup."""
    from doc_rag.raglib.pipeline import _postprocess_chunks

    table = "| Name | Value |\n|------|-------|\n| alpha | 1 |\n| beta | 2 |"
    chunks = [{"text": table, "chunk_id": "a:0"}]
    result = _postprocess_chunks(chunks, min_text_len=10)
    assert len(result) == 1
    text = result[0]["text"]
    assert "\n" in text, "single newlines were collapsed to spaces"
    assert text == table


def test_postprocess_collapses_three_plus_newlines_to_paragraph_break():
    """Runs of 3+ newlines collapse to \\n\\n -- and the rule is reachable,
    unlike the old order where \\s+ had already destroyed all newlines.
    """
    import re

    from doc_rag.raglib.pipeline import _postprocess_chunks

    text = "First paragraph with enough content.\n\n\n\nSecond paragraph with enough content."
    chunks = [{"text": text, "chunk_id": "a:0"}]
    result = _postprocess_chunks(chunks, min_text_len=10)
    assert len(result) == 1
    cleaned = result[0]["text"]
    assert "\n\n\n" not in cleaned, "3+ newlines were not collapsed"
    assert "\n\n" in cleaned, "paragraph break must remain after collapsing"
    # The collapse regex actually did the work (no empty assertions):
    assert re.search(r"\n{3,}", cleaned) is None
    assert (
        cleaned == "First paragraph with enough content.\n\nSecond paragraph with enough content."
    )
