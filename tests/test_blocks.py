"""Tests for the v1.5 typed-blocks layer (Block dataclass + JSONL I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag.raglib.blocks import (
    BLOCKS_SCHEMA_VERSION,
    Block,
    BlocksSchemaTooNew,
    blocks_to_markdown,
    dump_blocks,
    iter_blocks,
    load_blocks,
)


def _para(text: str, idx: int = 0) -> Block:
    return Block(
        block_id=f"tmp:{idx:04d}",
        doc_id="tmp",
        type="paragraph",
        text=text,
        source_backend="pymupdf",
    )


# --------------------------------------------------------------------------
# Block construction / validation
# --------------------------------------------------------------------------


def test_block_minimal_fields():
    b = Block(
        block_id="doc-a:0000",
        doc_id="doc-a",
        type="paragraph",
        text="hello",
        source_backend="pymupdf",
    )
    assert b.block_id == "doc-a:0000"
    assert b.type == "paragraph"
    assert b.level is None
    assert b.metadata == {}


def test_block_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown block type"):
        Block(
            block_id="doc-a:0000",
            doc_id="doc-a",
            type="nonsense",  # type: ignore[arg-type]
            text="x",
            source_backend="pymupdf",
        )


def test_block_rejects_empty_ids():
    with pytest.raises(ValueError, match="block_id"):
        Block(block_id="", doc_id="d", type="paragraph", text="x", source_backend="pymupdf")
    with pytest.raises(ValueError, match="doc_id"):
        Block(block_id="b", doc_id="", type="paragraph", text="x", source_backend="pymupdf")
    with pytest.raises(ValueError, match="source_backend"):
        Block(block_id="b", doc_id="d", type="paragraph", text="x", source_backend="")


def test_block_validates_confidence_range():
    with pytest.raises(ValueError, match="confidence"):
        Block(
            block_id="b",
            doc_id="d",
            type="paragraph",
            text="x",
            source_backend="docling",
            confidence=1.5,
        )


def test_block_validates_negative_level():
    with pytest.raises(ValueError, match="level"):
        Block(
            block_id="b",
            doc_id="d",
            type="heading",
            text="x",
            source_backend="pymupdf",
            level=-1,
        )


def test_block_validates_bbox_length():
    with pytest.raises(ValueError, match="bbox"):
        Block(
            block_id="b",
            doc_id="d",
            type="paragraph",
            text="x",
            source_backend="pymupdf",
            bbox=(1.0, 2.0, 3.0),  # type: ignore[arg-type]
        )


def test_block_coerces_bbox_to_floats():
    b = Block(
        block_id="b",
        doc_id="d",
        type="paragraph",
        text="x",
        source_backend="pymupdf",
        bbox=(1, 2, 3, 4),  # type: ignore[arg-type]
    )
    assert b.bbox == (1.0, 2.0, 3.0, 4.0)


# --------------------------------------------------------------------------
# Serialization shape
# --------------------------------------------------------------------------


def test_to_jsonl_dict_omits_none_and_empty_metadata():
    b = Block(
        block_id="b",
        doc_id="d",
        type="paragraph",
        text="x",
        source_backend="pymupdf",
    )
    out = b.to_jsonl_dict()
    assert out == {
        "block_id": "b",
        "doc_id": "d",
        "type": "paragraph",
        "text": "x",
        "source_backend": "pymupdf",
    }
    assert "level" not in out
    assert "page" not in out
    assert "bbox" not in out
    assert "metadata" not in out


def test_to_jsonl_dict_includes_optional_fields_when_set():
    b = Block(
        block_id="b",
        doc_id="d",
        type="heading",
        text="Section 1",
        source_backend="python-docx",
        level=1,
        page=3,
        bbox=(0.0, 0.0, 100.0, 20.0),
        confidence=0.95,
        metadata={"original_index": 0},
    )
    out = b.to_jsonl_dict()
    assert out["level"] == 1
    assert out["page"] == 3
    assert out["bbox"] == [0.0, 0.0, 100.0, 20.0]
    assert out["confidence"] == 0.95
    assert out["metadata"] == {"original_index": 0}


def test_from_jsonl_dict_tolerates_unknown_fields():
    """A future writer may add fields; old readers must ignore them."""
    raw = {
        "block_id": "b",
        "doc_id": "d",
        "type": "paragraph",
        "text": "x",
        "source_backend": "docling",
        "future_field": "ignored",
        "another_future": [1, 2, 3],
    }
    b = Block.from_jsonl_dict(raw)
    assert b.text == "x"


def test_from_jsonl_dict_missing_required_raises():
    raw = {"block_id": "b", "doc_id": "d", "type": "paragraph"}
    with pytest.raises(ValueError, match="missing required field"):
        Block.from_jsonl_dict(raw)


# --------------------------------------------------------------------------
# Round-trip through JSONL
# --------------------------------------------------------------------------


def test_round_trip_through_file(tmp_path: Path):
    src_blocks = [
        Block(
            block_id="doc-x:0000",
            doc_id="doc-x",
            type="heading",
            text="Section 1",
            source_backend="python-docx",
            level=1,
        ),
        Block(
            block_id="doc-x:0001",
            doc_id="doc-x",
            type="paragraph",
            text="Lead paragraph.",
            source_backend="python-docx",
            page=1,
        ),
        Block(
            block_id="doc-x:0002",
            doc_id="doc-x",
            type="table",
            text="A1 | B1\nA2 | B2",
            source_backend="python-docx",
            metadata={"cells": [["A1", "B1"], ["A2", "B2"]]},
        ),
    ]

    out_path = tmp_path / "blocks.jsonl"
    n = dump_blocks(out_path, src_blocks)
    assert n == 3

    loaded = load_blocks(out_path)
    assert len(loaded) == 3
    for a, b in zip(src_blocks, loaded, strict=True):
        assert a == b


def test_iter_blocks_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "blocks.jsonl"
    rec = json.dumps(
        {
            "block_id": "b",
            "doc_id": "d",
            "type": "paragraph",
            "text": "x",
            "source_backend": "pymupdf",
        }
    )
    p.write_text(f"\n{rec}\n\n{rec}\n", encoding="utf-8")

    blocks = list(iter_blocks(p))
    assert len(blocks) == 2


def test_dump_creates_parent_directory(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "blocks.jsonl"
    n = dump_blocks(
        out,
        [
            Block(
                block_id="b",
                doc_id="d",
                type="paragraph",
                text="x",
                source_backend="pymupdf",
            )
        ],
    )
    assert n == 1
    assert out.is_file()


# --------------------------------------------------------------------------
# Schema-version guard
# --------------------------------------------------------------------------


def test_load_blocks_accepts_current_version(tmp_path: Path):
    p = tmp_path / "blocks.jsonl"
    dump_blocks(
        p,
        [
            Block(
                block_id="b",
                doc_id="d",
                type="paragraph",
                text="x",
                source_backend="pymupdf",
            )
        ],
    )
    out = load_blocks(p, schema_version=BLOCKS_SCHEMA_VERSION)
    assert len(out) == 1


# --------------------------------------------------------------------------
# Markdown derivation
# --------------------------------------------------------------------------


def test_blocks_to_markdown_baseline_single_paragraph():
    """The v1.5 baseline (one paragraph for the whole document) must
    produce byte-identical markdown to what the legacy parser did:
        `# {title}\n\n{body}\n`
    """
    md = blocks_to_markdown([_para("hello world")], "doc.pdf")
    assert md == "# doc.pdf\n\nhello world\n"


def test_blocks_to_markdown_baseline_multiline_paragraph_preserved():
    body = "line one\nline two\nline three"
    md = blocks_to_markdown([_para(body)], "x.txt")
    assert md == "# x.txt\n\nline one\nline two\nline three\n"


def test_blocks_to_markdown_no_blocks_yields_title_only():
    md = blocks_to_markdown([], "empty.txt")
    assert md == "# empty.txt\n"


def test_blocks_to_markdown_headings_offset_by_one():
    """A v1.5 parser may emit a `heading` block at level 1; the document
    title already occupies H1, so headings shift by one level down."""
    blocks = [
        Block(
            block_id="tmp:0000",
            doc_id="tmp",
            type="heading",
            text="Section 1",
            source_backend="python-docx",
            level=1,
        ),
        _para("body of section 1", idx=1),
    ]
    md = blocks_to_markdown(blocks, "report.docx")
    assert md == "# report.docx\n\n## Section 1\n\nbody of section 1\n"


def test_blocks_to_markdown_list_items_indented_by_level():
    blocks = [
        Block(
            block_id="tmp:0000",
            doc_id="tmp",
            type="list_item",
            text="top",
            source_backend="python-docx",
            level=0,
        ),
        Block(
            block_id="tmp:0001",
            doc_id="tmp",
            type="list_item",
            text="nested",
            source_backend="python-docx",
            level=1,
        ),
    ]
    md = blocks_to_markdown(blocks, "list.docx")
    assert md == "# list.docx\n\n- top\n\n  - nested\n"


def test_blocks_to_markdown_table_text_used_verbatim():
    blocks = [
        Block(
            block_id="tmp:0000",
            doc_id="tmp",
            type="table",
            text="A | B\nC | D",
            source_backend="python-docx",
        )
    ]
    md = blocks_to_markdown(blocks, "t.docx")
    assert md == "# t.docx\n\nA | B\nC | D\n"


def test_load_blocks_refuses_future_version(tmp_path: Path):
    p = tmp_path / "blocks.jsonl"
    dump_blocks(
        p,
        [
            Block(
                block_id="b",
                doc_id="d",
                type="paragraph",
                text="x",
                source_backend="pymupdf",
            )
        ],
    )
    with pytest.raises(BlocksSchemaTooNew) as excinfo:
        load_blocks(p, schema_version=BLOCKS_SCHEMA_VERSION + 5)
    assert "doc-rag migrate" in str(excinfo.value)
