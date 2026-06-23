"""Tests for the structure-aware recursive chunker."""

from __future__ import annotations

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.chunker_recursive import (
    ChunkNode,
    _build_tree,
    _estimate_tokens,
    _render_table_summary,
    _section_path,
    chunk_recursive,
)


def _block(**kwargs) -> Block:
    defaults = dict(
        block_id="tmp:0000",
        doc_id="test",
        type="paragraph",
        text="hello world",
        source_backend="docling",
    )
    defaults.update(kwargs)
    return Block(**defaults)


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("") >= 1
        assert _estimate_tokens("x" * 80) == 20


class TestBuildTree:
    def test_flat_blocks(self):
        blocks = [_block(text="p1"), _block(text="p2")]
        tree = _build_tree(blocks)
        assert len(tree) == 1
        assert len(tree[0].blocks) == 2

    def test_heading_creates_children(self):
        blocks = [
            _block(type="heading", level=1, text="H1"),
            _block(text="body1"),
            _block(type="heading", level=2, text="H2"),
            _block(text="body2"),
            _block(type="heading", level=1, text="H1b"),
            _block(text="body3"),
        ]
        tree = _build_tree(blocks)
        assert len(tree) == 2  # two H1 nodes
        assert tree[0].heading_text == "H1"
        assert len(tree[0].children) == 1  # H2 under H1
        assert tree[0].children[0].heading_text == "H2"
        assert tree[1].heading_text == "H1b"

    def test_heading_level_pop(self):
        blocks = [
            _block(type="heading", level=1, text="A"),
            _block(type="heading", level=3, text="A3"),
            _block(type="heading", level=2, text="B2"),
            _block(text="body"),
        ]
        tree = _build_tree(blocks)
        assert len(tree) == 1  # A at top level
        assert tree[0].heading_text == "A"
        assert len(tree[0].children) == 2  # A3 and B2 under A
        assert tree[0].children[0].heading_text == "A3"
        assert tree[0].children[1].heading_text == "B2"


class TestSectionPath:
    def test_empty(self):
        assert _section_path(ChunkNode()) == ""

    def test_with_heading(self):
        node = ChunkNode(heading_level=1, heading_text="Intro")
        assert _section_path(node) == "Intro"

    def test_nested(self):
        child = ChunkNode(heading_level=2, heading_text="Section 1.1")
        assert _section_path(child, ["Chapter 1"]) == "Chapter 1.Section 1.1"


class TestRenderTableSummary:
    def test_basic(self):
        table = _block(type="table", text="A | B\n1 | 2\n3 | 4")
        summary = _render_table_summary(table)
        assert "3 строк" in summary
        assert "A | B" in summary

    def test_empty_table(self):
        table = _block(type="table", text="")
        summary = _render_table_summary(table)
        assert "Таблица" in summary


class TestChunkRecursive:
    def test_empty_blocks(self):
        assert chunk_recursive([], target_tokens=100) == []

    def test_single_paragraph(self):
        blocks = [_block(text="Simple text")]
        chunks = chunk_recursive(blocks, target_tokens=100)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Simple text"
        assert chunks[0]["chunk_idx"] == 0

    def test_heading_creates_section_path(self):
        blocks = [
            _block(type="heading", level=1, text="Chapter"),
            _block(text="Content under chapter"),
        ]
        chunks = chunk_recursive(blocks, target_tokens=100)
        assert len(chunks) == 1
        assert chunks[0]["section_path"] == "Chapter"

    def test_table_is_atomic(self):
        table = _block(type="table", text="H1 | H2\nA | B\nC | D")
        blocks = [_block(text="Before"), table, _block(text="After")]
        chunks = chunk_recursive(blocks, target_tokens=100)
        table_chunks = [c for c in chunks if c.get("is_table")]
        assert len(table_chunks) == 1
        summary_chunks = [c for c in chunks if c.get("is_table_summary")]
        assert len(summary_chunks) == 1

    def test_splitting_when_over_target(self):
        blocks = [_block(text="x " * 600)]
        chunks = chunk_recursive(blocks, target_tokens=10)
        assert len(chunks) > 1

    def test_chunks_have_text_key(self):
        blocks = [_block(text="alpha"), _block(text="beta")]
        chunks = chunk_recursive(blocks, target_tokens=1000)
        for c in chunks:
            assert "text" in c
            assert "chunk_idx" in c

    def test_nested_headings_preserve_path(self):
        blocks = [
            _block(type="heading", level=1, text="A"),
            _block(type="heading", level=2, text="B"),
            _block(text="body"),
        ]
        chunks = chunk_recursive(blocks, target_tokens=1000)
        assert chunks[0]["section_path"] == "A.B"

    def test_mixed_content(self):
        blocks = [
            _block(type="heading", level=1, text="Intro"),
            _block(text="First paragraph."),
            _block(type="table", text="X | Y\n1 | 2"),
            _block(text="Second paragraph after table."),
            _block(type="heading", level=1, text="Section 2"),
            _block(text="More content."),
        ]
        chunks = chunk_recursive(blocks, target_tokens=1000)
        assert len(chunks) >= 3
        paths = [c.get("section_path", "") for c in chunks]
        assert any("Intro" in p for p in paths)
        assert any("Section 2" in p for p in paths)
