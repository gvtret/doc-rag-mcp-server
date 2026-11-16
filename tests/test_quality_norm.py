"""Tests for quality module: remove_duplicate_headers and scoring."""

from __future__ import annotations

import pytest

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.quality import (
    Warning,
    _compute_score,
    remove_duplicate_headers,
    run_quality_checks,
)


def _block(text: str, btype: str = "paragraph", page: int = 1) -> Block:
    return Block(
        block_id=f"b_{hash(text)}",
        doc_id="test",
        type=btype,
        text=text,
        source_backend="docling",
        page=page,
    )


def test_remove_duplicate_headers_removes_repeated_text():
    """Text appearing on 3+ pages should be removed from blocks."""
    blocks = [
        _block("ISO 13239", "heading", 1),
        _block("Some content on page 1", "paragraph", 1),
        _block("ISO 13239", "heading", 2),
        _block("Content on page 2", "paragraph", 2),
        _block("ISO 13239", "heading", 3),
        _block("Content on page 3", "paragraph", 3),
        _block("Unique text", "paragraph", 1),
    ]
    cleaned = remove_duplicate_headers(blocks, threshold=3)
    texts = [b.text for b in cleaned]
    assert "ISO 13239" not in texts
    assert "Some content on page 1" in texts
    assert "Unique text" in texts


def test_remove_duplicate_headers_keeps_below_threshold():
    """Text on fewer pages than threshold should be kept."""
    blocks = [
        _block("Rare header", "heading", 1),
        _block("Rare header", "heading", 2),
        _block("Content", "paragraph", 1),
    ]
    cleaned = remove_duplicate_headers(blocks, threshold=3)
    assert len(cleaned) == 3


def test_remove_duplicate_headers_no_duplicates():
    """No changes when there are no duplicates."""
    blocks = [
        _block("A", "heading", 1),
        _block("B", "paragraph", 2),
        _block("C", "heading", 3),
    ]
    cleaned = remove_duplicate_headers(blocks)
    assert cleaned is blocks


def test_remove_duplicate_headers_empty():
    """Empty input returns empty output."""
    assert remove_duplicate_headers([]) == []


def test_compute_score_all_info():
    """Info-level warnings should have minimal penalty."""
    warnings = [Warning(severity="info", code="test", message="x")] * 10
    score = _compute_score(warnings, 100)
    assert score == pytest.approx(0.95, abs=0.01)


def test_compute_score_mixed_severities():
    """Mix of severities should produce expected score."""
    warnings = [
        Warning(severity="error", code="e", message="x"),
        Warning(severity="warn", code="w", message="x"),
        Warning(severity="info", code="i", message="x"),
    ]
    score = _compute_score(warnings, 100)
    expected = 1.0 - (0.10 + 0.02 + 0.005)
    assert score == pytest.approx(expected, abs=0.01)


def test_compute_score_scale_independent():
    """Score must not depend on document size (no saturation to 0.00).

    Regression for the GOST 870-5-101 case: the same 22 empty-table
    warnings must give the same score for a 10-block and a 1600-block
    document, instead of flooring both towards 0.00 with a fixed
    per-document penalty.
    """
    warnings = [Warning(severity="warn", code="t", message="x")] * 5
    small = _compute_score(warnings, 10)
    large = _compute_score(warnings, 1600)
    assert small == pytest.approx(large)
    assert small > 0.5


def test_compute_score_unknown_severity_penalized():
    """Unknown severity still counts as a (small) warning, not a pass."""
    warnings = [Warning(severity="critical", code="t", message="x")]
    assert _compute_score(warnings, 100) == pytest.approx(0.995)


def test_run_quality_checks_empty_blocks():
    """Empty block list should return perfect score."""
    report = run_quality_checks("test", [])
    assert report.score == 1.0
    assert len(report.warnings) == 0


def test_remove_duplicate_headers_only_affects_heading_paragraph():
    """Table blocks should not be removed even if text repeats."""
    blocks = [
        _block("Repeated", "heading", 1),
        _block("Repeated", "heading", 2),
        _block("Repeated", "heading", 3),
        _block("Repeated", "table", 4),
    ]
    cleaned = remove_duplicate_headers(blocks)
    table_blocks = [b for b in cleaned if b.type == "table"]
    assert len(table_blocks) == 1
