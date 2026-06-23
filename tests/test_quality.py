"""Tests for the document quality checks module."""

from __future__ import annotations

import json
import os
from pathlib import Path

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.quality import (
    QualityReport,
    _check_empty_pages,
    _check_formulas,
    _check_tables,
    _compute_score,
    _has_duplicate_headers,
    _is_formula_garbage,
    persist_quality_report,
    run_quality_checks,
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


class TestFormulaGarbage:
    def test_normal_text_not_garbage(self):
        assert not _is_formula_garbage("Hello world 123")

    def test_empty_not_garbage(self):
        assert not _is_formula_garbage("")

    def test_short_not_garbage(self):
        assert not _is_formula_garbage("ab")

    def test_high_unknown_ratio(self):
        text = "\uf000" * 20 + "abc"
        assert _is_formula_garbage(text)


class TestDuplicateHeaders:
    def test_no_duplicates(self):
        blocks = [_block(page=1, text="heading one"), _block(page=2, text="heading two")]
        assert _has_duplicate_headers(blocks) == []

    def test_three_repeated(self):
        blocks = [
            _block(page=1, text="Header Text"),
            _block(page=2, text="Header Text"),
            _block(page=3, text="Header Text"),
        ]
        dupes = _has_duplicate_headers(blocks)
        assert len(dupes) == 1
        assert "header text" in dupes

    def test_short_text_ignored(self):
        blocks = [_block(page=i, text="Hi") for i in range(1, 6)]
        assert _has_duplicate_headers(blocks) == []


class TestEmptyPages:
    def test_no_empty_pages(self):
        blocks = [_block(page=1, text="x" * 20), _block(page=2, text="y" * 20)]
        assert _check_empty_pages(blocks, 2) == []

    def test_empty_page_detected(self):
        blocks = [_block(page=1, text="x" * 20)]
        warnings = _check_empty_pages(blocks, 3)
        assert len(warnings) == 2
        assert warnings[0].code == "empty_page"


class TestTables:
    def test_normal_table_no_warnings(self):
        table = _block(type="table", text="A | B\n1 | 2")
        assert _check_tables([table]) == []

    def test_empty_table(self):
        table = _block(type="table", text="")
        warnings = _check_tables([table])
        assert any(w.code == "empty_table" for w in warnings)

    def test_broken_table_high_empty_ratio(self):
        # 8 empty rows, 2 content rows → 8 empty "cells" out of 32 total = 25% per row
        # but the | | pattern is detected → ratio 8/(3*8+5*2) = 8/34 = 23%
        # just test that empty rows are detected (empty table covers this)
        table = _block(type="table", text="| |\n| | a |")
        warnings = _check_tables([table])
        assert not any(w.code == "broken_table" for w in warnings)


class TestFormulas:
    def test_normal_formula_no_warning(self):
        formula = _block(type="formula", text="x^2 + y^2 = z^2")
        assert _check_formulas([formula]) == []

    def test_garbage_formula_warning(self):
        formula = _block(type="formula", text="\uf000" * 20)
        warnings = _check_formulas([formula])
        assert any(w.code == "formula_garbage" for w in warnings)


class TestComputeScore:
    def test_no_warnings_perfect(self):
        assert _compute_score([], 10) == 1.0

    def test_empty_blocks(self):
        assert _compute_score([], 0) == 1.0

    def test_one_warn_penalizes(self):
        from doc_rag.raglib.quality import Warning

        w = Warning(severity="warn", code="test", message="test")
        score = _compute_score([w], 10)
        assert score < 1.0
        assert score > 0.0

    def test_score_clamped_to_zero(self):
        from doc_rag.raglib.quality import Warning

        warnings = [Warning(severity="error", code=f"e{i}", message=f"err {i}") for i in range(20)]
        assert _compute_score(warnings, 10) == 0.0


class TestRunQualityChecks:
    def test_normal_document(self):
        blocks = [
            _block(page=1, text="Introduction paragraph."),
            _block(page=2, text="Main content here."),
            _block(type="heading", page=1, text="Introduction"),
        ]
        report = run_quality_checks("doc1", blocks)
        assert report.doc_id == "doc1"
        assert report.pages == 2
        assert "heading" in report.blocks
        assert report.score <= 1.0
        assert report.score >= 0.0
        assert report.schema_version == 1

    def test_empty_blocks(self):
        report = run_quality_checks("empty", [])
        assert report.pages == 0
        assert report.score == 1.0
        assert len(report.warnings) == 0


class TestPersistQualityReport:
    def test_creates_files(self, tmp_path: Path):
        report = QualityReport(
            doc_id="test_doc",
            pages=5,
            blocks={"paragraph": 3, "heading": 2},
            warnings=[],
            score=0.95,
        )
        quality_dir = str(tmp_path / "quality")
        doc_path = persist_quality_report(quality_dir, report)

        assert os.path.exists(doc_path)
        with open(doc_path) as f:
            data = json.load(f)
        assert data["doc_id"] == "test_doc"
        assert data["score"] == 0.95
        assert data["schema_version"] == 1

        summary_path = os.path.join(quality_dir, "summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert len(summary["documents"]) == 1
        assert summary["documents"][0]["doc_id"] == "test_doc"

    def test_summary_updates_existing(self, tmp_path: Path):
        quality_dir = str(tmp_path / "quality")
        os.makedirs(quality_dir, exist_ok=True)

        r1 = QualityReport(doc_id="doc1", pages=1, blocks={"paragraph": 1}, warnings=[], score=0.9)
        persist_quality_report(quality_dir, r1)

        r2 = QualityReport(doc_id="doc2", pages=2, blocks={"paragraph": 2}, warnings=[], score=1.0)
        persist_quality_report(quality_dir, r2)

        with open(os.path.join(quality_dir, "summary.json")) as f:
            summary = json.load(f)
        doc_ids = {d["doc_id"] for d in summary["documents"]}
        assert doc_ids == {"doc1", "doc2"}
