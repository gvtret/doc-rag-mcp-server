"""Tests for parsers.py quality cascade logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_quality_cascade_triggers_on_low_score(tmp_path: Path):
    """When Docling score < threshold, Unstructured should be attempted."""
    from doc_rag.raglib.blocks import Block
    from doc_rag.raglib.quality import QualityReport, Warning

    with patch("doc_rag.raglib.docling_backend.parse_pdf_docling") as mock_docling:
        blocks = [Block(f"b{i}", "test", "paragraph", "x" * 10, "docling", page=i) for i in range(5)]
        mock_docling.return_value = ("text", blocks, {"pages": 5})

        with patch("doc_rag.raglib.quality.run_quality_checks") as mock_qc:
            # First call: Docling score low, second call: Unstructured score high
            mock_qc.side_effect = [
                QualityReport(doc_id="test", pages=5, score=0.3, blocks={},
                              warnings=[Warning(severity="warn", code="test", message="x")]),
                QualityReport(doc_id="test", pages=5, score=0.8, blocks={},
                              warnings=[]),
            ]

            with patch("doc_rag.raglib.unstructured_backend.parse_pdf_unstructured") as mock_unstructured:
                good_blocks = [Block(f"g{i}", "test", "paragraph", "good content " * 10, "unstructured", page=i) for i in range(5)]
                mock_unstructured.return_value = ("good text", good_blocks, {"pages": 5})

                with patch("doc_rag.raglib.parsers.detect_supported_extension") as mock_ext:
                    mock_ext.return_value = ".pdf"

                    from doc_rag.raglib.parsers import parse_document
                    result = parse_document(
                        {"parsing": {"quality_fallback_threshold": 0.5}},
                        str(tmp_path / "test.pdf"),
                    )
                    assert result["blocks"][0].source_backend == "unstructured"


def test_quality_cascade_keeps_docling_when_better(tmp_path: Path):
    """When Unstructured scores lower, Docling result should be kept."""
    from doc_rag.raglib.blocks import Block
    from doc_rag.raglib.quality import QualityReport, Warning

    with patch("doc_rag.raglib.docling_backend.parse_pdf_docling") as mock_docling:
        blocks = [Block(f"b{i}", "test", "paragraph", "content " * 10, "docling", page=i) for i in range(5)]
        mock_docling.return_value = ("text", blocks, {"pages": 5})

        with patch("doc_rag.raglib.quality.run_quality_checks") as mock_qc:
            mock_qc.side_effect = [
                QualityReport(doc_id="test", pages=5, score=0.6, blocks={},
                              warnings=[Warning(severity="warn", code="test", message="x")]),
                QualityReport(doc_id="test", pages=5, score=0.3, blocks={},
                              warnings=[Warning(severity="warn", code="test", message="x")]),
            ]

            with patch("doc_rag.raglib.unstructured_backend.parse_pdf_unstructured") as mock_unstructured:
                bad_blocks = [Block(f"u{i}", "test", "paragraph", "bad", "unstructured", page=i) for i in range(5)]
                mock_unstructured.return_value = ("bad", bad_blocks, {"pages": 5})

                with patch("doc_rag.raglib.parsers.detect_supported_extension") as mock_ext:
                    mock_ext.return_value = ".pdf"

                    from doc_rag.raglib.parsers import parse_document
                    result = parse_document(
                        {"parsing": {"quality_fallback_threshold": 0.5}},
                        str(tmp_path / "test.pdf"),
                    )
                    assert result["blocks"][0].source_backend == "docling"
