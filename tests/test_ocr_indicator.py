"""Tests for the per-document OCR-applied indicator (v2.0+).

The pipeline: Docling's `ConversionResult.confidence.pages[i].ocr_score`
is the canonical "did OCR fire on this page?" signal. A non-NaN positive
value means RapidOCR produced text for that page; NaN means the page had
a native text layer and OCR was skipped.

We aggregate that signal at three layers:

  1. `_ocr_signal_from_confidence` (docling_backend.py): page-level →
     dict of {ocr_pages_count, ocr_pages, ocr_mean_score}.
  2. `_build_ocr_stats_block` (parsers.py): the aggregate → the
     `stats.ocr` shape that downstream callers (manifest, UI) already
     read.
  3. `_ocr_badge_html` (mcp_http.py): coverage dict from a manifest
     entry → inline `<span class="ocr-badge">OCR</span>` for the UI
     document table, or empty string when OCR did not fire.
"""

from __future__ import annotations

import math
from typing import Any

from doc_rag.raglib.docling_backend import _ocr_signal_from_confidence
from doc_rag.raglib.parsers import _build_ocr_stats_block


class _StubPageScores:
    """Minimal stand-in for Docling's per-page confidence scores."""

    def __init__(self, ocr_score: float) -> None:
        self.ocr_score = ocr_score


class _StubConfidence:
    """Minimal stand-in for Docling's ConfidenceReport."""

    def __init__(self, pages: dict[int, _StubPageScores]) -> None:
        self.pages = pages


# --------------------------------------------------------------------------
# Layer 1 — docling_backend._ocr_signal_from_confidence
# --------------------------------------------------------------------------


def test_no_confidence_returns_empty_signal():
    assert _ocr_signal_from_confidence(None) == {}


def test_all_pages_native_text_returns_empty_signal():
    """When every page has a text layer, ocr_score is NaN → no signal."""
    pages = {1: _StubPageScores(math.nan), 2: _StubPageScores(math.nan)}
    assert _ocr_signal_from_confidence(_StubConfidence(pages)) == {}


def test_mixed_pages_count_only_ocr_pages():
    pages = {
        1: _StubPageScores(math.nan),
        2: _StubPageScores(0.95),
        3: _StubPageScores(0.85),
        4: _StubPageScores(math.nan),
    }
    sig = _ocr_signal_from_confidence(_StubConfidence(pages))
    assert sig["ocr_pages_count"] == 2
    assert sig["ocr_pages"] == [2, 3]
    assert sig["ocr_mean_score"] == 0.9


def test_all_pages_ocr():
    pages = {1: _StubPageScores(0.97), 2: _StubPageScores(0.96), 3: _StubPageScores(0.98)}
    sig = _ocr_signal_from_confidence(_StubConfidence(pages))
    assert sig["ocr_pages_count"] == 3
    assert sig["ocr_pages"] == [1, 2, 3]
    assert 0.96 < sig["ocr_mean_score"] < 0.98


def test_dict_shaped_pages_also_accepted():
    """Some Docling versions hand back plain dicts instead of pydantic models."""
    pages = {1: {"ocr_score": 0.9}, 2: {"ocr_score": math.nan}}
    sig = _ocr_signal_from_confidence(_StubConfidence(pages))
    assert sig["ocr_pages_count"] == 1
    assert sig["ocr_pages"] == [1]


# --------------------------------------------------------------------------
# Layer 2 — parsers._build_ocr_stats_block
# --------------------------------------------------------------------------


def test_build_ocr_block_without_native_keeps_legacy_shape():
    """No native stats dict → backwards-compatible empty block."""
    block = _build_ocr_stats_block()
    assert block["applied"] is False
    assert block["pages_recognized"] == 0
    assert block["before_ocr"] == {"chars": None}
    assert block["after_ocr"] == {"chars": None}


def test_build_ocr_block_when_signal_absent_stays_negative():
    """Docling stats without ocr_pages_count → still 'OCR did not fire'."""
    native: dict[str, Any] = {"pages": 5, "blocks_by_type": {"paragraph": 12}}
    block = _build_ocr_stats_block(native)
    assert block["applied"] is False
    assert block["pages_recognized"] == 0


def test_build_ocr_block_when_signal_present_reports_truth():
    native: dict[str, Any] = {
        "pages": 5,
        "ocr_pages_count": 3,
        "ocr_pages": [1, 2, 3],
        "ocr_mean_score": 0.94,
    }
    block = _build_ocr_stats_block(native)
    assert block["applied"] is True
    assert block["pages_recognized"] == 3
    assert block["confidence"] == 0.94
    assert block["ocr_pages"] == [1, 2, 3]


# --------------------------------------------------------------------------
# Layer 3 — mcp_http._ocr_badge_html
# --------------------------------------------------------------------------


def test_badge_empty_when_no_coverage():
    from doc_rag.server.mcp_http import _ocr_badge_html

    assert _ocr_badge_html(None) == ""
    assert _ocr_badge_html({}) == ""


def test_badge_empty_when_ocr_not_applied():
    from doc_rag.server.mcp_http import _ocr_badge_html

    cov = {"ocr": {"applied": False, "pages_recognized": 0}}
    assert _ocr_badge_html(cov) == ""


def test_badge_renders_when_ocr_applied_with_tooltip():
    from doc_rag.server.mcp_http import _ocr_badge_html

    cov = {"ocr": {"applied": True, "pages_recognized": 3, "confidence": 0.97}}
    html = _ocr_badge_html(cov)
    assert "ocr-badge" in html
    assert ">OCR<" in html
    assert "страниц: 3" in html
    assert "0.97" in html


def test_badge_omits_unknown_fields_gracefully():
    from doc_rag.server.mcp_http import _ocr_badge_html

    cov = {"ocr": {"applied": True}}
    html = _ocr_badge_html(cov)
    assert ">OCR<" in html
    assert "страниц:" not in html
    assert "уверенность" not in html
