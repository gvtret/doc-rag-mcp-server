"""Tests for pipeline postprocessing: _postprocess_chunks."""

from __future__ import annotations

from doc_rag.raglib.pipeline import _postprocess_chunks


def test_postprocess_removes_short_chunks():
    """Chunks shorter than min_text_len should be removed."""
    chunks = [
        {"text": "short", "chunk_id": "a:0"},
        {"text": "a" * 25, "chunk_id": "a:1"},
    ]
    result = _postprocess_chunks(chunks, min_text_len=20)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "a:1"


def test_postprocess_removes_duplicate_headers():
    """Short text repeated in 3+ chunks should be stripped."""
    # Duplicate detection checks entire chunk text (3-100 chars)
    # But min_text_len=20 removes short chunks
    # So we need texts between 20-100 chars to test this
    chunks = [
        {"text": "ISO 13239 Header Line One Two", "chunk_id": "a:0"},
        {"text": "ISO 13239 Header Line One Two", "chunk_id": "a:1"},
        {"text": "ISO 13239 Header Line One Two", "chunk_id": "a:2"},
        {"text": "Unique Header Line One Two Three", "chunk_id": "a:3"},
    ]
    result = _postprocess_chunks(chunks)
    texts = [c["text"] for c in result]
    assert "ISO 13239 Header Line One Two" not in texts
    assert "Unique Header Line One Two Three" in texts


def test_postprocess_deduplicates_identical_chunks():
    """Identical chunk texts should be deduplicated."""
    chunks = [
        {"text": "Same text here repeated enough times", "chunk_id": "a:0"},
        {"text": "Same text here repeated enough times", "chunk_id": "b:0"},
    ]
    result = _postprocess_chunks(chunks)
    assert len(result) == 1


def test_postprocess_cleans_ocr_artifacts():
    """Unicode replacement chars and control chars should be removed."""
    chunks = [
        {"text": "Hello \ufffd world with enough content to pass filter", "chunk_id": "a:0"},
    ]
    result = _postprocess_chunks(chunks)
    assert "\ufffd" not in result[0]["text"]


def test_postprocess_collapses_whitespace():
    """Multiple spaces and newlines should be collapsed."""
    chunks = [
        {"text": "Hello  \n\n\n  world with enough content to pass filter", "chunk_id": "a:0"},
    ]
    result = _postprocess_chunks(chunks)
    assert len(result) == 1
    assert "\n\n\n" not in result[0]["text"]


def test_postprocess_empty_input():
    """Empty list returns empty list."""
    assert _postprocess_chunks([]) == []


def test_postprocess_preserves_content():
    """Good quality chunks should pass through unchanged."""
    chunks = [
        {
            "text": "This is a well-formed paragraph with enough content to pass the filter.",
            "chunk_id": "a:0",
        },
    ]
    result = _postprocess_chunks(chunks)
    assert len(result) == 1
    assert "well-formed" in result[0]["text"]
