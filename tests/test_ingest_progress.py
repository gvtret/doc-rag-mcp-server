"""Unit tests for the per-document progress parser used by /ui/status.

`_apply_progress_from_line` reads pipeline `_log` lines and updates
`_INGEST_STATE` so the UI can show "currently parsing X · n/N · ETA".
The parser is a regex over existing log markers; this suite pins the
markers it relies on so a future log-format tweak in pipeline.py
cannot silently break the UI."""

from __future__ import annotations

import time

import pytest

from doc_rag.server import mcp_http


@pytest.fixture(autouse=True)
def _reset_state():
    mcp_http._reset_progress_state()
    mcp_http._INGEST_STATE["running"] = False
    mcp_http._INGEST_STATE["last_started"] = None
    yield
    mcp_http._reset_progress_state()
    mcp_http._INGEST_STATE["running"] = False
    mcp_http._INGEST_STATE["last_started"] = None


def test_found_line_sets_total():
    mcp_http._apply_progress_from_line(
        "2026-06-05T10:00:00 INFO  ingest: found 7 file(s) in sources/incoming"
    )
    assert mcp_http._INGEST_STATE["docs_total"] == 7


def test_rebuild_two_passes_accumulate_total():
    """rebuild() emits one 'found N file(s)' line per pass."""
    mcp_http._apply_progress_from_line("INFO ingest: found 5 file(s) in sources/archived")
    mcp_http._apply_progress_from_line("INFO ingest: found 3 file(s) in sources/incoming")
    assert mcp_http._INGEST_STATE["docs_total"] == 8


def test_parse_line_sets_current_doc_basename_only():
    mcp_http._apply_progress_from_line("INFO parse: sources/incoming/sub/report.pdf")
    assert mcp_http._INGEST_STATE["current_doc"] == "report.pdf"
    assert mcp_http._INGEST_STATE["current_doc_started_at"] is not None


def test_ok_line_increments_done_and_clears_current():
    mcp_http._apply_progress_from_line("INFO parse: sources/incoming/a.pdf")
    assert mcp_http._INGEST_STATE["current_doc"] == "a.pdf"

    mcp_http._apply_progress_from_line(
        "INFO ok: sources/incoming/a.pdf -> build/docs_md/a.md (chunks=12)"
    )
    assert mcp_http._INGEST_STATE["docs_done"] == 1
    assert mcp_http._INGEST_STATE["current_doc"] is None
    assert mcp_http._INGEST_STATE["current_doc_started_at"] is None


def test_skip_line_counts_as_done():
    mcp_http._apply_progress_from_line(
        "INFO skip: sources/incoming/dup.pdf (already in manifest by sha256)"
    )
    assert mcp_http._INGEST_STATE["docs_done"] == 1


def test_failed_line_counts_as_done():
    mcp_http._apply_progress_from_line(
        "ERROR failed: sources/incoming/broken.pdf: Unsupported file type"
    )
    assert mcp_http._INGEST_STATE["docs_done"] == 1


def test_unrelated_lines_do_not_perturb_state():
    mcp_http._apply_progress_from_line("INFO dedup: removed 3 near-duplicate chunks")
    mcp_http._apply_progress_from_line("WARN index update skipped: foo")
    mcp_http._apply_progress_from_line("")
    assert mcp_http._INGEST_STATE["docs_done"] == 0
    assert mcp_http._INGEST_STATE["docs_total"] is None
    assert mcp_http._INGEST_STATE["current_doc"] is None


def test_eta_unknown_when_no_doc_finished():
    mcp_http._INGEST_STATE["running"] = True
    mcp_http._INGEST_STATE["last_started"] = time.time() - 5
    mcp_http._INGEST_STATE["docs_total"] = 10
    mcp_http._INGEST_STATE["docs_done"] = 0
    eta_s, eta_h = mcp_http._compute_eta()
    assert eta_s is None
    assert eta_h is None


def test_eta_extrapolates_after_first_done():
    mcp_http._INGEST_STATE["running"] = True
    mcp_http._INGEST_STATE["last_started"] = time.time() - 10.0
    mcp_http._INGEST_STATE["docs_total"] = 5
    mcp_http._INGEST_STATE["docs_done"] = 1
    eta_s, eta_h = mcp_http._compute_eta()
    # avg=10s/doc; 4 remaining → ~40s.
    assert eta_s is not None and 35 < eta_s < 45
    assert eta_h is not None


def test_eta_zero_when_all_done():
    mcp_http._INGEST_STATE["running"] = True
    mcp_http._INGEST_STATE["last_started"] = time.time() - 30.0
    mcp_http._INGEST_STATE["docs_total"] = 3
    mcp_http._INGEST_STATE["docs_done"] = 3
    eta_s, _ = mcp_http._compute_eta()
    assert eta_s == 0.0


def test_eta_none_when_not_running():
    mcp_http._INGEST_STATE["running"] = False
    mcp_http._INGEST_STATE["docs_total"] = 5
    mcp_http._INGEST_STATE["docs_done"] = 2
    assert mcp_http._compute_eta() == (None, None)


def test_format_eta_seconds_buckets():
    assert mcp_http._format_eta_seconds(45) == "45 с"
    assert mcp_http._format_eta_seconds(60) == "1 мин"
    assert mcp_http._format_eta_seconds(72) == "1 мин 12 с"
    assert mcp_http._format_eta_seconds(3600) == "1 ч"
    assert mcp_http._format_eta_seconds(3660) == "1 ч 01 мин"


def test_status_payload_includes_progress_fields():
    mcp_http._INGEST_STATE["running"] = True
    mcp_http._INGEST_STATE["last_started"] = time.time() - 5
    mcp_http._INGEST_STATE["current_doc"] = "x.pdf"
    mcp_http._INGEST_STATE["docs_done"] = 1
    mcp_http._INGEST_STATE["docs_total"] = 4
    payload = mcp_http._ui_status_payload()
    assert payload["current_doc"] == "x.pdf"
    assert payload["docs_done"] == 1
    assert payload["docs_total"] == 4
    assert "eta_seconds" in payload
    assert "eta_human" in payload
