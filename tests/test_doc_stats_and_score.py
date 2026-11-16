"""Regression tests for /ui/doc-stats file-size resolution and quality score.

Two reported defects:

1. UI Statistics showed "0 B" for most files: the endpoint resolved sizes
   only by basename in the *root* of sources/archived and sources/incoming,
   missing files stored in subdirectories (dnp3/, iec61850/, ...) and files
   whose manifest `source_file` points at a path that still exists.
2. Documents with many small warnings saturated the absolute penalty scale
   and showed score 0.00 even when extraction quality was mediocre, not
   catastrophic (e.g. 23 warnings over a 1598-block GOST scan).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag.raglib.quality import Warning, _compute_score


@pytest.fixture
def subcorpus(built_corpus):
    """A corpus whose manifest references a file in a subdirectory of archived/.

    doc-aaa keeps its root-level source; doc-bbb is "moved" to
    sources/archived/iec61850/bbb.pdf (file on disk there, manifest updated),
    plus a third manifest-only doc whose file is gone entirely.
    """
    root, doc_ids = built_corpus
    sub = root / "sources" / "archived" / "iec61850"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "bbb.pdf").write_bytes(b"%PDF-1.4 subdir stub\n")

    manifest_path = root / "build" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for d in manifest["documents"]:
        if d["doc_id"] == "doc-bbb":
            d["source_file"] = "sources/archived/iec61850/bbb.pdf"
    # Manifest-only entry: no file anywhere on disk.
    manifest["documents"].append(
        {
            "doc_id": "doc-gone",
            "source_file": "sources/archived/gone.pdf",
            "md_path": "build/docs_md/doc-gone.md",
            "sha256": "f" * 64,
            "chunks": 0,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    return root


class TestDocStatsSizeResolution:
    def _sizes(self, root: Path) -> dict[str, int | None]:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        import os

        os.environ["DOC_RAG_ROOT"] = str(root)
        from doc_rag.server.mcp_http import app

        c = TestClient(app)
        r = c.get("/ui/doc-stats")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        docs = {d["doc_id"]: d["size_bytes"] for d in data["documents"]}
        os.environ.pop("DOC_RAG_ROOT", None)
        return docs

    def test_root_level_file_size(self, subcorpus):
        """Root-level archived file keeps its real size."""
        root = subcorpus
        sizes = self._sizes(root)
        assert sizes["doc-aaa"] == len(b"%PDF-1.4 stub\n")

    def test_subdirectory_file_size(self, subcorpus):
        """A file stored under sources/archived/<subdir>/ is found by its
        manifest source_file (previously reported as 0 bytes)."""
        root = subcorpus
        sizes = self._sizes(root)
        assert sizes["doc-bbb"] == len(b"%PDF-1.4 subdir stub\n")

    def test_missing_file_reports_null_not_zero(self, subcorpus):
        """A manifest entry whose source file is gone reports null (UI shows
        a dash) instead of a misleading 0 bytes."""
        root = subcorpus
        sizes = self._sizes(root)
        assert sizes["doc-gone"] is None


class TestComputeScoreNormalization:
    def test_many_warns_do_not_saturate_on_large_doc(self):
        """23 warn + 1 info over 1598 blocks (the GOST 870-5-101 case)
        must not clamp to 0.0 anymore."""
        warnings = [Warning(severity="warn", code=f"t{i}", message=f"w {i}") for i in range(22)]
        warnings.append(Warning(severity="info", code="p", message="empty page"))
        score = _compute_score(warnings, 1598)
        assert score > 0.0

    def test_small_doc_still_penalized(self):
        """A tiny document with many warnings still saturates: penalties
        relative to size stay meaningful."""
        warnings = [Warning(severity="error", code=f"e{i}", message=f"e {i}") for i in range(20)]
        assert _compute_score(warnings, 10) == 0.0

    def test_single_warn_small_doc(self):
        """One warning on a small document gives a small, non-zero penalty."""
        w = Warning(severity="warn", code="t", message="w")
        score = _compute_score([w], 10)
        assert 0.0 < score < 1.0

    def test_no_warnings_perfect(self):
        assert _compute_score([], 10) == 1.0

    def test_empty_blocks_perfect(self):
        assert _compute_score([], 0) == 1.0
