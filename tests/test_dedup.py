"""Upload deduplication tests.

The Web UI deduplicates uploads twice:

1. against the existing manifest (a file we have already ingested);
2. against the current incoming queue (a file we are about to ingest).

The first one prevents re-ingesting an already-known document. The
second one prevents the operator from accidentally uploading the same
file twice in one batch and ending up with two manifest entries after
the next ingest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _client(tmp_corpus_root: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    return TestClient(app), tmp_corpus_root


def test_upload_same_payload_twice_in_one_batch_drops_the_second(tmp_corpus_root: Path):
    client, root = _client(tmp_corpus_root)

    payload = b"%PDF-1.4 fixture\nfoo\n"
    multi = [
        ("files", ("doc_one.pdf", payload, "application/pdf")),
        ("files", ("doc_two.pdf", payload, "application/pdf")),
    ]
    r = client.post("/ui/upload", files=multi, follow_redirects=False)

    assert r.status_code == 303
    location = r.headers.get("location", "")
    assert "up_saved=1" in location, location
    assert "up_dup=1" in location, location

    incoming = root / "sources" / "incoming"
    files = sorted(p.name for p in incoming.glob("*.pdf"))
    assert len(files) == 1, files


def test_re_upload_of_archived_file_is_reported_as_duplicate(tmp_corpus_root: Path):
    """Once a file is in the manifest, re-uploading it must be flagged."""
    import json

    client, root = _client(tmp_corpus_root)

    payload = b"%PDF-1.4 archived\nbar\n"
    sha = hashlib.sha256(payload).hexdigest()

    # Pre-seed the manifest as if we'd already ingested this file.
    archived = root / "sources" / "archived"
    archived.mkdir(parents=True, exist_ok=True)
    (archived / "already.pdf").write_bytes(payload)
    manifest = {
        "documents": [
            {
                "doc_id": "already-doc",
                "source_file": "sources/archived/already.pdf",
                "md_path": "build/docs_md/already-doc.md",
                "sha256": sha,
                "chunks": 0,
            }
        ]
    }
    (root / "build" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Now try to upload the same payload under a fresh filename.
    r = client.post(
        "/ui/upload",
        files=[("files", ("again.pdf", payload, "application/pdf"))],
        follow_redirects=False,
    )

    assert r.status_code == 303
    location = r.headers.get("location", "")
    assert "up_dup=1" in location, location
    # When every file is a duplicate, the redirect omits up_saved.
    assert "up_saved=" not in location or "up_saved=0" in location, location
    # Nothing landed in the incoming queue.
    assert list((root / "sources" / "incoming").glob("*.pdf")) == []
    # Manifest is untouched (still exactly one document).
    manifest_after = json.loads((root / "build" / "manifest.json").read_text())
    assert len(manifest_after["documents"]) == 1


def test_distinct_payloads_all_pass_through(tmp_corpus_root: Path):
    """Sanity: dedup must not be over-eager."""
    client, root = _client(tmp_corpus_root)

    files = [
        ("files", (f"d{i}.pdf", f"%PDF-1.4 payload-{i}\n".encode(), "application/pdf"))
        for i in range(3)
    ]
    r = client.post("/ui/upload", files=files, follow_redirects=False)

    assert r.status_code == 303
    assert "up_saved=3" in r.headers.get("location", "")
    assert len(list((root / "sources" / "incoming").glob("*.pdf"))) == 3
