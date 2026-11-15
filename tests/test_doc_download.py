"""Task 15: doc_download size limit + manifest-first resolution."""

from __future__ import annotations

import base64
import json

import pytest


@pytest.fixture
def sources_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Corpus root with config.yaml + sources/archived, DOC_RAG_ROOT set."""
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "sources" / "archived").mkdir(parents=True)
    (tmp_path / "sources" / "incoming").mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)
    cfg = {
        "paths": {"sources_archived": "sources/archived", "sources_incoming": "sources/incoming"}
    }
    (tmp_path / "config" / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    return tmp_path


def _tool(filename: str):
    from doc_rag.server.mcp_http import _doc_download_tool

    return _doc_download_tool({"filename": filename})


def test_doc_download_ok_small_file(sources_root) -> None:
    p = sources_root / "sources" / "archived" / "small.bin"
    p.write_bytes(b"hello")
    out = _tool("small.bin")
    assert out[0]["type"] == "file"
    assert base64.b64decode(out[0]["file"]["data"]) == b"hello"
    assert out[0]["file"]["name"] == "small.bin"


def test_doc_download_oversize_rejected(sources_root) -> None:
    """A file over the 25 MB limit must be rejected by text, not read
    into memory and base64-encoded (~33 MB payload)."""
    big = sources_root / "sources" / "archived" / "big.bin"
    # 25 MB limit is inclusive of the file size check: write limit + 1 byte.
    big.write_bytes(b"x" * (25 * 1024 * 1024 + 1))
    out = _tool("big.bin")
    assert out[0]["type"] == "text"
    assert out[0]["text"] == "Error: file exceeds 25 MB limit"


def test_doc_download_at_limit_allowed(sources_root) -> None:
    """Exactly 25 MB is allowed (limit is '>', not '>=')."""
    p = sources_root / "sources" / "archived" / "edge.bin"
    p.write_bytes(b"y" * (25 * 1024 * 1024))
    out = _tool("edge.bin")
    assert out[0]["type"] == "file"


def test_doc_download_unlisted_file_fallback(sources_root) -> None:
    """A file present in sources/ but NOT in the manifest stays reachable
    via the rglob fallback scan."""
    p = sources_root / "sources" / "incoming" / "unlisted.pdf"
    p.write_bytes(b"%PDF-fallback")
    out = _tool("unlisted.pdf")
    assert out[0]["type"] == "file"
    assert base64.b64decode(out[0]["file"]["data"]) == b"%PDF-fallback"


def test_doc_download_resolves_via_manifest(sources_root) -> None:
    """A filename the rglob scan would miss is resolved through the
    manifest source_file (exact path under sources/)."""
    # Real file lives in a subdirectory; the manifest points at it.
    subdir = sources_root / "sources" / "archived" / "nested"
    subdir.mkdir(parents=True)
    real = subdir / "mani.bin"
    real.write_bytes(b"from-manifest")
    # A different file with the same basename elsewhere: manifest must win.
    decoy = sources_root / "sources" / "incoming" / "mani.bin"
    decoy.write_bytes(b"decoy")

    (sources_root / "build" / "manifest.json").write_text(
        json.dumps(
            {"documents": [{"doc_id": "d1", "source_file": "sources/archived/nested/mani.bin"}]}
        ),
        encoding="utf-8",
    )
    out = _tool("mani.bin")
    assert out[0]["type"] == "file"
    assert base64.b64decode(out[0]["file"]["data"]) == b"from-manifest"


def test_doc_download_manifest_present_but_file_not_listed(sources_root) -> None:
    """Variant (c): the manifest EXISTS, but the requested file is not in
    it -- the rglob fallback must still find and serve it."""
    (sources_root / "build" / "manifest.json").write_text(
        json.dumps({"documents": [{"doc_id": "d1", "source_file": "sources/archived/other.pdf"}]}),
        encoding="utf-8",
    )
    p = sources_root / "sources" / "incoming" / "unlisted.pdf"
    p.write_bytes(b"%PDF-not-in-manifest")
    out = _tool("unlisted.pdf")
    assert out[0]["type"] == "file"
    assert base64.b64decode(out[0]["file"]["data"]) == b"%PDF-not-in-manifest"
