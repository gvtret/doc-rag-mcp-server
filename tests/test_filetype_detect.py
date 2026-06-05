"""Tests for the v1.5 magic-bytes file-type detection helper.

`detect_supported_extension(path)` must prefer the content over the
filename when they disagree, and fall back to the filename only for
plain-text formats that have no magic bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_rag.raglib.filetype_detect import (
    SUPPORTED_EXTENSIONS,
    _diagnostic_summary,
    detect_supported_extension,
    filename_extension_disagrees_with_content,
    is_supported,
)

# --------------------------------------------------------------------------
# Bundled real-content fixtures (preferred — they reflect actual files
# users will drop into sources/incoming).
# --------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SAMPLE_DOCX = _FIXTURES / "sample.docx"
_SAMPLE_DOC = _FIXTURES / "sample.doc"


def test_detect_docx_by_content():
    if not _SAMPLE_DOCX.exists():
        pytest.skip("sample.docx fixture missing")
    assert detect_supported_extension(_SAMPLE_DOCX) == ".docx"


def test_detect_doc_by_content():
    if not _SAMPLE_DOC.exists():
        pytest.skip("sample.doc fixture missing")
    assert detect_supported_extension(_SAMPLE_DOC) == ".doc"


def test_detect_docx_overrides_misleading_extension(tmp_path: Path):
    """Renamed `report.pdf` that is actually a DOCX must be detected as DOCX."""
    if not _SAMPLE_DOCX.exists():
        pytest.skip("sample.docx fixture missing")
    misnamed = tmp_path / "report.pdf"
    misnamed.write_bytes(_SAMPLE_DOCX.read_bytes())

    assert detect_supported_extension(misnamed) == ".docx"
    assert filename_extension_disagrees_with_content(misnamed) is True


def test_detect_doc_overrides_misleading_extension(tmp_path: Path):
    """Renamed `something.docx` that is actually a legacy .doc."""
    if not _SAMPLE_DOC.exists():
        pytest.skip("sample.doc fixture missing")
    misnamed = tmp_path / "something.docx"
    misnamed.write_bytes(_SAMPLE_DOC.read_bytes())

    assert detect_supported_extension(misnamed) == ".doc"
    assert filename_extension_disagrees_with_content(misnamed) is True


# --------------------------------------------------------------------------
# Synthetic content (PDF and minimum-viable magic-bytes test)
# --------------------------------------------------------------------------


def _minimal_pdf_bytes() -> bytes:
    """Tiny PDF that PyMuPDF would refuse, but whose magic header is valid.

    We do not need to be able to parse it — only that
    `filetype.guess()` recognises the `%PDF-` magic.
    """
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\n1 0 obj\n<< >>\nendobj\n"


def test_detect_pdf_by_magic_bytes(tmp_path: Path):
    p = tmp_path / "weird_name.bin"  # no extension
    p.write_bytes(_minimal_pdf_bytes())
    assert detect_supported_extension(p) == ".pdf"


def test_detect_pdf_overrides_misleading_extension(tmp_path: Path):
    """A PDF saved as `note.txt` must still be routed to the PDF parser."""
    p = tmp_path / "note.txt"
    p.write_bytes(_minimal_pdf_bytes())
    assert detect_supported_extension(p) == ".pdf"
    assert filename_extension_disagrees_with_content(p) is True


# --------------------------------------------------------------------------
# Plain-text formats — no magic bytes, fall back to extension.
# --------------------------------------------------------------------------


def test_md_falls_back_to_extension(tmp_path: Path):
    p = tmp_path / "notes.md"
    p.write_text("# heading\n\nbody.\n", encoding="utf-8")
    assert detect_supported_extension(p) == ".md"


def test_txt_falls_back_to_extension(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text("ordinary text\n", encoding="utf-8")
    assert detect_supported_extension(p) == ".txt"


def test_md_and_txt_no_content_extension_mismatch(tmp_path: Path):
    """Plain text has no magic bytes — sniffed == filename, so no flag."""
    md = tmp_path / "x.md"
    md.write_text("hi", encoding="utf-8")
    assert filename_extension_disagrees_with_content(md) is False

    txt = tmp_path / "y.txt"
    txt.write_text("hi", encoding="utf-8")
    assert filename_extension_disagrees_with_content(txt) is False


# --------------------------------------------------------------------------
# Unsupported / non-existent inputs
# --------------------------------------------------------------------------


def test_unsupported_format_returns_none(tmp_path: Path):
    """A real PNG should be reported as unsupported, not silently mapped."""
    # Minimal PNG header.
    p = tmp_path / "picture.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    assert detect_supported_extension(p) is None
    assert is_supported(p) is False


def test_missing_file_returns_none(tmp_path: Path):
    assert detect_supported_extension(tmp_path / "ghost.pdf") is None


def test_unknown_zip_is_not_docx(tmp_path: Path):
    """A plain ZIP with no `word/document.xml` must not be mis-detected
    as DOCX."""
    import zipfile

    p = tmp_path / "archive.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "hello")
    assert detect_supported_extension(p) is None


def test_zip_with_word_document_xml_is_docx(tmp_path: Path):
    """The DOCX path: ZIP container with `word/document.xml` inside."""
    import zipfile

    p = tmp_path / "fake_doc.zip"  # extension wrong on purpose
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("word/document.xml", "<?xml version='1.0'?><doc/>")
        zf.writestr("_rels/.rels", "<?xml version='1.0'?><rels/>")
    assert detect_supported_extension(p) == ".docx"


# --------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------


def test_supported_extensions_is_the_canonical_five():
    assert SUPPORTED_EXTENSIONS == (".pdf", ".docx", ".doc", ".md", ".txt")


def test_diagnostic_summary_shape(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_bytes(_minimal_pdf_bytes())
    summary = _diagnostic_summary(p)
    assert summary["filename_extension"] == ".pdf"
    assert summary["filetype_mime"] == "application/pdf"
    assert summary["resolved_extension"] == ".pdf"
