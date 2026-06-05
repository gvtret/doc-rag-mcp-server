"""Parser tests for every supported source format.

`parse_document(cfg, path)` is the single public entry point; we test
its behaviour for `.md`, `.txt`, `.docx`, `.doc`, and `.pdf`. Since
v2.0 the PDF backend is Docling-only, so PDF tests monkeypatch
`parse_pdf_docling` instead of synthesizing real PDFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.parsers import parse_document

_MIN_CFG: dict[str, Any] = {
    "parsing": {
        "pdf_backend": "auto",
        "normalize_whitespace": True,
        "min_chars_per_page": 1,
    },
}


def test_parse_md_keeps_blank_lines_and_emits_markdown(make_md):
    body = "# Heading\n\nfirst paragraph.\n\nsecond paragraph.\n"
    src = make_md("intro.md", body)

    result = parse_document(_MIN_CFG, str(src))

    md = result["markdown"]
    assert md.startswith("# intro.md\n\n")
    assert "first paragraph." in md
    assert "second paragraph." in md
    # Blank lines must be preserved in .md (they are semantic).
    assert "first paragraph.\n\nsecond paragraph." in md.replace("\r\n", "\n")
    assert result["stats"]["ocr"]["applied"] is False


def test_parse_txt_keeps_blank_lines(make_txt):
    body = "alpha\n\nbeta\n\ngamma\n"
    src = make_txt("notes.txt", body)

    result = parse_document(_MIN_CFG, str(src))

    md = result["markdown"]
    assert "alpha" in md and "beta" in md and "gamma" in md
    # Blank lines preserved for .txt too.
    assert "alpha\n\nbeta\n\ngamma" in md.replace("\r\n", "\n")


def test_parse_docx_emits_text_from_paragraphs(make_docx):
    src = make_docx("report.docx", ["Lead paragraph.", "Follow-up paragraph."])

    result = parse_document(_MIN_CFG, str(src))

    md = result["markdown"]
    assert "Lead paragraph." in md
    assert "Follow-up paragraph." in md


def test_parse_docx_includes_table_cells(make_docx):
    src = make_docx(
        "with_table.docx",
        ["Caption above the table."],
        table_rows=[["A1", "B1"], ["A2", "B2"]],
    )

    result = parse_document(_MIN_CFG, str(src))

    md = result["markdown"]
    assert "Caption above the table." in md
    # Cells must end up in the extracted text in some form.
    for cell in ("A1", "B1", "A2", "B2"):
        assert cell in md, f"missing cell {cell} in markdown"


def test_parse_unsupported_extension_raises(tmp_path: Path):
    bad = tmp_path / "unknown.xyz"
    bad.write_text("anything", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported"):
        parse_document(_MIN_CFG, str(bad))


# --------------------------------------------------------------------------
# v1.5 — parse_document also returns a `blocks` list
# --------------------------------------------------------------------------


def _expected_backend(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf": "docling",
        ".docx": "python-docx",
        ".doc": "antiword",
        ".md": "direct",
        ".txt": "direct",
    }[suffix]


def test_parse_document_includes_blocks_for_md(make_md):
    src = make_md("notes.md", "alpha\n\nbeta\n")
    result = parse_document(_MIN_CFG, str(src))

    assert "blocks" in result, "parse_document must expose a blocks key (v1.5)"
    blocks: list[Block] = result["blocks"]
    assert all(isinstance(b, Block) for b in blocks)
    assert len(blocks) == 1
    only = blocks[0]
    assert only.type == "paragraph"
    assert only.source_backend == _expected_backend(src)
    # The baseline block carries the same text the markdown derivation uses,
    # so structure-aware downstream code in v1.9 can be migrated incrementally.
    assert "alpha" in only.text
    assert "beta" in only.text


def test_parse_document_includes_blocks_for_txt(make_txt):
    src = make_txt("notes.txt", "first\n\nsecond\n")
    result = parse_document(_MIN_CFG, str(src))

    blocks: list[Block] = result["blocks"]
    assert len(blocks) == 1
    assert blocks[0].source_backend == "direct"
    assert "first" in blocks[0].text
    assert "second" in blocks[0].text


def test_parse_document_includes_blocks_for_docx(make_docx):
    src = make_docx("note.docx", ["Lead paragraph.", "Follow-up paragraph."])
    result = parse_document(_MIN_CFG, str(src))

    blocks: list[Block] = result["blocks"]
    assert len(blocks) == 1
    assert blocks[0].source_backend == "python-docx"
    assert "Lead paragraph." in blocks[0].text


def test_parse_document_blocks_use_tmp_block_id_prefix(make_md):
    """Until the pipeline assigns a real doc_id, parsers emit `tmp:NNNN`."""
    src = make_md("any.md", "content\n")
    result = parse_document(_MIN_CFG, str(src))

    blocks: list[Block] = result["blocks"]
    assert blocks[0].block_id.startswith("tmp:")
    assert blocks[0].doc_id == "tmp"


def test_parse_document_blocks_text_matches_markdown_body(make_md):
    """Markdown derivation in #54 will route via blocks; assert today's
    raw equivalence so the future refactor cannot silently drift."""
    body = "alpha line\n\nbeta line"
    src = make_md("eq.md", body)
    result = parse_document(_MIN_CFG, str(src))

    md_body = result["markdown"].split("\n\n", 1)[1].rstrip("\n")
    blocks_text = result["blocks"][0].text
    assert blocks_text == md_body


def test_parse_document_empty_input_yields_zero_blocks(make_txt):
    src = make_txt("blank.txt", "")
    result = parse_document(_MIN_CFG, str(src))
    assert result["blocks"] == []


def test_parse_document_routes_misnamed_docx_by_content(tmp_path: Path):
    """v1.5: a .docx renamed to .pdf must be parsed via python-docx,
    not by the PDF parser. This is the canonical magic-bytes case."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "sample.docx"
    if not fixture.exists():
        pytest.skip("sample.docx fixture missing")

    # Drop a DOCX into tmp_path under a misleading `.pdf` filename.
    misnamed = tmp_path / "report.pdf"
    misnamed.write_bytes(fixture.read_bytes())

    result = parse_document(_MIN_CFG, str(misnamed))

    # The sample.docx is a known fixture containing "АВ-12" and a
    # registration code (see fixture invariants block below). If we had
    # routed this through the PDF parser, we'd get noise or an error
    # rather than the document's real text.
    assert "АВ-12" in result["markdown"]
    assert result["blocks"][0].source_backend == "python-docx"


# --------------------------------------------------------------------------
# Bundled rich-document fixture
#
# tests/fixtures/sample.docx is a committed, real .docx file. The .doc
# counterpart is the same content re-saved as legacy Word 97-2003.
# Both are maintained by hand — to edit, open in Word or LibreOffice
# Writer, make the change, and re-export. Keep the markers below intact.
#
# Required content invariants (the tests assert these):
#   - several Heading 1 / Heading 2 levels
#   - plain paragraphs, with at least one bold and one italic run
#   - a bulleted list and a numbered list
#   - a real 4×3 table; cell value "АС-22А" and "230 В" must be present
#   - one embedded image (not asserted on, but documents the test scope)
#   - the string "АВ-12" appears at least once
#   - the registration code "РК-22.04-2024" appears at least once
# --------------------------------------------------------------------------


_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SAMPLE_DOCX = _FIXTURES / "sample.docx"
_SAMPLE_DOC = _FIXTURES / "sample.doc"

# Markers that appear in sample.docx / sample.doc:
_MARKER_SERIES = "АВ-12"
_MARKER_REGCODE = "РК-22.04-2024"
# A table cell value:
_MARKER_TABLE_CELL = "АС-22А"


def test_parse_docx_bundled_fixture_text_and_table():
    if not _SAMPLE_DOCX.exists():
        pytest.skip(f"missing fixture: {_SAMPLE_DOCX}")

    result = parse_document(_MIN_CFG, str(_SAMPLE_DOCX))
    md = result["markdown"]

    # Body text and the registration code marker are present.
    assert _MARKER_SERIES in md
    assert _MARKER_REGCODE in md
    # Table content is extracted alongside paragraphs.
    assert _MARKER_TABLE_CELL in md
    # The numeric voltage value from the table:
    assert "230 В" in md


def test_parse_doc_bundled_fixture(antiword_available: bool):
    """If a sample.doc fixture is present and antiword is installed, parse it."""
    if not _SAMPLE_DOC.exists():
        pytest.skip(f"missing fixture: {_SAMPLE_DOC} (convert sample.docx → .doc to enable)")
    if not antiword_available:
        pytest.skip("neither antiword nor catdoc on PATH")

    result = parse_document(_MIN_CFG, str(_SAMPLE_DOC))
    md = result["markdown"]

    # antiword does not preserve formatting, but the running text and the
    # registration code marker must survive.
    assert _MARKER_SERIES in md
    assert _MARKER_REGCODE in md


# --------------------------------------------------------------------------
# .pdf — Docling backend (v2.0+). PDF tests mock parse_pdf_docling so
# they stay hermetic and fast; the real Docling integration is exercised
# in tests/test_docling_backend.py with proper backend fixtures.
# --------------------------------------------------------------------------


def _make_pdf_with_magic_bytes(tmp_path: Path) -> Path:
    """Write a minimal valid PDF header so filetype detection routes the
    file to the PDF branch; content is then served by the mocked Docling
    backend."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    return path


def _stub_parse_pdf_docling(text: str, *, source_backend: str = "docling"):
    """Return a stand-in for parse_pdf_docling that yields the given text
    as a single paragraph block plus minimal stats."""

    def _stub(path: str):
        block = Block(
            block_id="tmp:0000",
            doc_id="tmp",
            type="paragraph",
            text=text,
            source_backend=source_backend,
        )
        stats = {"format": "pdf", "pdf_backend": "docling"}
        return text, [block], stats

    return _stub


def test_parse_pdf_text(tmp_path: Path, monkeypatch):
    marker = "pdf-marker-9f8c2"
    body = f"doc-rag pdf parser test\n{marker}\ntrailing line"
    path = _make_pdf_with_magic_bytes(tmp_path)

    monkeypatch.setattr(
        "doc_rag.raglib.docling_backend.parse_pdf_docling",
        _stub_parse_pdf_docling(body),
    )

    result = parse_document(_MIN_CFG, str(path))

    assert marker in result["markdown"]
    assert result["stats"]["ocr"]["applied"] is False
    assert result["blocks"][0].source_backend == "docling"


def test_parse_pdf_stats_carry_ocr_block(tmp_path: Path, monkeypatch):
    """v2.0: the stats dict still surfaces an `ocr` block (shape kept for
    manifest schema v1 compatibility) even though Docling runs OCR
    internally via RapidOCR and we no longer maintain a Tesseract path."""
    path = _make_pdf_with_magic_bytes(tmp_path)
    monkeypatch.setattr(
        "doc_rag.raglib.docling_backend.parse_pdf_docling",
        _stub_parse_pdf_docling("just some text"),
    )

    result = parse_document(_MIN_CFG, str(path))
    ocr = result["stats"]["ocr"]

    assert ocr["applied"] is False
    assert ocr["pages_recognized"] == 0


def test_parse_pdf_rejects_legacy_pymupdf_backend(tmp_path: Path):
    """v2.0: configs that still set pdf_backend: pymupdf must fail loudly."""
    path = _make_pdf_with_magic_bytes(tmp_path)
    cfg = {
        "parsing": {
            "pdf_backend": "pymupdf",
            "normalize_whitespace": True,
            "min_chars_per_page": 1,
        },
    }

    with pytest.raises(RuntimeError, match="pdf_backend"):
        parse_document(cfg, str(path))


def test_parse_pdf_rejects_legacy_pypdf2_backend(tmp_path: Path):
    """Same gate covers pypdf2."""
    path = _make_pdf_with_magic_bytes(tmp_path)
    cfg = {
        "parsing": {
            "pdf_backend": "pypdf2",
            "normalize_whitespace": True,
            "min_chars_per_page": 1,
        },
    }

    with pytest.raises(RuntimeError, match="pdf_backend"):
        parse_document(cfg, str(path))
