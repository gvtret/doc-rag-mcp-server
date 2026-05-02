import pytest

from doc_rag.raglib.edition_year import resolve_edition_year


def test_resolve_by_basename():
    cfg = {
        "parsing": {
            "edition_year": {
                "by_basename": {"manual.pdf": 2019},
                "from_pdf_metadata": False,
            }
        }
    }
    y = resolve_edition_year(
        cfg,
        abs_path="/x/sources/incoming/manual.pdf",
        rel_path="sources/incoming/manual.pdf",
        sha256_hex="ab" * 32,
    )
    assert y == 2019


def test_resolve_by_sha256():
    hx = "a" * 64
    cfg = {"parsing": {"edition_year": {"by_sha256": {hx: 2024}, "from_pdf_metadata": False}}}
    y = resolve_edition_year(
        cfg,
        abs_path="/x/doc.pdf",
        rel_path="doc.pdf",
        sha256_hex=hx,
    )
    assert y == 2024


def test_filename_regex():
    cfg = {
        "parsing": {
            "edition_year": {
                "filename_regex": r"(?P<year>20\d{2})",
                "from_pdf_metadata": False,
            }
        }
    }
    y = resolve_edition_year(
        cfg,
        abs_path="/tmp/GOST_2021_final.pdf",
        rel_path="GOST_2021_final.pdf",
        sha256_hex="b" * 64,
    )
    assert y == 2021


def test_parse_document_coverage_ocr_and_native(tmp_path):
    pytest.importorskip("docx")
    from docx import Document

    from doc_rag.raglib.parsers import parse_document

    p = tmp_path / "sample.docx"
    d = Document()
    d.add_paragraph("Hello coverage")
    d.save(str(p))

    cfg = {"parsing": {"normalize_whitespace": True, "min_chars_per_page": 20}}
    out = parse_document(cfg, str(p))
    st = out["stats"]
    assert st["ocr"]["applied"] is False
    assert st["ocr"]["before_ocr"]["chars"] is None
    assert st["ocr"]["after_ocr"]["chars"] is None
    assert st["native_text_extraction"]["before_normalize"]["chars"] > 0
    assert st["native_text_extraction"]["after_normalize"]["chars"] > 0
    assert st["native_text_extraction"]["markdown"]["chars"] > 0

