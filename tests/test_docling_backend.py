"""Tests for the v1.5 Docling backend.

Docling itself is a heavy optional dependency (~600 MB + ~300 MB of ML
models). These tests do not install it. Instead they:

  1. Verify that an actionable error is raised when `[docling]` is
     missing.
  2. Exercise the `_docling_doc_to_blocks` translation against fake
     items that mimic Docling's `DocItem` shape.
  3. Cover the public entry point through a monkey-patched converter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from doc_rag.raglib import docling_backend as db
from doc_rag.raglib.docling_backend import (
    _docling_doc_to_blocks,
    _docling_stats,
    parse_pdf_docling,
    reset_converter_cache,
)

# --------------------------------------------------------------------------
# Fakes that imitate the shape of Docling DocItem objects.
# --------------------------------------------------------------------------


@dataclass
class _Label:
    value: str


@dataclass
class _BBox:
    l: float = 0.0  # noqa: E741 — mirrors Docling's attribute name
    t: float = 0.0
    r: float = 0.0
    b: float = 0.0


@dataclass
class _Prov:
    page_no: int = 1
    bbox: _BBox = field(default_factory=_BBox)


@dataclass
class _TextItem:
    text: str
    label: _Label
    level: int = 0
    prov: list[_Prov] = field(default_factory=list)


@dataclass
class _Cell:
    text: str
    start_row_offset_idx: int
    start_col_offset_idx: int


@dataclass
class _TableData:
    grid_cells: list[_Cell]


@dataclass
class _TableItem:
    label: _Label
    data: _TableData
    prov: list[_Prov] = field(default_factory=list)


@dataclass
class _PictureItem:
    label: _Label
    captions: list[_TextItem]
    prov: list[_Prov] = field(default_factory=list)
    text: str = ""


class _FakeDoc:
    def __init__(self, items: list[tuple[Any, int]], *, pages: int = 1) -> None:
        self._items = items
        self.pages = [None] * pages

    def iterate_items(self):
        return iter(self._items)

    def export_to_markdown(self) -> str:
        return "# fake-doc\n\n" + "\n\n".join(
            getattr(it, "text", "") for it, _ in self._items if getattr(it, "text", "")
        )


# --------------------------------------------------------------------------
# Translation: DoclingDocument → list[Block]
# --------------------------------------------------------------------------


def test_heading_becomes_heading_block():
    item = _TextItem(text="Section 1", label=_Label("section_header"), level=1)
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert len(blocks) == 1
    assert blocks[0].type == "heading"
    assert blocks[0].text == "Section 1"
    assert blocks[0].level == 1
    assert blocks[0].source_backend == "docling"


def test_paragraph_becomes_paragraph_block():
    item = _TextItem(text="body of the section.", label=_Label("paragraph"))
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].type == "paragraph"
    assert blocks[0].text == "body of the section."


def test_empty_text_paragraph_is_skipped():
    item = _TextItem(text="", label=_Label("paragraph"))
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))
    assert blocks == []


def test_list_item_carries_indentation_level():
    nested = _TextItem(text="nested", label=_Label("list_item"))
    blocks = _docling_doc_to_blocks(_FakeDoc([(nested, 2)]))

    assert blocks[0].type == "list_item"
    assert blocks[0].level == 1


def test_table_with_grid_cells_yields_text_and_metadata():
    cells = [
        _Cell("A1", 0, 0),
        _Cell("B1", 0, 1),
        _Cell("A2", 1, 0),
        _Cell("B2", 1, 1),
    ]
    item = _TableItem(label=_Label("table"), data=_TableData(grid_cells=cells))

    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].type == "table"
    assert blocks[0].text == "A1 | B1\nA2 | B2"
    assert blocks[0].metadata["cells"] == [["A1", "B1"], ["A2", "B2"]]


def test_picture_uses_caption_text():
    cap = _TextItem(text="Figure 1: schematic", label=_Label("caption"))
    item = _PictureItem(label=_Label("picture"), captions=[cap])

    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].type == "figure"
    assert blocks[0].text == "Figure 1: schematic"


def test_formula_becomes_formula_block():
    item = _TextItem(text="E = mc^2", label=_Label("formula"))
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].type == "formula"
    assert blocks[0].text == "E = mc^2"


def test_unknown_label_with_text_becomes_other():
    item = _TextItem(text="weird content", label=_Label("page_marginalia"))
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].type == "other"
    assert blocks[0].text == "weird content"


def test_bbox_and_page_propagate_from_prov():
    item = _TextItem(
        text="positioned",
        label=_Label("paragraph"),
        prov=[_Prov(page_no=7, bbox=_BBox(l=10.0, t=20.0, r=110.0, b=40.0))],
    )
    blocks = _docling_doc_to_blocks(_FakeDoc([(item, 0)]))

    assert blocks[0].page == 7
    assert blocks[0].bbox == (10.0, 20.0, 110.0, 40.0)


def test_blocks_assigned_sequential_tmp_ids():
    items = [_TextItem(text=f"para {i}", label=_Label("paragraph")) for i in range(3)]
    blocks = _docling_doc_to_blocks(_FakeDoc([(it, 0) for it in items]))
    assert [b.block_id for b in blocks] == ["tmp:0000", "tmp:0001", "tmp:0002"]


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_docling_stats_count_pages_and_block_types():
    items = [
        _TextItem(text="t1", label=_Label("section_header"), level=1),
        _TextItem(text="t2", label=_Label("paragraph")),
        _TextItem(text="t3", label=_Label("paragraph")),
    ]
    doc = _FakeDoc([(it, 0) for it in items], pages=4)
    blocks = _docling_doc_to_blocks(doc)
    stats = _docling_stats(doc, blocks)

    assert stats["pages"] == 4
    assert stats["blocks_by_type"]["heading"] == 1
    assert stats["blocks_by_type"]["paragraph"] == 2


# --------------------------------------------------------------------------
# Public entry point: parse_pdf_docling()
# --------------------------------------------------------------------------


def test_parse_pdf_docling_uses_monkeypatched_converter(monkeypatch):
    """Smoke-test the full conversion path with no real Docling involved."""

    items = [
        _TextItem(text="Intro", label=_Label("section_header"), level=1),
        _TextItem(text="body.", label=_Label("paragraph")),
    ]
    fake_doc = _FakeDoc([(it, 0) for it in items], pages=2)

    class _FakeResult:
        document = fake_doc

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    monkeypatch.setattr(db, "_CONVERTER", _FakeConverter())

    text, blocks, stats = parse_pdf_docling("/whatever.pdf")

    assert "Intro" in text
    assert len(blocks) == 2
    assert blocks[0].type == "heading"
    assert stats["pages"] == 2
    # Cleanup so other tests don't see the planted converter.
    reset_converter_cache()


def test_missing_docling_raises_actionable_error(monkeypatch):
    """When the [docling] extra is not installed, the user must see a
    concrete `pip install` hint."""

    reset_converter_cache()

    # Force the lazy import to fail.
    def _raise_import_error(name, *args, **kwargs):
        if "docling" in name:
            raise ImportError(f"No module named {name!r}")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_import_error)

    with pytest.raises(RuntimeError) as excinfo:
        db._get_converter()
    assert "pip install -e .[docling]" in str(excinfo.value)


def test_converter_cached_across_calls(monkeypatch):
    """The lazy initialisation should only run once per process."""

    reset_converter_cache()

    call_count = {"n": 0}

    class _FakeConverter:
        def convert(self, path):
            return None

    class _FakeModule:
        class DocumentConverter:
            def __new__(cls):
                call_count["n"] += 1
                return _FakeConverter()

    monkeypatch.setitem(
        __import__("sys").modules,
        "docling",
        _FakeModule,  # type: ignore[arg-type]
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "docling.document_converter",
        _FakeModule,  # type: ignore[arg-type]
    )

    db._get_converter()
    db._get_converter()
    db._get_converter()

    assert call_count["n"] == 1
    reset_converter_cache()
