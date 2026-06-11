"""Docling-based parser backend (v1.5, opt-in via `pdf_backend: docling`).

Docling is IBM Research's structure-aware document parser. Compared to
the PyMuPDF default, it brings:

  - TableFormer-based table extraction (preserves cell grid).
  - Formula recognition with LaTeX output.
  - DocLayout-YOLO for multi-column and complex layouts.

In exchange it costs disk space (~600 MB of dependencies, plus ~300 MB
of ML models downloaded on first use) and CPU time. The backend is
behind the `[docling]` optional extra and lazily imported, so the
default install footprint is unchanged when this backend isn't used.

The public surface is one function — `parse_pdf_docling(path)` — that
returns `(text, blocks, stats)` in the same shape the other parsers in
this package produce, so `parse_document` can dispatch on the
`pdf_backend` config without further special-casing.

Models are loaded on first call by `_get_converter()`; subsequent calls
in the same process reuse the cached converter. The cache is reset to
None on `ImportError` so a deployment that lacks Docling fails fast
with an actionable message instead of importing-on-every-call.
"""

from __future__ import annotations

import math
from typing import Any

from doc_rag.raglib.blocks import Block

#: Cached `docling.document_converter.DocumentConverter` instance, lazily
#: initialised on first `parse_pdf_docling()` call. Initialisation is
#: expensive (model load) and the converter is reusable across documents.
_CONVERTER: Any = None


def _get_converter() -> Any:
    """Return a cached DocumentConverter, initialising it on first use.

    Lazy import is deliberate: the `[docling]` extra is optional, and
    we want servers that do not use this backend to start without
    Docling on the import path.
    """
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Docling is not installed. Install the optional extra: pip install -e .[docling]"
        ) from e
    _CONVERTER = DocumentConverter()
    return _CONVERTER


def reset_converter_cache() -> None:
    """Drop the cached converter — primarily a hook for tests."""
    global _CONVERTER
    _CONVERTER = None


def _iter_items(doc: Any) -> list[tuple[Any, int]]:
    """Walk a DoclingDocument and return its items in reading order.

    Docling exposes the traversal as `doc.iterate_items()` which yields
    `(item, level)` tuples, where `level` is the hierarchical depth in
    the document tree. We materialise the iterator into a list so the
    caller can index into it.
    """
    iterate = getattr(doc, "iterate_items", None)
    if iterate is None:
        return []
    out: list[tuple[Any, int]] = []
    for entry in iterate():
        if isinstance(entry, tuple) and len(entry) == 2:
            out.append((entry[0], int(entry[1])))
        else:
            # Some Docling versions yield only the item; treat hierarchy
            # level as 0 in that case.
            out.append((entry, 0))
    return out


def _label_of(item: Any) -> str:
    """Extract the Docling label of a node as a lowercase string.

    `DocItemLabel` is an enum in Docling; `.value` gives the canonical
    short string ("section_header", "paragraph", "table", …). When the
    attribute is missing or the enum hasn't been imported, we fall back
    to the type name in lowercase, then to the empty string.
    """
    label = getattr(item, "label", None)
    if label is None:
        return type(item).__name__.lower()
    return str(getattr(label, "value", label)).lower()


def _text_of(item: Any) -> str:
    """Best-effort text extraction from any Docling item type."""
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    for attr in ("orig", "content", "value"):
        v = getattr(item, attr, None)
        if isinstance(v, str):
            return v
    return ""


def _caption_of(item: Any) -> str:
    """Best-effort caption extraction for PictureItem / TableItem."""
    get_caption = getattr(item, "caption_text", None)
    if callable(get_caption):
        try:
            captured = get_caption()
            if isinstance(captured, str):
                return captured
        except Exception:
            pass
    caps = getattr(item, "captions", None) or []
    for c in caps:
        t = _text_of(c)
        if t:
            return t
    return ""


def _table_to_text(item: Any) -> tuple[str, list[list[str]] | None]:
    """Render a TableItem as plain text and (when possible) a cell grid.

    Returns (text, cells). `cells` is a list-of-rows of strings when
    Docling exposes a structured grid; otherwise `None`.
    """
    cells_grid: list[list[str]] | None = None
    data = getattr(item, "data", None)
    grid = getattr(data, "grid_cells", None) if data is not None else None
    if grid is None and data is not None:
        # Some Docling versions name this `.table_cells`.
        grid = getattr(data, "table_cells", None)
    if grid is not None:
        rows: dict[int, dict[int, str]] = {}
        for cell in grid:
            row = int(getattr(cell, "start_row_offset_idx", 0) or 0)
            col = int(getattr(cell, "start_col_offset_idx", 0) or 0)
            txt = _text_of(cell)
            rows.setdefault(row, {})[col] = txt
        if rows:
            cells_grid = []
            for r in sorted(rows.keys()):
                row_cells = rows[r]
                if not row_cells:
                    continue
                ncols = max(row_cells.keys()) + 1
                cells_grid.append([row_cells.get(c, "") for c in range(ncols)])

    if cells_grid:
        text = "\n".join(" | ".join(row) for row in cells_grid)
    else:
        text = _text_of(item)
    return text, cells_grid


def _bbox_of(item: Any) -> tuple[float, float, float, float] | None:
    """Extract a (x0, y0, x1, y1) bbox if Docling exposes a prov entry."""
    prov = getattr(item, "prov", None) or []
    for p in prov:
        bbox = getattr(p, "bbox", None)
        if bbox is None:
            continue
        try:
            return (
                float(getattr(bbox, "l", 0.0)),
                float(getattr(bbox, "t", 0.0)),
                float(getattr(bbox, "r", 0.0)),
                float(getattr(bbox, "b", 0.0)),
            )
        except (TypeError, ValueError):
            continue
    return None


def _page_of(item: Any) -> int | None:
    """1-based page number of the item, or None if unknown."""
    prov = getattr(item, "prov", None) or []
    for p in prov:
        page = getattr(p, "page_no", None)
        if page is None:
            page = getattr(p, "page", None)
        if isinstance(page, int):
            return page
    return None


_HEADING_LABELS: frozenset[str] = frozenset({"section_header", "title", "heading"})
_PARAGRAPH_LABELS: frozenset[str] = frozenset(
    {"paragraph", "text", "page_header", "page_footer", "footnote"}
)
_LIST_LABELS: frozenset[str] = frozenset({"list_item", "checkbox_selected", "checkbox_unselected"})
_TABLE_LABELS: frozenset[str] = frozenset({"table", "tableitem"})
_FIGURE_LABELS: frozenset[str] = frozenset({"picture", "pictureitem", "figure"})
_FORMULA_LABELS: frozenset[str] = frozenset({"formula"})
_CODE_LABELS: frozenset[str] = frozenset({"code"})
_CAPTION_LABELS: frozenset[str] = frozenset({"caption"})


def _item_to_block(item: Any, hier_level: int, idx: int) -> Block | None:
    """Translate one Docling item into a Block, or None to skip.

    Block IDs use the placeholder `tmp:NNNN` prefix that the pipeline
    rewrites to `<doc_id>:NNNN` when it persists `build/blocks/<doc_id>.jsonl`.
    """
    block_id = f"tmp:{idx:04d}"
    label = _label_of(item)
    text = _text_of(item)

    if label in _HEADING_LABELS:
        level = int(getattr(item, "level", 0) or 0) or max(1, hier_level)
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="heading",
            text=text,
            source_backend="docling",
            level=level,
            page=_page_of(item),
            bbox=_bbox_of(item),
        )
    if label in _LIST_LABELS:
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="list_item",
            text=text,
            source_backend="docling",
            level=max(0, hier_level - 1),
            page=_page_of(item),
            bbox=_bbox_of(item),
        )
    if label in _TABLE_LABELS:
        table_text, cells = _table_to_text(item)
        metadata: dict[str, Any] = {}
        if cells is not None:
            metadata["cells"] = cells
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="table",
            text=table_text,
            source_backend="docling",
            page=_page_of(item),
            bbox=_bbox_of(item),
            metadata=metadata,
        )
    if label in _FIGURE_LABELS:
        caption = _caption_of(item) or text
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="figure",
            text=caption,
            source_backend="docling",
            page=_page_of(item),
            bbox=_bbox_of(item),
        )
    if label in _FORMULA_LABELS:
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="formula",
            text=text,
            source_backend="docling",
            page=_page_of(item),
            bbox=_bbox_of(item),
        )
    if label in _CODE_LABELS:
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="code",
            text=text,
            source_backend="docling",
            page=_page_of(item),
        )
    if label in _CAPTION_LABELS:
        # Skip stand-alone captions — they are typically attached to a
        # neighbouring figure or table already.
        if not text:
            return None
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="paragraph",
            text=text,
            source_backend="docling",
            page=_page_of(item),
        )
    if label in _PARAGRAPH_LABELS:
        if not text:
            return None
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="paragraph",
            text=text,
            source_backend="docling",
            page=_page_of(item),
            bbox=_bbox_of(item),
        )
    if text:
        return Block(
            block_id=block_id,
            doc_id="tmp",
            type="other",
            text=text,
            source_backend="docling",
            page=_page_of(item),
        )
    return None


def _docling_doc_to_blocks(doc: Any) -> list[Block]:
    """Convert a DoclingDocument into a Block list in reading order."""
    blocks: list[Block] = []
    idx = 0
    for item, level in _iter_items(doc):
        b = _item_to_block(item, hier_level=level, idx=idx)
        if b is not None:
            blocks.append(b)
            idx += 1
    return blocks


def _ocr_signal_from_confidence(confidence: Any) -> dict[str, Any]:
    """Read per-page ocr_score from a Docling ConfidenceReport.

    A non-NaN, positive `ocr_score` on a page is the canonical "OCR ran
    on this page" signal — Docling fills it only when its OCR step
    produced text. Native-text pages get NaN.

    Returns a small dict the caller merges into the stats. Empty when
    Docling did not supply a confidence report (older versions, or a
    backend that bypassed the OCR step entirely).
    """
    if confidence is None:
        return {}
    pages = getattr(confidence, "pages", None)
    if not isinstance(pages, dict):
        return {}
    ocr_pages: list[int] = []
    ocr_scores: list[float] = []
    for page_no, scores in pages.items():
        score = getattr(scores, "ocr_score", None)
        if score is None and isinstance(scores, dict):
            score = scores.get("ocr_score")
        try:
            s = float(score) if score is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if math.isnan(s) or s <= 0.0:
            continue
        try:
            pn = int(page_no)
        except (TypeError, ValueError):
            continue
        ocr_pages.append(pn)
        ocr_scores.append(s)
    if not ocr_pages:
        return {}
    return {
        "ocr_pages_count": len(ocr_pages),
        "ocr_pages": sorted(ocr_pages),
        "ocr_mean_score": round(sum(ocr_scores) / len(ocr_scores), 4),
    }


def _docling_stats(doc: Any, blocks: list[Block], result: Any = None) -> dict[str, Any]:
    """Build a stats dict mirroring the shape PyMuPDF/python-docx return.

    Keeps `parse_document`'s native_text_extraction merging simple.
    """
    page_count = 0
    pages = getattr(doc, "pages", None)
    if pages is not None:
        try:
            page_count = len(pages)
        except TypeError:
            pass

    block_counts: dict[str, int] = {}
    for b in blocks:
        block_counts[b.type] = block_counts.get(b.type, 0) + 1

    stats: dict[str, Any] = {
        "pages": page_count,
        "blocks_by_type": block_counts,
    }
    stats.update(_ocr_signal_from_confidence(getattr(result, "confidence", None)))
    return stats


def parse_pdf_docling(path: str) -> tuple[str, list[Block], dict[str, Any]]:
    """Parse a PDF or DOCX through Docling.

    Returns:
        (text, blocks, stats)

        - `text` is a plain text rendering of the document, suitable for
          the legacy `text`/markdown path. We use Docling's
          `export_to_markdown()` when available and fall back to
          concatenated block text otherwise.
        - `blocks` is the structured `list[Block]` for downstream
          consumers.
        - `stats` mirrors the dict shape that `_finalize_pdf_stats`
          expects (page count + per-type block counts).

    Raises:
        RuntimeError: if the `[docling]` extra is not installed, with an
            actionable installation hint.
    """
    converter = _get_converter()
    result = converter.convert(path)
    doc = getattr(result, "document", result)

    blocks = _docling_doc_to_blocks(doc)

    text: str = ""
    exporter = getattr(doc, "export_to_markdown", None)
    if callable(exporter):
        try:
            exported = exporter()
            if isinstance(exported, str):
                text = exported
        except Exception:
            text = ""
    if not text:
        text = "\n\n".join(b.text for b in blocks if b.text)

    stats = _docling_stats(doc, blocks, result=result)
    return text, blocks, stats
