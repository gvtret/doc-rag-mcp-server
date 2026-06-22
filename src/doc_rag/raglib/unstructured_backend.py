"""Unstructured (hi_res) PDF backend — second-tier fallback for Docling.

Handles PDFs that Docling cannot process (broken layout, exotic fonts,
OCR-only scans where Docling's preprocessing fails). Uses the
`unstructured` library with the `hi_res` partitioning strategy.

Follows the same lazy-import pattern as `docling_backend.py`: the
library is imported only when the backend is actually invoked, so
installations that don't need it pay zero cost.
"""

from __future__ import annotations

import logging
from typing import Any

from doc_rag.raglib.blocks import Block

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE_FALLBACK = 5


def _get_partition_function():  # type: ignore[no-untyped-def]
    """Lazy-import Unstructured's partition_pdf and return it."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError(
            "Unstructured is not installed. Install it with: "
            "uv sync --extra unstructured  (or pip install 'unstructured[hi-res]')"
        ) from exc
    return partition_pdf


def _elements_to_blocks(elements: list, doc_id: str) -> tuple[str, list[Block]]:
    """Convert Unstructured elements into (text, blocks).

    Maps Unstructured element types to Block types:
      - Title / SectionHeader → heading
      - NarrativeText / Text → paragraph
      - Table → table
      - ListItem → list_item
      - FigureCaption / Image → figure
      - Formula → formula
      - CodeBlock → code
      - PageBreak / Header / Footer → skipped
    """
    _TYPE_MAP: dict[str, str] = {
        "Title": "heading",
        "SectionHeader": "heading",
        "NarrativeText": "paragraph",
        "Text": "paragraph",
        "UncategorizedText": "paragraph",
        "Table": "table",
        "ListItem": "list_item",
        "BulletedList": "list_item",
        "FigureCaption": "figure",
        "Image": "figure",
        "Figure": "figure",
        "Formula": "formula",
        "CodeBlock": "code",
        "EmailAddress": "paragraph",
        "PageBreak": "_skip",
        "Header": "_skip",
        "Footer": "_skip",
    }

    blocks: list[Block] = []
    parts: list[str] = []
    page_counter = 0

    for i, el in enumerate(elements):
        el_type = type(el).__name__
        text = (el.text or "").strip()
        if not text and el_type != "Table":
            continue

        mapped = _TYPE_MAP.get(el_type, "paragraph")
        if mapped == "_skip":
            if el_type == "PageBreak":
                page_counter += 1
            continue

        page_num = getattr(el.metadata, "page_number", None)
        if page_num is None:
            page_num = page_counter + 1

        bbox = None
        coords = getattr(el.metadata, "coordinates", None)
        if coords and hasattr(coords, "points") and coords.points:
            pts = coords.points
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

        blocks.append(
            Block(
                block_id=f"tmp:{i:04d}",
                doc_id=doc_id,
                type=mapped,  # type: ignore[arg-type]
                text=text,
                source_backend="unstructured",
                page=page_num,
                bbox=bbox,
            )
        )
        parts.append(text)

    full_text = "\n\n".join(parts)
    return full_text, blocks


def parse_pdf_unstructured(path: str) -> tuple[str, list[Block], dict[str, Any]]:
    """Parse a PDF through Unstructured (hi_res strategy).

    Returns:
        (text, blocks, stats)

        Same signature as `parse_pdf_docling` so callers can swap freely.

    Raises:
        RuntimeError: if the `[unstructured]` extra is not installed.
    """
    partition_pdf = _get_partition_function()

    elements = partition_pdf(
        path,
        strategy="hi_res",
        include_page_breaks=True,
    )

    text, blocks = _elements_to_blocks(elements, doc_id="tmp")

    block_counts: dict[str, int] = {}
    for b in blocks:
        block_counts[b.type] = block_counts.get(b.type, 0) + 1

    page_count = 1
    for el in elements:
        pg = getattr(el.metadata, "page_number", None)
        if isinstance(pg, int) and pg > page_count:
            page_count = pg

    stats: dict[str, Any] = {
        "pages": page_count,
        "blocks_by_type": block_counts,
    }
    logger.info(
        "unstructured parsed %s: %d pages, %d blocks",
        path,
        page_count,
        len(blocks),
    )
    return text, blocks, stats
