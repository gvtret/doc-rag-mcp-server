from __future__ import annotations

import os
from typing import Any

from doc_rag.raglib.blocks import Block, blocks_to_markdown
from doc_rag.raglib.filetype_detect import detect_supported_extension

_BACKEND_BY_EXT: dict[str, str] = {
    ".pdf": "docling",
    ".docx": "python-docx",
    ".doc": "antiword",
    ".md": "direct",
    ".txt": "direct",
}

_VALID_PDF_BACKENDS: frozenset[str] = frozenset({"docling", "auto"})
_VALID_DOCX_BACKENDS: frozenset[str] = frozenset({"python-docx", "docling"})


def _baseline_blocks(text: str, source_backend: str) -> list[Block]:
    """One paragraph block carrying the full document text.

    Used by backends without structural awareness (python-docx, antiword,
    direct read for .md/.txt). Block IDs are sequence-numbered without
    the doc_id prefix (`tmp:0000`); the pipeline rewrites them at save
    time."""
    if not text.strip():
        return []
    return [
        Block(
            block_id="tmp:0000",
            doc_id="tmp",
            type="paragraph",
            text=text,
            source_backend=source_backend,
        )
    ]


def _max_int(cfg: dict[str, Any], key: str, default: int) -> int:
    try:
        v = int((cfg.get("parsing", {}) or {}).get(key, default))
    except Exception:
        v = default
    return max(1, v)


def _max_file_bytes(cfg: dict[str, Any]) -> int:
    parsing = cfg.get("parsing", {}) or {}
    mb = parsing.get("max_file_mb", 50)
    try:
        mb_i = int(mb)
    except Exception:
        mb_i = 50
    mb_i = max(1, mb_i)
    return mb_i * 1024 * 1024


def _enforce_size_limit(cfg: dict[str, Any], path: str) -> None:
    try:
        size = os.path.getsize(path)
    except Exception:
        return
    limit = _max_file_bytes(cfg)
    if size > limit:
        raise RuntimeError(f"File too large ({size} bytes > {limit} bytes): {path}")


def _min_chars_per_page(cfg: dict[str, Any]) -> int:
    try:
        v = int((cfg.get("parsing", {}) or {}).get("min_chars_per_page", 20))
    except Exception:
        v = 20
    return max(0, v)


def _validate_pdf_backend(cfg: dict[str, Any]) -> None:
    backend = (cfg.get("parsing", {}) or {}).get("pdf_backend", "docling")
    if backend not in _VALID_PDF_BACKENDS:
        raise RuntimeError(
            f"parsing.pdf_backend={backend!r} is no longer supported. "
            f"Since v2.0 the only PDF backend is Docling. "
            f"Set parsing.pdf_backend to 'docling' (or remove the key)."
        )


def _validate_docx_backend(cfg: dict[str, Any]) -> None:
    backend = (cfg.get("parsing", {}) or {}).get("docx_backend", "python-docx")
    if backend not in _VALID_DOCX_BACKENDS:
        raise RuntimeError(
            f"parsing.docx_backend={backend!r} is unknown. "
            f"Valid choices: 'python-docx' (default, fast) or 'docling' (structure-aware)."
        )


def _empty_ocr_summary() -> dict[str, Any]:
    return {
        "applied": False,
        "pages_recognized": 0,
        "native_chars_total": 0,
        "after_merge_chars_total": 0,
        "before_ocr_chars": None,
        "after_ocr_chars": None,
        "routing": None,
        "detected_scan": None,
        "pages_with_embedded_images": None,
    }


def _parse_docx(path: str, *, max_paragraphs: int) -> tuple[str, dict[str, Any]]:
    from docx import Document
    from docx.oxml.ns import qn  # type: ignore

    d = Document(path)
    parts: list[str] = []
    used = 0
    para_count = 0
    tables_count = 0

    body = d.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            if para_count >= max_paragraphs:
                continue
            para_count += 1
            text = "".join(r.text for r in child.iter(qn("w:t")) if r.text)
            if text.strip():
                parts.append(text)
                used += 1
        elif tag == "tbl":
            tables_count += 1
            rows_out: list[str] = []
            for tr in child.iter(qn("w:tr")):
                cells: list[str] = []
                prev_cell_text = None
                for tc in tr.iter(qn("w:tc")):
                    cell_text = "".join(r.text for r in tc.iter(qn("w:t")) if r.text).strip()
                    if cell_text != prev_cell_text:
                        cells.append(cell_text)
                    prev_cell_text = cell_text
                if any(cells):
                    rows_out.append(" | ".join(cells))
            if rows_out:
                parts.append("\n".join(rows_out))

    text = "\n\n".join(parts)
    stats: dict[str, Any] = {
        "format": "docx",
        "paragraphs_extracted": used,
        "tables_extracted": tables_count,
        "text_chars_extracted": len(text),
    }
    return text, stats


def _parse_md(path: str) -> tuple[str, dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text, {"format": "md", "text_chars_extracted": len(text)}


def _parse_txt(path: str) -> tuple[str, dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text, {"format": "txt", "text_chars_extracted": len(text)}


def _parse_doc(path: str) -> tuple[str, dict[str, Any]]:
    """Parse legacy binary .doc via antiword/catdoc subprocess."""
    import shutil
    import subprocess

    tool = None
    for candidate in ("antiword", "catdoc"):
        if shutil.which(candidate):
            tool = candidate
            break
    if tool is None:
        raise RuntimeError(
            ".doc парсинг требует antiword или catdoc. Установите: sudo apt install antiword"
        )

    try:
        result = subprocess.run(
            [tool, path],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{tool} timed out on {path}") from None

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{tool} failed (exit {result.returncode}): {stderr or path}")

    text = result.stdout.decode("utf-8", errors="replace")
    return text, {"format": "doc", "tool": tool, "text_chars_extracted": len(text)}


def _finalize_pdf_stats(raw: dict[str, Any], *, min_chars: int) -> dict[str, Any]:
    chars_per = raw.get("chars_per_page")
    if not isinstance(chars_per, list):
        chars_per = []
    ints = [int(x) for x in chars_per if isinstance(x, (int, float))]
    below = sum(1 for c in ints if c < min_chars)
    min_c = min(ints) if ints else 0
    max_c = max(ints) if ints else 0
    out = dict(raw)
    # `chars_per_page` was a PyMuPDF/PyPDF2-specific raw field. Docling
    # does not populate it; `.pop(..., None)` keeps this defensive.
    out.pop("chars_per_page", None)
    out["min_chars_per_page_threshold"] = min_chars
    out["pages_below_min_chars"] = below
    out["min_chars_on_extracted_page"] = min_c
    out["max_chars_on_extracted_page"] = max_c
    return out


def _build_ocr_stats_block(native: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stats shape kept for downstream consumers.

    Docling handles OCR internally via RapidOCR. We surface the
    real signal from Docling's per-page confidence report (non-NaN
    `ocr_score` = OCR fired on that page) through the same shape the
    manifest schema v1 reader expects:

      - `applied: True` if at least one page was OCR-derived;
      - `pages_recognized` is the count of those pages;
      - `confidence` is the mean OCR quality score across them.

    `before_ocr` / `after_ocr` are kept as null fields — they were
    Tesseract-routing artefacts and no longer carry a meaningful
    value, but their presence keeps older readers happy."""
    block: dict[str, Any] = {
        "applied": False,
        "before_ocr": {"chars": None},
        "after_ocr": {"chars": None},
        "pages_recognized": 0,
    }
    if not isinstance(native, dict):
        return block
    ocr_pages_count = native.get("ocr_pages_count")
    if isinstance(ocr_pages_count, int) and ocr_pages_count > 0:
        block["applied"] = True
        block["pages_recognized"] = ocr_pages_count
        mean_score = native.get("ocr_mean_score")
        if isinstance(mean_score, (int, float)):
            block["confidence"] = float(mean_score)
        ocr_pages = native.get("ocr_pages")
        if isinstance(ocr_pages, list):
            block["ocr_pages"] = ocr_pages
    return block


def parse_document(cfg: dict[str, Any], path: str) -> dict[str, Any]:
    _enforce_size_limit(cfg, path)
    _validate_pdf_backend(cfg)
    _validate_docx_backend(cfg)

    text = ""
    tables: list[Any] = []
    min_thr = _min_chars_per_page(cfg)
    extract_stats: dict[str, Any] = {}

    # Route by magic bytes, falling back to filename extension only for
    # plain-text formats (`.md`, `.txt`) that have no magic bytes.
    effective_ext = detect_supported_extension(path)
    if effective_ext is None:
        raise RuntimeError(f"Unsupported file type: {path}")

    source_backend = ""
    # Backends that emit typed blocks (Docling) supply them directly;
    # baseline backends go through `_baseline_blocks` (one paragraph).
    backend_blocks: list[Block] | None = None

    if effective_ext == ".pdf":
        from doc_rag.raglib.docling_backend import parse_pdf_docling

        text, backend_blocks, raw_stats = parse_pdf_docling(path)
        source_backend = "docling"
        extract_stats = _finalize_pdf_stats(raw_stats, min_chars=min_thr)
    elif effective_ext == ".docx":
        docx_backend = (cfg.get("parsing", {}) or {}).get("docx_backend", "python-docx")
        if docx_backend == "docling":
            from doc_rag.raglib.docling_backend import parse_pdf_docling

            text, backend_blocks, extract_stats = parse_pdf_docling(path)
            source_backend = "docling"
        else:
            max_paragraphs = _max_int(cfg, "max_docx_paragraphs", 20000)
            text, extract_stats = _parse_docx(path, max_paragraphs=max_paragraphs)
            source_backend = "python-docx"
    elif effective_ext == ".doc":
        text, extract_stats = _parse_doc(path)
        source_backend = "antiword"
    elif effective_ext == ".md":
        text, extract_stats = _parse_md(path)
        source_backend = "direct"
    elif effective_ext == ".txt":
        text, extract_stats = _parse_txt(path)
        source_backend = "direct"
    else:
        # Defensive: detect_supported_extension only returns one of the
        # five we already handle above.
        raise RuntimeError(f"Unsupported file type: {path}")

    text_before_norm = text
    # Minimal normalization. Skip blank-line removal for .md/.txt where
    # blank lines are semantic. Use the content-resolved extension, not
    # the filename, so a misnamed `.txt` that is really a PDF still gets
    # the blank-line stripping that PDF text extraction needs.
    if cfg.get("parsing", {}).get("normalize_whitespace", True):
        text = "\n".join([line.rstrip() for line in text.splitlines()])
        if effective_ext not in (".md", ".txt"):
            text = "\n".join([line for line in text.splitlines() if line.strip() != ""])

    if backend_blocks is not None:
        blocks = backend_blocks
    else:
        blocks = _baseline_blocks(text.strip(), source_backend)
    md = blocks_to_markdown(blocks, os.path.basename(path))
    text_chars_after_norm = len(text.strip())
    before_norm_chars = len(text_before_norm.strip()) if text_before_norm else 0

    native: dict[str, Any] = dict(extract_stats)
    native["before_normalize"] = {"chars": before_norm_chars}
    native["after_normalize"] = {"chars": text_chars_after_norm}
    native["markdown"] = {"chars": len(md)}

    is_pdf = effective_ext == ".pdf"
    ocr_block = (
        _build_ocr_stats_block(extract_stats)
        if is_pdf
        else {
            "applied": False,
            "before_ocr": {"chars": None},
            "after_ocr": {"chars": None},
            "pages_recognized": 0,
        }
    )

    stats: dict[str, Any] = {
        "ocr": ocr_block,
        "native_text_extraction": native,
    }

    return {"markdown": md, "tables": tables, "stats": stats, "blocks": blocks}
