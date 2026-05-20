from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

def _max_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        v = int((cfg.get("parsing", {}) or {}).get(key, default))
    except Exception:
        v = default
    return max(1, v)


def _max_file_bytes(cfg: Dict[str, Any]) -> int:
    parsing = cfg.get("parsing", {}) or {}
    mb = parsing.get("max_file_mb", 50)
    try:
        mb_i = int(mb)
    except Exception:
        mb_i = 50
    mb_i = max(1, mb_i)
    return mb_i * 1024 * 1024


def _enforce_size_limit(cfg: Dict[str, Any], path: str) -> None:
    try:
        size = os.path.getsize(path)
    except Exception:
        return
    limit = _max_file_bytes(cfg)
    if size > limit:
        raise RuntimeError(f"File too large ({size} bytes > {limit} bytes): {path}")


def _min_chars_per_page(cfg: Dict[str, Any]) -> int:
    try:
        v = int((cfg.get("parsing", {}) or {}).get("min_chars_per_page", 20))
    except Exception:
        v = 20
    return max(0, v)


def _ocr_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    o = (cfg.get("parsing", {}) or {}).get("ocr")
    return o if isinstance(o, dict) else {}


def _ocr_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_ocr_config(cfg).get("enabled"))


def _ocr_runtime_imports() -> Tuple[Any, Any]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "parsing.ocr.enabled requires pytesseract and Pillow. "
            "Install: pip install 'doc-rag[ocr]' or pytesseract Pillow."
        ) from e
    return pytesseract, Image


def _ocr_page_text(page: Any, fitz_mod: Any, pytesseract: Any, Image: Any, ocr_cfg: Dict[str, Any]) -> str:
    scale = float(ocr_cfg.get("render_scale", 2.0))
    scale = max(0.5, min(4.0, scale))
    mat = fitz_mod.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    lang = str(ocr_cfg.get("tesseract_lang", "rus+eng+equ"))
    cmd = ocr_cfg.get("tesseract_cmd")
    if isinstance(cmd, str) and cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = cmd.strip()
    cfg_extra = ocr_cfg.get("tesseract_config")
    extra = cfg_extra.strip() if isinstance(cfg_extra, str) else ""
    def _run(lang_arg: str) -> str:
        if extra:
            return str(pytesseract.image_to_string(img, lang=lang_arg, config=extra) or "")
        return str(pytesseract.image_to_string(img, lang=lang_arg) or "")

    try:
        return _run(lang)
    except Exception as e:
        err = str(e).lower()
        if "equ" in lang.lower() and "+equ" in lang.lower() and (
            "traineddata" in err or "could not load" in err or "failed loading" in err
        ):
            # Часто на минимальных образах нет tesseract-ocr-equ — откатываемся на rus+eng.
            fallback = "+".join(p for p in lang.split("+") if p.strip().lower() != "equ")
            if fallback and fallback != lang:
                try:
                    return _run(fallback)
                except Exception:
                    pass
        if "tesseract" in err or "not installed" in err:
            raise RuntimeError(
                "Tesseract OCR engine or language data missing. "
                "Install: apt install tesseract-ocr-rus tesseract-ocr-eng tesseract-ocr-equ "
                "(or adjust parsing.ocr.tesseract_lang) and set parsing.ocr.tesseract_cmd if needed."
            ) from e
        raise


def _empty_ocr_summary() -> Dict[str, Any]:
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


def _page_has_embedded_images(page: Any) -> bool:
    """Страница с растровыми вложениями (не вектор-only)."""
    try:
        ims = page.get_images(full=True)
        return len(ims) > 0
    except Exception:
        try:
            return len(page.get_images()) > 0
        except Exception:
            return False


def _extract_page_text_structured(page: Any, fitz_mod: Any) -> str:
    """Extract page text, replacing table regions with pipe-separated row text.

    Uses page.find_tables() (PyMuPDF >= 1.23) when tables are present so that
    14-column tables like Приложение И produce lines like:
        1 | 3 | 1.0.12.7.0.255 | Напряжение U | - | G | G | - |  |  |  |  |  | Да
    instead of one value per line, preserving row context for RAG chunking.
    Falls back to get_text("text") if find_tables is unavailable or finds nothing.
    """
    try:
        finder = page.find_tables()
        tables = finder.tables if finder and hasattr(finder, "tables") else []
    except Exception:
        return page.get_text("text") or ""

    if not tables:
        return page.get_text("text") or ""

    # Build (Rect, formatted_text) for each detected table
    table_entries: List[Tuple[Any, str]] = []
    for tab in tables:
        try:
            trect = fitz_mod.Rect(tab.bbox)
        except Exception:
            continue
        rows_out: List[str] = []
        try:
            for row in (tab.extract() or []):
                if not row:
                    continue
                cells = [(c or "").strip() if c is not None else "" for c in row]
                if any(cells):
                    rows_out.append(" | ".join(cells))
        except Exception:
            pass
        if rows_out:
            table_entries.append((trect, "\n".join(rows_out)))

    if not table_entries:
        return page.get_text("text") or ""

    # Walk text blocks top-to-bottom; replace blocks that fall inside a table rect
    # with the formatted table text (emitted once per table).
    try:
        blocks = page.get_text("blocks", sort=True)
    except Exception:
        return page.get_text("text") or ""

    output: List[str] = []
    emitted: set = set()

    for b in blocks:
        if len(b) < 5:
            continue
        bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
        if len(b) >= 7 and b[6] != 0:  # skip image blocks
            continue
        block_rect = fitz_mod.Rect(bx0, by0, bx1, by1)

        matched = None
        for i, (trect, _) in enumerate(table_entries):
            inter = trect & block_rect
            if inter.is_empty:
                continue
            block_area = block_rect.get_area()
            if block_area > 0 and inter.get_area() / block_area > 0.4:
                matched = i
                break

        if matched is not None:
            if matched not in emitted:
                emitted.add(matched)
                output.append(table_entries[matched][1])
        else:
            text = (btext or "").strip()
            if text:
                output.append(text)

    # Emit any tables not reached via blocks (safety net)
    for i, (_, ttext) in enumerate(table_entries):
        if i not in emitted:
            output.append(ttext)

    return "\n\n".join(output)


def _detect_pdf_is_scan(
    parts: List[str], *, threshold: int, n_pages: int, ocr_cfg: Dict[str, Any]
) -> bool:
    """Эвристика «весь документ — скан»: почти все страницы без нативного текста."""
    if n_pages <= 0:
        return False
    sparse = sum(1 for p in parts if len((p or "").strip()) < threshold)
    frac = sparse / float(n_pages)
    try:
        need_frac = float(ocr_cfg.get("scan_sparse_page_fraction", 0.72))
    except Exception:
        need_frac = 0.72
    need_frac = min(1.0, max(0.0, need_frac))

    try:
        max_total = int(ocr_cfg.get("scan_max_total_native_chars", 0))
    except Exception:
        max_total = 0
    total_native = sum(len((p or "").strip()) for p in parts)

    if max_total > 0 and total_native <= max_total:
        return True
    return frac >= need_frac


def _pdf_fitz_extract_with_ocr(
    cfg: Dict[str, Any], path: str, *, max_pages: int
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """PyMuPDF native text + optional Tesseract per page. Returns text, raw_stats, ocr_summary."""
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise RuntimeError(
            "PDF processing requires pymupdf when using the PyMuPDF code path. Install: pip install pymupdf"
        ) from e

    ocr_cfg = _ocr_config(cfg)
    ocr_on = bool(ocr_cfg.get("enabled"))
    pytesseract = None
    Image = None
    if ocr_on:
        pytesseract, Image = _ocr_runtime_imports()

    doc = fitz.open(path)
    try:
        total_pages = len(doc)
        n_read = min(total_pages, max_pages)
        parts: List[str] = []
        for i in range(n_read):
            parts.append(_extract_page_text_structured(doc[i], fitz) or "")

        ocr_summary: Dict[str, Any] = {
            "applied": False,
            "pages_recognized": 0,
            "native_chars_total": sum(len((p or "").strip()) for p in parts),
            "after_merge_chars_total": 0,
            "before_ocr_chars": None,
            "after_ocr_chars": None,
        }

        if ocr_on and pytesseract is not None and Image is not None:
            threshold = int(ocr_cfg.get("page_native_chars_threshold", 30))
            threshold = max(0, threshold)
            force_manual = bool(ocr_cfg.get("force_all_pages", False))

            page_has_img = [_page_has_embedded_images(doc[i]) for i in range(n_read)]
            is_scan = _detect_pdf_is_scan(parts, threshold=threshold, n_pages=n_read, ocr_cfg=ocr_cfg)

            if force_manual:
                routing = "force_all_pages"
            elif is_scan:
                routing = "scan_all_pages"
            else:
                routing = "embedded_images_only"

            before_chars = sum(len((p or "").strip()) for p in parts)
            pages_hit = 0
            for i in range(n_read):
                if force_manual:
                    need = True
                elif is_scan:
                    need = True
                else:
                    need = page_has_img[i]

                if not need:
                    continue
                try:
                    ocr_txt = _ocr_page_text(doc[i], fitz, pytesseract, Image, ocr_cfg)
                except RuntimeError:
                    raise
                except Exception:
                    continue
                if (ocr_txt or "").strip():
                    parts[i] = ocr_txt
                    pages_hit += 1
            after_chars = sum(len((p or "").strip()) for p in parts)
            ocr_summary["applied"] = pages_hit > 0
            ocr_summary["pages_recognized"] = pages_hit
            ocr_summary["native_chars_total"] = before_chars
            ocr_summary["after_merge_chars_total"] = after_chars
            ocr_summary["before_ocr_chars"] = before_chars
            ocr_summary["after_ocr_chars"] = after_chars
            ocr_summary["routing"] = routing
            ocr_summary["detected_scan"] = bool(is_scan)
            ocr_summary["pages_with_embedded_images"] = int(sum(page_has_img))

        text = "\n\n".join(parts)
        chars_per = [len((p or "").strip()) for p in parts]
        raw_stats: Dict[str, Any] = {
            "format": "pdf",
            "pdf_backend": "pymupdf+ocr" if ocr_summary.get("applied") else "pymupdf",
            "source_page_count": total_pages,
            "pages_extracted": n_read,
            "chars_per_page": chars_per,
            "text_chars_extracted": len(text),
        }
        return text, raw_stats, ocr_summary
    finally:
        doc.close()


def _parse_pdf_pypdf2(path: str, *, max_pages: int) -> Tuple[str, Dict[str, Any]]:
    from PyPDF2 import PdfReader
    r = PdfReader(path)
    total_pages = len(r.pages)
    n_read = min(total_pages, max_pages)
    parts: List[str] = []
    for page in r.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n\n".join(parts)
    chars_per = [len((p or "").strip()) for p in parts]
    stats: Dict[str, Any] = {
        "format": "pdf",
        "pdf_backend": "pypdf2",
        "source_page_count": total_pages,
        "pages_extracted": n_read,
        "chars_per_page": chars_per,
        "text_chars_extracted": len(text),
    }
    return text, stats


def _parse_docx(path: str, *, max_paragraphs: int) -> Tuple[str, Dict[str, Any]]:
    from docx import Document
    from docx.oxml.ns import qn  # type: ignore

    d = Document(path)
    parts: List[str] = []
    used = 0
    para_count = 0
    tables_count = 0

    # Iterate body in document order so tables appear at their correct position
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
            rows_out: List[str] = []
            for tr in child.iter(qn("w:tr")):
                cells: List[str] = []
                prev_cell_text = None
                for tc in tr.iter(qn("w:tc")):
                    cell_text = "".join(
                        r.text for r in tc.iter(qn("w:t")) if r.text
                    ).strip()
                    # Skip merged-cell duplicates (python-docx repeats merged cells)
                    if cell_text != prev_cell_text:
                        cells.append(cell_text)
                    prev_cell_text = cell_text
                if any(cells):
                    rows_out.append(" | ".join(cells))
            if rows_out:
                parts.append("\n".join(rows_out))

    text = "\n\n".join(parts)
    stats: Dict[str, Any] = {
        "format": "docx",
        "paragraphs_extracted": used,
        "tables_extracted": tables_count,
        "text_chars_extracted": len(text),
    }
    return text, stats


def _finalize_pdf_stats(raw: Dict[str, Any], *, min_chars: int) -> Dict[str, Any]:
    chars_per = raw.get("chars_per_page")
    if not isinstance(chars_per, list):
        chars_per = []
    ints = [int(x) for x in chars_per if isinstance(x, (int, float))]
    below = sum(1 for c in ints if c < min_chars)
    min_c = min(ints) if ints else 0
    max_c = max(ints) if ints else 0
    out = dict(raw)
    del out["chars_per_page"]
    out["min_chars_per_page_threshold"] = min_chars
    out["pages_below_min_chars"] = below
    out["min_chars_on_extracted_page"] = min_c
    out["max_chars_on_extracted_page"] = max_c
    return out


def _build_ocr_stats_block(cfg: Dict[str, Any], ocr_summary: Dict[str, Any]) -> Dict[str, Any]:
    oc = _ocr_config(cfg)
    lang = oc.get("tesseract_lang")
    block: Dict[str, Any] = {
        "applied": bool(ocr_summary.get("applied")),
        "before_ocr": {"chars": ocr_summary.get("before_ocr_chars")},
        "after_ocr": {"chars": ocr_summary.get("after_ocr_chars")},
        "pages_recognized": int(ocr_summary.get("pages_recognized") or 0),
    }
    if _ocr_enabled(cfg):
        block["tesseract_lang"] = str(lang) if lang is not None else "rus+eng+equ"
    for key in ("routing", "detected_scan", "pages_with_embedded_images"):
        v = ocr_summary.get(key)
        if v is not None:
            block[key] = v
    return block


def parse_document(cfg: Dict[str, Any], path: str) -> Dict[str, Any]:
    _enforce_size_limit(cfg, path)
    text = ""
    tables: List[Any] = []
    min_thr = _min_chars_per_page(cfg)
    extract_stats: Dict[str, Any] = {}
    ocr_summary: Dict[str, Any] = _empty_ocr_summary()

    if path.lower().endswith(".pdf"):
        backend = cfg.get("parsing", {}).get("pdf_backend", "auto")
        max_pages = _max_int(cfg, "max_pdf_pages", 2000)
        if _ocr_enabled(cfg) and backend == "pypdf2":
            raise RuntimeError(
                "parsing.ocr.enabled requires pdf_backend 'auto' or 'pymupdf' (PyMuPDF), not 'pypdf2'."
            )

        raw_stats: Dict[str, Any] = {}
        if backend == "pypdf2":
            text, raw_stats = _parse_pdf_pypdf2(path, max_pages=max_pages)
            ocr_summary = _empty_ocr_summary()
        elif backend in ("pymupdf", "auto"):
            try:
                import fitz  # noqa: F401
            except ImportError:
                if backend == "pymupdf":
                    raise RuntimeError(
                        "pdf_backend is 'pymupdf' but pymupdf is not installed. pip install pymupdf"
                    ) from None
                if _ocr_enabled(cfg):
                    raise RuntimeError(
                        "parsing.ocr.enabled requires pymupdf. Install: pip install pymupdf"
                    ) from None
                text, raw_stats = _parse_pdf_pypdf2(path, max_pages=max_pages)
                ocr_summary = _empty_ocr_summary()
            else:
                text, raw_stats, ocr_summary = _pdf_fitz_extract_with_ocr(cfg, path, max_pages=max_pages)
        else:
            raise RuntimeError(f"Unknown pdf_backend: {backend!r}")
        extract_stats = _finalize_pdf_stats(raw_stats, min_chars=min_thr)
    elif path.lower().endswith(".docx"):
        max_paragraphs = _max_int(cfg, "max_docx_paragraphs", 20000)
        text, extract_stats = _parse_docx(path, max_paragraphs=max_paragraphs)
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

    text_before_norm = text
    # Minimal normalization
    if cfg.get("parsing", {}).get("normalize_whitespace", True):
        text = "\n".join([line.rstrip() for line in text.splitlines()])
        text = "\n".join([line for line in text.splitlines() if line.strip() != ""])

    md = f"# {os.path.basename(path)}\n\n" + (text.strip() + "\n")
    text_chars_after_norm = len(text.strip())
    before_norm_chars = len(text_before_norm.strip()) if text_before_norm else 0

    native: Dict[str, Any] = dict(extract_stats)
    native["before_normalize"] = {"chars": before_norm_chars}
    native["after_normalize"] = {"chars": text_chars_after_norm}
    native["markdown"] = {"chars": len(md)}

    is_pdf = path.lower().endswith(".pdf")
    ocr_block: Dict[str, Any]
    if is_pdf:
        ocr_block = _build_ocr_stats_block(cfg, ocr_summary)
    else:
        ocr_block = {
            "applied": False,
            "before_ocr": {"chars": None},
            "after_ocr": {"chars": None},
            "pages_recognized": 0,
        }

    stats: Dict[str, Any] = {
        "ocr": ocr_block,
        "native_text_extraction": native,
    }
    return {"markdown": md, "tables": tables, "stats": stats}
