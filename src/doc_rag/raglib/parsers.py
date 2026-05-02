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

def _parse_pdf_pymupdf(path: str, *, max_pages: int) -> Tuple[str, Dict[str, Any]]:
    import fitz  # pymupdf
    doc = fitz.open(path)
    total_pages = len(doc)
    n_read = min(total_pages, max_pages)
    parts: List[str] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        parts.append(page.get_text("text") or "")
    text = "\n\n".join(parts)
    chars_per = [len((p or "").strip()) for p in parts]
    stats: Dict[str, Any] = {
        "format": "pdf",
        "pdf_backend": "pymupdf",
        "source_page_count": total_pages,
        "pages_extracted": n_read,
        "chars_per_page": chars_per,
        "text_chars_extracted": len(text),
    }
    doc.close()
    return text, stats

def _parse_docx(path: str, *, max_paragraphs: int) -> Tuple[str, Dict[str, Any]]:
    from docx import Document
    d = Document(path)
    parts: List[str] = []
    used = 0
    for i, p in enumerate(d.paragraphs):
        if i >= max_paragraphs:
            break
        if p.text:
            parts.append(p.text)
            used += 1
    text = "\n\n".join(parts)
    stats: Dict[str, Any] = {
        "format": "docx",
        "paragraphs_extracted": used,
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


def parse_document(cfg: Dict[str, Any], path: str) -> Dict[str, Any]:
    _enforce_size_limit(cfg, path)
    text = ""
    tables: List[Any] = []
    min_thr = _min_chars_per_page(cfg)
    extract_stats: Dict[str, Any] = {}

    if path.lower().endswith(".pdf"):
        backend = cfg.get("parsing", {}).get("pdf_backend", "auto")
        max_pages = _max_int(cfg, "max_pdf_pages", 2000)
        raw_stats: Dict[str, Any] = {}
        if backend in ("pymupdf", "auto"):
            try:
                text, raw_stats = _parse_pdf_pymupdf(path, max_pages=max_pages)
            except Exception:
                text, raw_stats = _parse_pdf_pypdf2(path, max_pages=max_pages)
        else:
            text, raw_stats = _parse_pdf_pypdf2(path, max_pages=max_pages)
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

    stats: Dict[str, Any] = {
        "ocr": {
            "applied": False,
            "before_ocr": {"chars": None},
            "after_ocr": {"chars": None},
        },
        "native_text_extraction": native,
    }
    return {"markdown": md, "tables": tables, "stats": stats}
