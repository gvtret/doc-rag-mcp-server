from __future__ import annotations
import os
from typing import Any, Dict, List

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


def _parse_pdf_pypdf2(path: str, *, max_pages: int) -> str:
    from PyPDF2 import PdfReader
    r = PdfReader(path)
    parts: List[str] = []
    for page in r.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)

def _parse_pdf_pymupdf(path: str, *, max_pages: int) -> str:
    import fitz  # pymupdf
    doc = fitz.open(path)
    parts: List[str] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        parts.append(page.get_text("text") or "")
    return "\n\n".join(parts)

def _parse_docx(path: str, *, max_paragraphs: int) -> str:
    from docx import Document
    d = Document(path)
    parts: List[str] = []
    for i, p in enumerate(d.paragraphs):
        if i >= max_paragraphs:
            break
        if p.text:
            parts.append(p.text)
    return "\n\n".join(parts)

def parse_document(cfg: Dict[str, Any], path: str) -> Dict[str, Any]:
    _enforce_size_limit(cfg, path)
    text = ""
    tables: List[Any] = []
    if path.lower().endswith(".pdf"):
        backend = cfg.get("parsing", {}).get("pdf_backend", "auto")
        max_pages = _max_int(cfg, "max_pdf_pages", 2000)
        if backend in ("pymupdf", "auto"):
            try:
                text = _parse_pdf_pymupdf(path, max_pages=max_pages)
            except Exception:
                text = _parse_pdf_pypdf2(path, max_pages=max_pages)
        else:
            text = _parse_pdf_pypdf2(path, max_pages=max_pages)
    elif path.lower().endswith(".docx"):
        max_paragraphs = _max_int(cfg, "max_docx_paragraphs", 20000)
        text = _parse_docx(path, max_paragraphs=max_paragraphs)
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

    # Minimal normalization
    if cfg.get("parsing", {}).get("normalize_whitespace", True):
        text = "\n".join([line.rstrip() for line in text.splitlines()])
        text = "\n".join([line for line in text.splitlines() if line.strip() != ""])

    md = f"# {os.path.basename(path)}\n\n" + (text.strip() + "\n")
    return {"markdown": md, "tables": tables}
