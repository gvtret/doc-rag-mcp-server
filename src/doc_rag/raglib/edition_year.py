"""Определение года редакции документа для manifest (ручные переопределения + эвристики)."""

from __future__ import annotations

import os
import re
from typing import Any


def _coerce_year(v: Any) -> int | None:
    if v is None:
        return None
    try:
        y = int(v)
    except (TypeError, ValueError):
        return None
    if 1000 <= y <= 9999:
        return y
    return None


def _norm_rel(rel: str) -> str:
    return rel.replace("\\", "/")


def _year_from_pdf_date_string(s: Any) -> int | None:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if t.startswith("D:") and len(t) >= 6:
        try:
            return int(t[2:6])
        except ValueError:
            pass
    m = re.search(r"(19|20)\d{2}", t)
    if m:
        try:
            y = int(m.group(0))
            if 1000 <= y <= 9999:
                return y
        except ValueError:
            pass
    return None


def _year_from_pypdf2_metadata(path: str) -> int | None:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return None
    try:
        r = PdfReader(path)
        meta = getattr(r, "metadata", None)
        if meta is None:
            return None
        for key in ("/ModDate", "/CreationDate"):
            raw = None
            if hasattr(meta, "get"):
                raw = meta.get(key)
            if raw is None and hasattr(meta, key):
                raw = getattr(meta, key, None)
            y = _year_from_pdf_date_string(str(raw) if raw is not None else None)
            if y is not None:
                return y
    except Exception:
        return None
    return None


def _year_from_pymupdf_metadata(path: str) -> int | None:
    try:
        import fitz  # pymupdf
    except Exception:
        return None
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        doc.close()
        for k in ("modDate", "creationDate", "date"):
            y = _year_from_pdf_date_string(meta.get(k))
            if y is not None:
                return y
    except Exception:
        return None
    return None


def _year_from_pdf_file(path: str) -> int | None:
    y = _year_from_pymupdf_metadata(path)
    if y is not None:
        return y
    return _year_from_pypdf2_metadata(path)


def _year_from_filename(basename: str, pattern: str | None) -> int | None:
    if not pattern or not isinstance(pattern, str) or not pattern.strip():
        return None
    try:
        m = re.search(pattern, basename)
    except re.error:
        return None
    if not m:
        return None
    if "year" in m.groupdict():
        return _coerce_year(m.group("year"))
    if m.lastindex and m.lastindex >= 1:
        return _coerce_year(m.group(1))
    return None


def resolve_edition_year(
    cfg: dict[str, Any],
    *,
    abs_path: str,
    rel_path: str,
    sha256_hex: str,
) -> int | None:
    """Приоритет: by_basename → by_source_rel_path → by_sha256 → PDF metadata → filename_regex."""
    ey = (cfg.get("parsing", {}) or {}).get("edition_year")
    if not isinstance(ey, dict):
        ey = {}

    base = os.path.basename(abs_path)
    rel_n = _norm_rel(rel_path)

    by_bn = ey.get("by_basename") or {}
    if isinstance(by_bn, dict) and base in by_bn:
        y = _coerce_year(by_bn.get(base))
        if y is not None:
            return y

    by_rel = ey.get("by_source_rel_path") or {}
    if isinstance(by_rel, dict):
        for key in (rel_n, rel_path):
            if key in by_rel:
                y = _coerce_year(by_rel.get(key))
                if y is not None:
                    return y

    hx = (sha256_hex or "").strip().lower()
    by_h = ey.get("by_sha256") or {}
    if isinstance(by_h, dict) and hx:
        for k, v in by_h.items():
            if isinstance(k, str) and k.strip().lower() == hx:
                y = _coerce_year(v)
                if y is not None:
                    return y

    use_meta = ey.get("from_pdf_metadata")
    if use_meta is None:
        use_meta = True
    if use_meta and abs_path.lower().endswith(".pdf"):
        y = _year_from_pdf_file(abs_path)
        if y is not None:
            return y

    pat = ey.get("filename_regex")
    if isinstance(pat, str) and pat.strip():
        y = _year_from_filename(base, pat.strip())
        if y is not None:
            return y

    return None
