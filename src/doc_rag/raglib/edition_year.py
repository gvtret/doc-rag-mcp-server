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
    """Приоритет: by_basename → by_source_rel_path → by_sha256 → filename_regex.

    Note: до v2.0 был ещё уровень «авточтение PDF metadata через PyMuPDF/PyPDF2».
    В v2.0 PyMuPDF и PyPDF2 убраны из зависимостей. Если нужен автогод
    для PDF — задайте его явно в `parsing.edition_year.by_basename` /
    `by_sha256` / `by_source_rel_path` или используйте `filename_regex`.
    """
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

    pat = ey.get("filename_regex")
    if isinstance(pat, str) and pat.strip():
        y = _year_from_filename(base, pat.strip())
        if y is not None:
            return y

    return None
