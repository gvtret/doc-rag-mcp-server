"""Document quality checks — mandatory pre-indexing QA on typed blocks.

Checks each document's blocks for common extraction problems and emits
a per-document JSON report under `build/quality/<doc_id>.json` plus a
roll-up summary at `build/quality/summary.json`.

Severity levels: info, warn, error. A composite score (0.0–1.0) is
derived from the weighted penalty of warnings and errors.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any

from doc_rag.raglib.blocks import Block

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class Warning:
    severity: str  # "info" | "warn" | "error"
    code: str
    message: str
    page: int | None = None
    block_id: str | None = None


@dataclass
class QualityReport:
    doc_id: str
    pages: int
    blocks: dict[str, int]
    warnings: list[Warning]
    score: float
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "pages": self.pages,
            "blocks": self.blocks,
            "warnings": [
                {
                    "severity": w.severity,
                    "code": w.code,
                    "message": w.message,
                    **({"page": w.page} if w.page is not None else {}),
                    **({"block_id": w.block_id} if w.block_id is not None else {}),
                }
                for w in self.warnings
            ],
            "score": round(self.score, 4),
            "schema_version": self.schema_version,
        }


def _page_set(blocks: list[Block]) -> set[int]:
    pages = set()
    for b in blocks:
        if b.page is not None and b.page > 0:
            pages.add(b.page)
    return pages


def _blocks_by_page(blocks: list[Block]) -> dict[int, list[Block]]:
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        p = b.page if b.page is not None else 0
        by_page.setdefault(p, []).append(b)
    return by_page


def _text_density(block: Block) -> float:
    if not block.bbox:
        return 0.0
    x0, y0, x1, y1 = block.bbox
    area = max((x1 - x0) * (y1 - y0), 1.0)
    return len(block.text) / area


def _is_formula_garbage(text: str) -> bool:
    if len(text) < 3:
        return False
    unknown = sum(
        1
        for ch in text
        if unicodedata.category(ch) in ("Cn", "Co", "Cf", "So")
        and ch not in "\u00a0\u200b\u200c\u200d\ufeff"
    )
    return unknown / len(text) > 0.15


def _has_duplicate_headers(blocks: list[Block], threshold: int = 3) -> list[str]:
    """Detect header/footer pollution: same text repeated on many pages."""
    header_counts: dict[str, set[int]] = {}
    for b in blocks:
        if b.type in ("heading", "paragraph") and b.page is not None:
            key = b.text.strip().lower()
            if 3 <= len(key) <= 100:
                header_counts.setdefault(key, set()).add(b.page)

    dupes: list[str] = []
    for text, pages in header_counts.items():
        if len(pages) >= threshold:
            dupes.append(text)
    return dupes


def _low_density_pages(by_page: dict[int, list[Block]], threshold: int = 5) -> list[int]:
    """Pages with suspiciously few total characters."""
    low: list[int] = []
    for page, blocks in sorted(by_page.items()):
        if page == 0:
            continue
        total_chars = sum(len(b.text) for b in blocks)
        if total_chars < threshold:
            low.append(page)
    return low


def _check_tables(blocks: list[Block]) -> list[Warning]:
    """Check for broken tables: empty cells, mismatched column counts."""
    warnings: list[Warning] = []
    for b in blocks:
        if b.type != "table":
            continue
        lines = [line for line in b.text.split("\n") if line.strip()]
        if not lines:
            warnings.append(
                Warning(
                    severity="warn",
                    code="empty_table",
                    message=f"table {b.block_id}: empty table",
                    block_id=b.block_id,
                    page=b.page,
                )
            )
            continue
        col_counts = [len(line.split("|")) for line in lines if "|" in line]
        if not col_counts:
            continue
        most_common = max(set(col_counts), key=col_counts.count)
        empty_cells = sum(line.count("| |") + line.count("|  |") for line in lines)
        total_cells = sum(col_counts)
        if total_cells > 0 and empty_cells / total_cells > 0.4:
            warnings.append(
                Warning(
                    severity="warn",
                    code="broken_table",
                    message=f"table {b.block_id}: empty cells ratio > 40%",
                    block_id=b.block_id,
                    page=b.page,
                )
            )
        inconsistent = [c for c in col_counts if c != most_common]
        if inconsistent and len(inconsistent) > len(col_counts) * 0.3:
            warnings.append(
                Warning(
                    severity="warn",
                    code="mismatched_columns",
                    message=f"table {b.block_id}: mismatched column counts",
                    block_id=b.block_id,
                    page=b.page,
                )
            )
    return warnings


def _check_empty_pages(blocks: list[Block], total_pages: int) -> list[Warning]:
    warnings: list[Warning] = []
    if total_pages <= 1:
        return warnings
    page_texts: dict[int, int] = {}
    for b in blocks:
        if b.page and b.page > 0:
            page_texts[b.page] = page_texts.get(b.page, 0) + len(b.text)
    for p in range(1, total_pages + 1):
        if p not in page_texts or page_texts[p] == 0:
            warnings.append(
                Warning(
                    severity="info",
                    code="empty_page",
                    message=f"page {p}: no content extracted",
                    page=p,
                )
            )
    return warnings


def _check_formulas(blocks: list[Block]) -> list[Warning]:
    warnings: list[Warning] = []
    for b in blocks:
        if b.type != "formula":
            continue
        if _is_formula_garbage(b.text):
            warnings.append(
                Warning(
                    severity="warn",
                    code="formula_garbage",
                    message=f"formula {b.block_id}: high ratio of unknown Unicode characters",
                    block_id=b.block_id,
                    page=b.page,
                )
            )
    return warnings


def _check_unreadable_chars(blocks: list[Block], threshold: float = 0.10) -> list[Warning]:
    warnings: list[Warning] = []
    total_chars = sum(len(b.text) for b in blocks)
    if total_chars == 0:
        return warnings
    unreadable = 0
    for b in blocks:
        for ch in b.text:
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in ("\n", "\t", "\r"):
                unreadable += 1
    ratio = unreadable / total_chars
    if ratio > threshold:
        warnings.append(
            Warning(
                severity="warn",
                code="high_unreadable_ratio",
                message=f"unreadable character ratio: {ratio:.1%} (threshold: {threshold:.0%})",
            )
        )
    return warnings


def _compute_score(warnings: list[Warning], total_blocks: int) -> float:
    if total_blocks == 0:
        return 1.0
    penalty = 0.0
    for w in warnings:
        if w.severity == "error":
            penalty += 0.15
        elif w.severity == "warn":
            penalty += 0.05
        elif w.severity == "info":
            penalty += 0.01
    raw = 1.0 - penalty
    return max(0.0, min(1.0, raw))


def run_quality_checks(doc_id: str, blocks: list[Block]) -> QualityReport:
    """Run all quality checks on a document's blocks and return a report."""
    pages = _page_set(blocks)
    total_pages = max(pages) if pages else 0
    by_page = _blocks_by_page(blocks)

    block_counts: dict[str, int] = {}
    for b in blocks:
        block_counts[b.type] = block_counts.get(b.type, 0) + 1

    warnings: list[Warning] = []
    warnings.extend(_check_empty_pages(blocks, total_pages))
    warnings.extend(_check_tables(blocks))
    warnings.extend(_check_formulas(blocks))
    warnings.extend(_check_unreadable_chars(blocks))

    low = _low_density_pages(by_page)
    for p in low:
        warnings.append(
            Warning(
                severity="info",
                code="low_text_density",
                message=f"page {p}: very low text density (<5 chars)",
                page=p,
            )
        )

    dupes = _duplicate_header_footer(blocks)
    for text in dupes:
        warnings.append(
            Warning(
                severity="info",
                code="duplicate_header_footer",
                message=f"repeated header/footer across pages: {text!r}",
            )
        )

    score = _compute_score(warnings, len(blocks))

    report = QualityReport(
        doc_id=doc_id,
        pages=total_pages,
        blocks=block_counts,
        warnings=warnings,
        score=score,
    )
    logger.info("quality: %s score=%.2f warnings=%d", doc_id, score, len(warnings))
    return report


def _duplicate_header_footer(blocks: list[Block]) -> list[str]:
    return _has_duplicate_headers(blocks)


def persist_quality_report(quality_dir: str, report: QualityReport) -> str:
    """Write per-doc quality JSON and update summary. Returns the doc report path."""
    os.makedirs(quality_dir, exist_ok=True)
    doc_path = os.path.join(quality_dir, f"{report.doc_id}.json")
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(quality_dir, "summary.json")
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "documents": []}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {"schema_version": SCHEMA_VERSION, "documents": []}

    docs = summary.get("documents", [])
    if not isinstance(docs, list):
        docs = []
    docs = [d for d in docs if isinstance(d, dict) and d.get("doc_id") != report.doc_id]
    docs.append(
        {
            "doc_id": report.doc_id,
            "pages": report.pages,
            "score": round(report.score, 4),
            "warning_count": len(report.warnings),
        }
    )
    summary["documents"] = docs
    summary["schema_version"] = SCHEMA_VERSION

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return doc_path
