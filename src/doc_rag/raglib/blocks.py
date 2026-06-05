"""Typed `Block` dataclass and JSONL serialization for the v1.5 blocks layer.

Every parser backend (PyMuPDF, python-docx, antiword, Docling, …) emits
a list of `Block`s. `build/blocks/<doc_id>.jsonl` is the canonical
on-disk form. Markdown and chunks downstream consume blocks rather than
re-parsing the source document.

The on-disk schema is documented in `docs/schemas/blocks-v1.md`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: Current schema version for `build/blocks/*.jsonl`. Stored in
#: `manifest.json` under `blocks_schema_version`, not in individual blocks.
BLOCKS_SCHEMA_VERSION = 1

#: Allowed values for `Block.type`. Update `docs/schemas/blocks-v1.md`
#: when this changes.
BlockType = Literal[
    "heading",
    "paragraph",
    "list_item",
    "table",
    "formula",
    "figure",
    "code",
    "quote",
    "other",
]

_VALID_TYPES: frozenset[str] = frozenset(
    {
        "heading",
        "paragraph",
        "list_item",
        "table",
        "formula",
        "figure",
        "code",
        "quote",
        "other",
    }
)

#: Allowed values for `Block.source_backend`. Open enum — backends may add
#: their own labels; the values here are the ones the project itself emits.
_KNOWN_BACKENDS: frozenset[str] = frozenset(
    {
        "pymupdf",
        "pypdf2",
        "python-docx",
        "antiword",
        "catdoc",
        "docling",
        "unstructured",
        "direct",
    }
)


class BlocksSchemaTooNew(RuntimeError):
    """Read attempted on a blocks file written by a newer doc-rag build.

    Raised at the boundary so the caller can decide whether to refuse, to
    fall through to a legacy code path, or to point the user at
    `doc-rag migrate`.
    """

    def __init__(self, found: int, supported: int) -> None:
        super().__init__(
            f"blocks schema_version={found} is newer than this build "
            f"supports (supported: {supported}). Upgrade doc-rag or run "
            "`doc-rag migrate`."
        )
        self.found = found
        self.supported = supported


@dataclass
class Block:
    """A typed block of content extracted from a source document.

    See `docs/schemas/blocks-v1.md` for the on-disk schema. Fields here
    mirror the schema exactly; `to_jsonl_dict()` drops keys whose value
    is `None` so files stay compact and round-trips are stable.
    """

    block_id: str
    doc_id: str
    type: BlockType
    text: str
    source_backend: str
    level: int | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if self.type not in _VALID_TYPES:
            raise ValueError(f"unknown block type {self.type!r}; allowed: {sorted(_VALID_TYPES)}")
        if not self.source_backend:
            raise ValueError("source_backend must be non-empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.level is not None and self.level < 0:
            raise ValueError(f"level must be non-negative, got {self.level}")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"bbox must have 4 floats, got {self.bbox}")
            x0, y0, x1, y1 = self.bbox
            self.bbox = (float(x0), float(y0), float(x1), float(y1))

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a dict suitable for `json.dumps` as a single JSONL line.

        Keys with `None` values are dropped; an empty `metadata` is also
        dropped. This keeps the on-disk file compact and the round-trip
        stable.
        """
        out: dict[str, Any] = {
            "block_id": self.block_id,
            "doc_id": self.doc_id,
            "type": self.type,
            "text": self.text,
            "source_backend": self.source_backend,
        }
        if self.level is not None:
            out["level"] = self.level
        if self.page is not None:
            out["page"] = self.page
        if self.bbox is not None:
            out["bbox"] = list(self.bbox)
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.metadata:
            out["metadata"] = self.metadata
        return out

    @classmethod
    def from_jsonl_dict(cls, data: dict[str, Any]) -> Block:
        """Construct a Block from a parsed JSONL line.

        Tolerant of unknown fields (older builds reading newer files
        ignore extras), but strict about missing required fields.
        """
        try:
            block_id = data["block_id"]
            doc_id = data["doc_id"]
            type_ = data["type"]
            text = data["text"]
            source_backend = data["source_backend"]
        except KeyError as e:
            raise ValueError(f"missing required field: {e.args[0]}") from None

        bbox = data.get("bbox")
        if bbox is not None:
            bbox = tuple(bbox)

        return cls(
            block_id=block_id,
            doc_id=doc_id,
            type=type_,
            text=text,
            source_backend=source_backend,
            level=data.get("level"),
            page=data.get("page"),
            bbox=bbox,  # type: ignore[arg-type]
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata") or {}),
        )


def dump_blocks(path: Path | str, blocks: Iterable[Block]) -> int:
    """Write a sequence of Blocks to a `.jsonl` file. Returns count written.

    The parent directory is created if needed. Existing file is
    overwritten. Each line ends with a single `\\n`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(b.to_jsonl_dict(), ensure_ascii=False))
            fh.write("\n")
            written += 1
    return written


def load_blocks(
    path: Path | str,
    *,
    schema_version: int | None = None,
) -> list[Block]:
    """Read a `.jsonl` file into a list of Blocks.

    Args:
        path: file to read.
        schema_version: if not `None`, raise `BlocksSchemaTooNew` when
            it is higher than `BLOCKS_SCHEMA_VERSION`. Callers that
            already validated the version (e.g. via the manifest) can
            leave this as `None`.

    Returns:
        Materialised list of `Block`s. Use `iter_blocks()` if you want
        streaming.
    """
    if schema_version is not None and schema_version > BLOCKS_SCHEMA_VERSION:
        raise BlocksSchemaTooNew(found=schema_version, supported=BLOCKS_SCHEMA_VERSION)
    return list(iter_blocks(path))


def iter_blocks(path: Path | str) -> Iterator[Block]:
    """Stream a `.jsonl` file as Blocks, one per yielded record."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield Block.from_jsonl_dict(data)
