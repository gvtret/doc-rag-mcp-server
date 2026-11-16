"""Shared helpers for UI stats endpoints (/ui/doc-stats, /ui/documents).

Extracted so both endpoints report sizes through the SAME resolver --
before this module the Documents page kept the old root-only lookup and
showed a dash for ~60 documents whose real size Stats rendered fine
(review finding I4). Also provides a single-pass chunk counter: the
previous per-document full-file scan of chunks.jsonl cost ~12 s for the
whole corpus (dominant cost of /ui/doc-stats).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# basename -> path preference on collision: archived first (final home
# after ingest), then incoming, then any other sources/ subdirectory.
_BN_RANK = {"archived": 0, "incoming": 1}


def build_sources_index(root: Path) -> dict[str, Path]:
    """One-time index of every file under <root>/sources/ by basename.

    Deterministic on collisions: archived > incoming > other subdirs,
    instead of whichever path rglob happened to reach first.
    """
    index: dict[str, Path] = {}
    sources_root = root / "sources"
    if not sources_root.is_dir():
        return index
    for p in sources_root.rglob("*"):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue  # vanished mid-walk (ingest move), skip
        name = p.name
        prev = index.get(name)
        if prev is None:
            index[name] = p
            continue
        rank_new = _BN_RANK.get(p.parent.name, 2)
        rank_old = _BN_RANK.get(prev.parent.name, 2)
        if rank_new < rank_old:
            index[name] = p
    return index


def stat_size(path: Path) -> int | None:
    """st_size with TOCTOU/OSError safety: ingest moves files mid-request."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def make_size_resolver(
    root: Path, paths: dict[str, Any], sources_index: dict[str, Path]
):
    """Return a resolver: manifest source_file -> byte size or None.

    Ladder: (1) exact manifest path, (2) basename in the roots of
    archived/incoming (legacy layout), (3) basename anywhere under
    sources/ (ingest moved it into a subdirectory). Missing file -> None,
    never 0 -- the UI renders a dash for None.
    """

    def _resolve_size(source_file: str) -> int | None:
        cand = root / source_file
        if cand.is_file():
            size = stat_size(cand)
            if size is not None:
                return size
        for dir_key in ("sources_archived", "sources_incoming"):
            d_rel = paths.get(dir_key)
            if d_rel:
                cand = root / d_rel / os.path.basename(source_file)
                if cand.is_file():
                    size = stat_size(cand)
                    if size is not None:
                        return size
        hit = sources_index.get(os.path.basename(source_file))
        if hit is not None:
            size = stat_size(hit)
            if size is not None:
                return size
        return None

    return _resolve_size


def count_chunks_per_doc(chunks_path: Path) -> dict[str, int]:
    """One pass over chunks.jsonl -> {doc_id: chunk_count}.

    The previous code re-opened and re-scanned the whole file for every
    document (substring match per line), which made /ui/doc-stats pay a
    full-file scan N times (~12 s at 199 docs); one pass is O(file).
    """
    counts: dict[str, int] = {}
    if not chunks_path.is_file():
        return counts
    try:
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                # Chunk records start with {"doc_id": "...": parse just
                # that key cheaply instead of json.loads on every line.
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                did = obj.get("doc_id")
                if isinstance(did, str):
                    counts[did] = counts.get(did, 0) + 1
    except OSError as e:
        logger.warning("chunks.jsonl unreadable (%s): %s", chunks_path, e)
    return counts
