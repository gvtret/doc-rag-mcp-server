"""Tests for _archive_or_dedup_sources function."""

from __future__ import annotations

import hashlib
import os
import textwrap
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_archive_moves_incoming_to_archived(tmp_path: Path) -> None:
    """File in incoming should be moved to archived when no conflict."""
    from doc_rag.raglib.pipeline import _archive_or_dedup_sources

    incoming = tmp_path / "sources" / "incoming"
    archived = tmp_path / "sources" / "archived"
    incoming.mkdir(parents=True)
    archived.mkdir(parents=True)

    (incoming / "test.txt").write_text("hello world")

    cfg = {
        "paths": {
            "sources_incoming": "sources/incoming",
            "sources_archived": "sources/archived",
        }
    }

    _archive_or_dedup_sources(cfg, str(tmp_path), [str(incoming / "test.txt")])

    assert not (incoming / "test.txt").exists()
    assert (archived / "test.txt").exists()
    assert (archived / "test.txt").read_text() == "hello world"


def test_archive_dedup_same_content(tmp_path: Path) -> None:
    """Same file in incoming and archived with same hash: delete incoming."""
    from doc_rag.raglib.pipeline import _archive_or_dedup_sources

    incoming = tmp_path / "sources" / "incoming"
    archived = tmp_path / "sources" / "archived"
    incoming.mkdir(parents=True)
    archived.mkdir(parents=True)

    (incoming / "dup.txt").write_text("same content")
    (archived / "dup.txt").write_text("same content")

    cfg = {
        "paths": {
            "sources_incoming": "sources/incoming",
            "sources_archived": "sources/archived",
        }
    }

    _archive_or_dedup_sources(cfg, str(tmp_path), [str(incoming / "dup.txt")])

    assert not (incoming / "dup.txt").exists()
    assert (archived / "dup.txt").exists()


def test_archive_conflict_different_hash(tmp_path: Path) -> None:
    """Same filename but different content: append suffix to avoid overwrite."""
    from doc_rag.raglib.pipeline import _archive_or_dedup_sources

    incoming = tmp_path / "sources" / "incoming"
    archived = tmp_path / "sources" / "archived"
    incoming.mkdir(parents=True)
    archived.mkdir(parents=True)

    (incoming / "conflict.txt").write_text("new version")
    (archived / "conflict.txt").write_text("old version")

    cfg = {
        "paths": {
            "sources_incoming": "sources/incoming",
            "sources_archived": "sources/archived",
        }
    }

    _archive_or_dedup_sources(cfg, str(tmp_path), [str(incoming / "conflict.txt")])

    # Original should still exist
    assert (archived / "conflict.txt").read_text() == "old version"
    # New version should have suffix
    suffix_files = list(archived.glob("conflict*.txt"))
    assert len(suffix_files) == 2


def test_archive_removes_empty_incoming_dirs(tmp_path: Path) -> None:
    """Empty directories in incoming should be cleaned up."""
    from doc_rag.raglib.pipeline import _archive_or_dedup_sources

    incoming = tmp_path / "sources" / "incoming"
    archived = tmp_path / "sources" / "archived"
    sub = incoming / "subdir"
    sub.mkdir(parents=True)
    archived.mkdir(parents=True)

    (incoming / "test.txt").write_text("content")

    cfg = {
        "paths": {
            "sources_incoming": "sources/incoming",
            "sources_archived": "sources/archived",
        }
    }

    _archive_or_dedup_sources(cfg, str(tmp_path), [str(incoming / "test.txt")])

    assert not sub.exists()


def test_archive_skips_when_no_archived_config(tmp_path: Path) -> None:
    """When sources_archived is not set, archiving should be skipped."""
    from doc_rag.raglib.pipeline import _archive_or_dedup_sources

    incoming = tmp_path / "sources" / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "test.txt").write_text("content")

    cfg = {"paths": {"sources_incoming": "sources/incoming"}}

    _archive_or_dedup_sources(cfg, str(tmp_path), [str(incoming / "test.txt")])

    assert (incoming / "test.txt").exists()
