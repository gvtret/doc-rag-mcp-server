"""Tests for the destructive-operations audit log."""

from __future__ import annotations

import json
from pathlib import Path


def _read_audit(root: Path):
    p = root / "build" / "audit.log"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_delete_appends_audit_record(built_corpus):
    from doc_rag.raglib.pipeline import delete_documents

    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    delete_documents(config_path, [doc_ids[0]])

    records = _read_audit(root)
    assert len(records) == 1
    rec = records[0]
    assert rec["op"] == "delete"
    assert rec["schema_version"] == 1
    assert rec["doc_ids"] == [doc_ids[0]]
    assert rec["counts"]["deleted"] == 1
    assert rec["counts"]["requested"] == 1


def test_wipe_appends_audit_record(built_corpus):
    from doc_rag.raglib.pipeline import wipe_index

    root, _ = built_corpus
    config_path = str(root / "config" / "config.yaml")

    wipe_index(config_path)

    records = _read_audit(root)
    assert any(r["op"] == "wipe" for r in records)
    wipe_rec = [r for r in records if r["op"] == "wipe"][0]
    assert "removed_entries" in wipe_rec["counts"]


def test_clean_orphans_appends_audit_record(built_corpus):
    from doc_rag.raglib.pipeline import clean_orphans

    root, _ = built_corpus
    config_path = str(root / "config" / "config.yaml")

    # Plant an orphan to ensure non-trivial counts.
    (root / "build" / "docs_md" / "orphan.md").write_text("# orphan\n", encoding="utf-8")

    clean_orphans(config_path)

    records = _read_audit(root)
    assert any(r["op"] == "clean_orphans" for r in records)
    rec = [r for r in records if r["op"] == "clean_orphans"][0]
    assert rec["counts"]["orphan_md_removed"] >= 1


def test_clear_incoming_appends_audit_record(tmp_corpus_root):
    from doc_rag.raglib.pipeline import clear_incoming

    config_path = str(tmp_corpus_root / "config" / "config.yaml")
    (tmp_corpus_root / "sources" / "incoming" / "junk.pdf").write_text("x", encoding="utf-8")

    clear_incoming(config_path)

    records = _read_audit(tmp_corpus_root)
    assert any(r["op"] == "clear_incoming" for r in records)
    rec = [r for r in records if r["op"] == "clear_incoming"][0]
    assert rec["counts"]["removed"] >= 1


def test_audit_records_are_jsonl(built_corpus):
    """One JSON object per line, parseable independently."""
    from doc_rag.raglib.pipeline import delete_documents

    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    delete_documents(config_path, [doc_ids[0]])
    delete_documents(config_path, [doc_ids[1]])

    lines = (root / "build" / "audit.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)  # must not raise
        assert "ts" in rec
        assert "op" in rec


def test_read_recent_returns_tail(built_corpus):
    from doc_rag.raglib import audit_log
    from doc_rag.raglib.pipeline import delete_documents

    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    delete_documents(config_path, [doc_ids[0]])

    recent = audit_log.read_recent(str(root), limit=10)
    assert len(recent) == 1
    assert recent[0]["op"] == "delete"
