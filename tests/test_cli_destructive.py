"""Tests for the destructive pipeline commands.

These exercise `delete_documents`, `wipe_index`, `clean_orphans`, and
`clear_incoming` directly via their Python entry points. The CLI
wrappers (`doc_rag.cli`) are also exercised through subprocess in a
single end-to-end test per command, just enough to be confident the
argparse glue is wired.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from doc_rag.raglib.pipeline import (
    clean_orphans,
    clear_incoming,
    delete_documents,
    wipe_index,
)

# --------------------------------------------------------------------------
# delete_documents
# --------------------------------------------------------------------------


def test_delete_documents_removes_manifest_and_prunes_index(built_corpus):
    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    # Sanity: corpus is non-empty.
    manifest = json.loads((root / "build" / "manifest.json").read_text())
    assert len(manifest["documents"]) == 2

    result = delete_documents(config_path, [doc_ids[0]])

    assert result["requested"] == 1
    assert result["deleted"] == 1
    assert result["missing"] == []
    # Manifest now has one document, not two.
    manifest = json.loads((root / "build" / "manifest.json").read_text())
    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["doc_id"] == doc_ids[1]
    # Md file of the deleted doc is gone, the other is kept.
    assert not (root / "build" / "docs_md" / f"{doc_ids[0]}.md").exists()
    assert (root / "build" / "docs_md" / f"{doc_ids[1]}.md").exists()


def test_delete_documents_reports_missing_ids(built_corpus):
    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    result = delete_documents(config_path, ["doc-aaa", "doc-not-there"])

    assert result["requested"] == 2
    assert result["deleted"] == 1
    assert "doc-not-there" in result["missing"]


def test_delete_documents_empty_list_is_noop(tmp_corpus_root):
    config_path = str(tmp_corpus_root / "config" / "config.yaml")
    # Manifest doesn't even need to exist for this fast path.
    result = delete_documents(config_path, [])
    assert result == {"requested": 0, "deleted": 0, "missing": 0}


# --------------------------------------------------------------------------
# wipe_index
# --------------------------------------------------------------------------


def test_wipe_index_clears_build_and_archived_but_not_incoming(built_corpus):
    root, _ = built_corpus
    config_path = str(root / "config" / "config.yaml")

    # Drop a file into sources/incoming to prove wipe doesn't touch it.
    incoming_marker = root / "sources" / "incoming" / "marker.txt"
    incoming_marker.write_text("untouched", encoding="utf-8")

    result = wipe_index(config_path)

    assert result["removed_entries"] >= 1
    # build/index, build/docs_md, build/chunks_jsonl are empty (dir kept).
    assert list((root / "build" / "index").iterdir()) == []
    assert list((root / "build" / "docs_md").iterdir()) == []
    assert list((root / "build" / "chunks_jsonl").iterdir()) == []
    # sources/archived is empty.
    assert list((root / "sources" / "archived").iterdir()) == []
    # Manifest is gone.
    assert not (root / "build" / "manifest.json").exists()
    # sources/incoming is intentionally left alone.
    assert incoming_marker.exists()


# --------------------------------------------------------------------------
# clean_orphans
# --------------------------------------------------------------------------


def test_clean_orphans_removes_unreferenced_md_and_chunks(built_corpus):
    root, doc_ids = built_corpus
    config_path = str(root / "config" / "config.yaml")

    # Plant an orphan md and an orphan chunk line.
    orphan_md = root / "build" / "docs_md" / "orphan-zzz.md"
    orphan_md.write_text("# orphan\n", encoding="utf-8")

    chunks_path = root / "build" / "chunks_jsonl" / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "chunk_id": "orphan-zzz:0000",
                    "doc_id": "orphan-zzz",
                    "text": "stray text",
                    "source_file": "orphan.pdf",
                }
            )
            + "\n"
        )

    result = clean_orphans(config_path)

    assert result["orphan_md_removed"] == 1
    assert result["orphan_chunks_removed"] == 1
    assert "orphan-zzz" in result["orphan_doc_ids"]
    # The orphan md file is gone; the known docs survive.
    assert not orphan_md.exists()
    for did in doc_ids:
        assert (root / "build" / "docs_md" / f"{did}.md").exists()


def test_clean_orphans_no_op_when_nothing_to_clean(built_corpus):
    root, _ = built_corpus
    config_path = str(root / "config" / "config.yaml")

    result = clean_orphans(config_path)

    assert result["orphan_md_removed"] == 0
    assert result["orphan_chunks_removed"] == 0
    assert result["orphan_doc_ids"] == []


# --------------------------------------------------------------------------
# clear_incoming
# --------------------------------------------------------------------------


def test_clear_incoming_removes_files_in_incoming(tmp_corpus_root):
    config_path = str(tmp_corpus_root / "config" / "config.yaml")
    incoming = tmp_corpus_root / "sources" / "incoming"

    (incoming / "a.pdf").write_text("a", encoding="utf-8")
    (incoming / "b.pdf").write_text("b", encoding="utf-8")
    (incoming / "subdir").mkdir()
    (incoming / "subdir" / "c.pdf").write_text("c", encoding="utf-8")

    result = clear_incoming(config_path)

    assert result["removed"] >= 2  # a.pdf, b.pdf, subdir (counted as one rmtree call)
    assert list(incoming.iterdir()) == []


def test_clear_incoming_empty_is_noop(tmp_corpus_root):
    config_path = str(tmp_corpus_root / "config" / "config.yaml")
    result = clear_incoming(config_path)
    assert result["removed"] == 0


# --------------------------------------------------------------------------
# CLI wrapper smoke tests
# --------------------------------------------------------------------------


def _run_cli(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Invoke doc-rag CLI as a subprocess in the test root."""
    env = os.environ.copy()
    env["DOC_RAG_ROOT"] = str(root)
    cmd = [
        sys.executable,
        "-m",
        "doc_rag.cli",
        "--config",
        str(root / "config" / "config.yaml"),
    ] + args
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, check=False)


def test_cli_wipe_refuses_without_confirm(built_corpus):
    root, _ = built_corpus
    res = _run_cli(root, ["wipe"])
    assert res.returncode != 0
    assert "DELETE" in res.stderr
    # Manifest is still there.
    assert (root / "build" / "manifest.json").exists()


def test_cli_wipe_with_confirm_removes_everything(built_corpus):
    root, _ = built_corpus
    res = _run_cli(root, ["wipe", "--confirm", "DELETE"])
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert not (root / "build" / "manifest.json").exists()


def test_cli_delete_smoke(built_corpus):
    root, doc_ids = built_corpus
    res = _run_cli(root, ["delete", doc_ids[0]])
    assert res.returncode == 0, f"stderr: {res.stderr}"
    out = json.loads(res.stdout)
    assert out["deleted"] == 1
    assert out["requested"] == 1


def test_cli_clean_orphans_smoke(built_corpus):
    root, _ = built_corpus
    # Plant an orphan.
    (root / "build" / "docs_md" / "stray.md").write_text("# stray\n", encoding="utf-8")
    res = _run_cli(root, ["clean-orphans"])
    assert res.returncode == 0, f"stderr: {res.stderr}"
    out = json.loads(res.stdout)
    assert out["orphan_md_removed"] >= 1


def test_cli_clear_incoming_smoke(tmp_corpus_root):
    (tmp_corpus_root / "sources" / "incoming" / "junk.pdf").write_text("x", encoding="utf-8")
    res = _run_cli(tmp_corpus_root, ["clear-incoming"])
    assert res.returncode == 0, f"stderr: {res.stderr}"
    out = json.loads(res.stdout)
    assert out["removed"] >= 1
