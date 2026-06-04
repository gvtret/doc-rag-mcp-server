"""Tests for the manifest schema_version guard and the migrate CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from doc_rag.raglib.pipeline import (
    MANIFEST_SCHEMA_VERSION,
    ManifestSchemaTooNew,
    _check_manifest_schema,
)


def test_check_schema_accepts_current_version():
    _check_manifest_schema({"schema_version": MANIFEST_SCHEMA_VERSION, "documents": []})


def test_check_schema_accepts_missing_version_as_legacy():
    """Missing schema_version is treated as 0 — older files stay readable."""
    _check_manifest_schema({"documents": []})


def test_check_schema_rejects_future_version():
    with pytest.raises(ManifestSchemaTooNew) as excinfo:
        _check_manifest_schema({"schema_version": MANIFEST_SCHEMA_VERSION + 1})
    msg = str(excinfo.value)
    assert "doc-rag migrate" in msg


def test_check_schema_ignores_non_integer_version():
    """Defensive: garbled value must not crash, must not silently 'reject'."""
    _check_manifest_schema({"schema_version": "not-a-number"})


def test_delete_documents_refuses_future_schema(built_corpus):
    from doc_rag.raglib.pipeline import delete_documents

    root, doc_ids = built_corpus
    manifest_path = root / "build" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION + 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestSchemaTooNew):
        delete_documents(str(root / "config" / "config.yaml"), [doc_ids[0]])


def _run_cli(root: Path, args: list[str]) -> subprocess.CompletedProcess:
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


def test_migrate_cli_reports_current_schema(built_corpus):
    root, _ = built_corpus
    res = _run_cli(root, ["migrate"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["supported_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["found_schema_version"] in (
        MANIFEST_SCHEMA_VERSION,
        0,
    )  # the test fixture writes no schema_version key, so 0 is also acceptable
    assert payload["migrations_applied"] == []


def test_migrate_cli_handles_missing_manifest(tmp_corpus_root):
    res = _run_cli(tmp_corpus_root, ["migrate"])

    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["supported_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["found_schema_version"] is None
