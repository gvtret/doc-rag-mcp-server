"""Tests for scripts/fetch_model.py (one-time embedding model download).

The scripts/ directory is not a package, so the script is loaded by file
location via importlib; sys.path is never modified here.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_fetch_model():
    spec = importlib.util.spec_from_file_location(
        "fetch_model", ROOT / "scripts" / "fetch_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_snapshot_download(repo_id, local_dir, ignore_patterns=None, **kw):
    p = Path(local_dir)
    p.mkdir(parents=True)
    (p / "config.json").write_text("{}")
    (p / "modules.json").write_text("[]")
    (p / "pytorch_model.bin").write_bytes(b"w")
    return str(p)


def test_download_uses_staging_and_writes_fingerprint(tmp_path, monkeypatch):
    mod = load_fetch_model()
    monkeypatch.setattr(mod, "snapshot_download", _fake_snapshot_download)
    info = mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert (tmp_path / "m" / "WEIGHTS.sha256").is_file()
    assert not (tmp_path / "m.staging").exists()
    assert info["weight_files"] == 1
    assert info["dest"] == str(tmp_path / "m")
    assert len(info["fingerprint"]) == 64


def test_download_replaces_existing_model(tmp_path, monkeypatch):
    mod = load_fetch_model()
    monkeypatch.setattr(mod, "snapshot_download", _fake_snapshot_download)
    dest = tmp_path / "m"
    mod.download_model("BAAI/bge-m3", dest)
    # A second download over the existing model must not fail with ENOTEMPTY.
    info = mod.download_model("BAAI/bge-m3", dest)
    assert (dest / "WEIGHTS.sha256").is_file()
    assert not (tmp_path / "m.old").exists()
    assert not (tmp_path / "m.staging").exists()
    assert info["weight_files"] == 1


def test_download_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    mod = load_fetch_model()

    def broken(repo_id, local_dir, **kw):
        Path(local_dir).mkdir(parents=True)
        return str(local_dir)

    monkeypatch.setattr(mod, "snapshot_download", broken)
    with pytest.raises(RuntimeError):
        mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert not (tmp_path / "m").exists()
    assert not (tmp_path / "m.staging").exists()


def test_ignore_patterns_exclude_onnx_and_imgs(tmp_path, monkeypatch):
    mod = load_fetch_model()
    captured: dict = {}

    def fake(repo_id, local_dir, ignore_patterns=None, **kw):
        captured["ignore_patterns"] = ignore_patterns
        return _fake_snapshot_download(repo_id, local_dir, ignore_patterns, **kw)

    monkeypatch.setattr(mod, "snapshot_download", fake)
    mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert captured["ignore_patterns"] is not None
    assert "onnx/*" in captured["ignore_patterns"]
    assert "imgs/*" in captured["ignore_patterns"]


def test_download_sets_hf_read_timeout(tmp_path, monkeypatch):
    mod = load_fetch_model()
    monkeypatch.setattr(mod, "snapshot_download", _fake_snapshot_download)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60"
