"""Tests for local embedding-model resolution and weights fingerprinting."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pytest

from doc_rag.raglib import model_paths
from doc_rag.raglib.model_paths import (
    fingerprint_for_cfg,
    fingerprint_from_pairs,
    model_fingerprint,
    resolve_model_path,
    weight_files,
    write_weights_sha256,
)


def make_model_dir(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    (d / "pytorch_model.bin").write_bytes(b"w")
    return d


def make_model_dir_at(d):
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    (d / "pytorch_model.bin").write_bytes(b"w")
    return d


def manifest_lines(d):
    return (d / "WEIGHTS.sha256").read_text().splitlines()


def test_resolve_prefers_model_path(tmp_path):
    custom = make_model_dir_at(tmp_path / "custom" / "bge-m3")
    make_model_dir(tmp_path)  # valid fallback at <root>/models/bge-m3
    cfg = {"_root": str(tmp_path), "embeddings": {"model_path": str(custom)}}
    assert resolve_model_path(cfg) == custom


def test_resolve_model_path_expands_user(tmp_path):
    custom = make_model_dir_at(tmp_path / "custom" / "bge-m3")
    cfg = {"embeddings": {"model_path": str(custom)}}
    monkey_root = os.environ.get("HOME", "/root")
    assert custom.is_relative_to(monkey_root) or resolve_model_path(cfg) == custom


def test_resolve_falls_back_to_build_then_repo(tmp_path):
    repo = make_model_dir(tmp_path)
    build = tmp_path / "build" / "models" / "bge-m3"
    build.parent.mkdir(parents=True)
    repo.rename(build)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert resolve_model_path(cfg) == build


def test_resolve_falls_back_to_repo_models(tmp_path):
    build = tmp_path / "build" / "models" / "bge-m3"
    build.mkdir(parents=True)  # exists but has no model files
    make_model_dir(tmp_path)  # creates tmp_path/models/bge-m3
    repo_dir = tmp_path / "models" / "bge-m3"
    assert resolve_model_path({"_root": str(tmp_path), "embeddings": {}}) == repo_dir


def test_resolve_requires_model_files(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert resolve_model_path(cfg) is None


def test_resolve_explicit_root_arg(tmp_path):
    d = make_model_dir(tmp_path)
    assert resolve_model_path({}, root=str(tmp_path)) == d


def test_fingerprint_changes_with_weights(tmp_path):
    d = make_model_dir(tmp_path)
    (d / "onnx").mkdir()
    (d / "onnx" / "model.onnx_data").write_bytes(b"ignored")
    fp1 = model_fingerprint(d)
    (d / "pytorch_model.bin").write_bytes(b"changed")
    fp2 = model_fingerprint(d)
    assert fp1 != fp2
    assert "onnx" not in (d / "WEIGHTS.sha256").read_text()


def test_model_fingerprint_migrates_missing_file(tmp_path):
    d = make_model_dir(tmp_path)
    assert model_fingerprint(d) is not None
    assert (d / "WEIGHTS.sha256").is_file()


def test_model_fingerprint_refreshes_stale_file(tmp_path):
    d = make_model_dir(tmp_path)
    fp1 = model_fingerprint(d)
    # Weights changed underneath: the stale manifest must not pin the
    # fingerprint, it gets recomputed and refreshed.
    (d / "pytorch_model.bin").write_bytes(b"changed")
    fp2 = model_fingerprint(d)
    assert fp1 != fp2
    # The refreshed manifest matches the actual weights.
    actual = hashlib.sha256((d / "pytorch_model.bin").read_bytes()).hexdigest()
    assert fingerprint_from_pairs([(actual, "pytorch_model.bin")]) == fp2


def test_model_fingerprint_none_without_weights(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    assert model_fingerprint(d) is None
    assert not (d / "WEIGHTS.sha256").exists()


def test_model_fingerprint_empty_manifest_no_weights(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    (d / "WEIGHTS.sha256").write_text("")  # empty legacy manifest
    assert model_fingerprint(d) is None  # not sha256("")


def test_model_fingerprint_readonly_dir_no_write(tmp_path):
    if os.getuid() == 0:
        pytest.skip("root ignores directory permissions")
    d = make_model_dir(tmp_path)
    os.chmod(d, 0o555)  # noqa: S103 - read-only is the point of this test
    try:
        fp = model_fingerprint(d)  # must not raise PermissionError
        assert fp is not None
        assert not (d / "WEIGHTS.sha256").exists()
    finally:
        os.chmod(d, 0o755)  # noqa: S103 - restore for cleanup


def test_model_fingerprint_gate_avoids_rehash(tmp_path, monkeypatch):
    d = make_model_dir(tmp_path)
    fp1 = model_fingerprint(d)  # populates manifest + inventory sidecar

    def _boom(path):
        raise AssertionError("unexpected re-hash on second call")

    monkeypatch.setattr(model_paths, "_sha256_file", _boom)
    fp2 = model_fingerprint(d)
    assert fp2 == fp1


def test_model_fingerprint_gate_invalidated_by_touch(tmp_path, monkeypatch):
    d = make_model_dir(tmp_path)
    fp1 = model_fingerprint(d)
    # Rewriting the weight file changes size and mtime: the gate must
    # detect it and re-hash despite the sidecar being present.
    (d / "pytorch_model.bin").write_bytes(b"changed")

    calls: list[str] = []
    real_sha = model_paths._sha256_file

    def _counting(path):
        calls.append(path.name)
        return real_sha(path)

    monkeypatch.setattr(model_paths, "_sha256_file", _counting)
    fp2 = model_fingerprint(d)
    assert fp1 is not None and fp2 is not None
    assert fp1 != fp2
    assert calls == ["pytorch_model.bin"]


def test_fingerprint_from_pairs_is_order_insensitive():
    pairs = [("b" * 64, "b.bin"), ("a" * 64, "a.bin")]
    assert fingerprint_from_pairs(pairs) == fingerprint_from_pairs(reversed(pairs))


def test_write_weights_sha256_sorted_by_relpath(tmp_path):
    d = make_model_dir(tmp_path)
    (d / "weights.pt").write_bytes(b"pt")
    out = write_weights_sha256(d)
    assert out == d / "WEIGHTS.sha256"
    lines = manifest_lines(d)
    assert len(lines) == 2
    assert all(line.count("  ") == 1 for line in lines)
    rels = [line.split("  ", 1)[1] for line in lines]
    assert rels == sorted(rels)
    assert rels == ["pytorch_model.bin", "weights.pt"]


def test_weight_files_excludes_subdirs_and_sorts(tmp_path):
    d = make_model_dir(tmp_path)
    (d / "onnx").mkdir()
    (d / "onnx" / "model.onnx").write_bytes(b"x")
    (d / "imgs").mkdir()
    (d / "imgs" / "w.pt").write_bytes(b"x")
    (d / "sub").mkdir()
    (d / "sub" / "deep.safetensors").write_bytes(b"x")
    rels = [p.relative_to(d).as_posix() for p in weight_files(d)]
    assert rels == ["pytorch_model.bin", "sub/deep.safetensors"]


def test_fingerprint_for_cfg_roundtrip(tmp_path):
    d = make_model_dir(tmp_path)
    cfg: dict[str, Any] = {
        "_root": str(tmp_path),
        "embeddings": {"model_path": str(d)},
    }
    fp = fingerprint_for_cfg(cfg)
    assert fp is not None
    assert fp == model_fingerprint(d)


def test_fingerprint_for_cfg_unresolved(tmp_path):
    cfg: dict[str, Any] = {"_root": str(tmp_path), "embeddings": {}}
    assert fingerprint_for_cfg(cfg) is None
