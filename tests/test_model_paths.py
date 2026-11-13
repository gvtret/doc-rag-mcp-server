"""Tests for local embedding-model resolution and weights fingerprinting."""

from __future__ import annotations

from typing import Any

from doc_rag.raglib.model_paths import (
    _parse_sha256_file,
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


def test_resolve_prefers_model_path(tmp_path):
    custom = make_model_dir(tmp_path)
    (tmp_path / "models" / "bge-m3").rename(custom)
    cfg = {"_root": str(tmp_path), "embeddings": {"model_path": str(custom)}}
    assert resolve_model_path(cfg) == custom


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
    # The refreshed manifest matches the new weights.
    assert fp2 is not None
    assert fingerprint_from_pairs(_parse_sha256_file(d / "WEIGHTS.sha256")) == fp2


def test_model_fingerprint_none_without_weights(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    assert model_fingerprint(d) is None
    assert not (d / "WEIGHTS.sha256").exists()


def test_fingerprint_from_pairs_is_order_insensitive():
    pairs = [("b" * 64, "b.bin"), ("a" * 64, "a.bin")]
    assert fingerprint_from_pairs(pairs) == fingerprint_from_pairs(reversed(pairs))


def test_write_weights_sha256_sorted_sha256sum_format(tmp_path):
    d = make_model_dir(tmp_path)
    (d / "weights.pt").write_bytes(b"pt")
    out = write_weights_sha256(d)
    assert out == d / "WEIGHTS.sha256"
    lines = out.read_text().strip().splitlines()
    assert lines == sorted(lines)
    assert all("  " in line for line in lines)
    assert len(lines) == 2
    assert any(line.endswith("  pytorch_model.bin") for line in lines)


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
