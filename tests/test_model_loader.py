"""Tests for the single thread-safe model loader."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from doc_rag.raglib import model_loader
from doc_rag.raglib.model_loader import load_embedder, reset_for_tests, status


@pytest.fixture(autouse=True)
def _fresh_loader():
    reset_for_tests()
    yield
    reset_for_tests()


def make_model_dir(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    (d / "pytorch_model.bin").write_bytes(b"w")
    return d


class FakeST:
    """Records constructor calls; fakes a sentence-transformers model."""

    calls: list[dict[str, Any]] = []
    device_used: str | None = None
    fail_on: set[str] = set()

    def __init__(self, path, device=None, local_files_only=False, **kw):
        record = {
            "path": path,
            "device": device,
            "local_files_only": local_files_only,
        }
        type(self).calls.append(record)
        if device in type(self).fail_on:
            raise RuntimeError(f"device {device} failed")
        self.max_seq_length = 0
        type(self).device_used = device


def test_load_uses_local_files_only(tmp_path, monkeypatch):
    d = make_model_dir(tmp_path)
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    model = load_embedder(cfg)
    assert model is not None
    assert FakeST.calls == [{"path": str(d), "device": "cpu", "local_files_only": True}]
    st = status()
    assert st["ready"] is True
    assert st["path"] == str(d)
    assert st["source"] == "local"
    assert st["error"] is None


def test_single_flight_loads_once(tmp_path, monkeypatch):
    make_model_dir(tmp_path)

    class SlowST(FakeST):
        calls = []
        device_used = None
        fail_on = set()

        def __init__(self, path, device=None, local_files_only=False, **kw):
            super().__init__(path, device=device, local_files_only=local_files_only, **kw)

    monkeypatch.setattr(model_loader, "SentenceTransformer", SlowST)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    results: list[Any] = []
    errors: list[Exception] = []
    gate = threading.Event()

    original_init = SlowST.__init__

    def slow_init(self, *args, **kw):
        gate.wait(5)  # hold the constructor until all threads are inside
        original_init(self, *args, **kw)

    monkeypatch.setattr(SlowST, "__init__", slow_init)

    def worker():
        try:
            results.append(load_embedder(cfg))
        except Exception as exc:  # pragma: no cover - collected for diagnostics
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert not errors
    assert len(SlowST.calls) == 1
    assert len(results) == 8
    assert all(r is results[0] for r in results)
    assert all(r is not None for r in results)


def test_missing_model_returns_none_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "SentenceTransformer", None)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert load_embedder(cfg) is None
    st = status()
    assert st["ready"] is False
    assert st["error"]


def test_bad_model_path_degrades(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{}")  # no modules.json -> not a model
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {"model_path": str(bad)}}
    assert load_embedder(cfg) is None
    st = status()
    assert st["ready"] is False
    assert st["error"]


def test_failure_on_device_retries_cpu(tmp_path, monkeypatch):
    make_model_dir(tmp_path)
    FakeST.calls = []
    FakeST.fail_on = {"cuda"}
    FakeST.device_used = None
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {"device": "cuda"}}
    model = load_embedder(cfg)
    assert model is not None
    assert [c["device"] for c in FakeST.calls] == ["cuda", "cpu"]
    assert FakeST.device_used == "cpu"
    assert status()["ready"] is True


def test_failure_on_cpu_returns_none(tmp_path, monkeypatch):
    make_model_dir(tmp_path)
    FakeST.calls = []
    FakeST.fail_on = {"cpu", "cuda"}
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert load_embedder(cfg) is None
    assert status()["ready"] is False
    assert status()["error"]


def test_max_seq_length_applied(tmp_path, monkeypatch):
    make_model_dir(tmp_path)
    FakeST.calls = []
    FakeST.fail_on = set()
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {"max_seq_length": 512}}
    model = load_embedder(cfg)
    assert model is not None
    assert model.max_seq_length == 512


def test_hf_cache_fallback(tmp_path, monkeypatch):
    # No local directory; model_name present -> load by name, offline.
    FakeST.calls = []
    FakeST.fail_on = set()
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {
        "_root": str(tmp_path),
        "embeddings": {"model_name": "BAAI/bge-large-en-v1.5"},
    }
    model = load_embedder(cfg)
    assert model is not None
    assert FakeST.calls == [
        {
            "path": "BAAI/bge-large-en-v1.5",
            "device": "cpu",
            "local_files_only": True,
        }
    ]
    st = status()
    assert st["ready"] is True
    assert st["source"] == "hf_cache"
    assert st["path"] == "BAAI/bge-large-en-v1.5"


def test_cached_second_call_skips_constructor(tmp_path, monkeypatch):
    make_model_dir(tmp_path)
    FakeST.calls = []
    FakeST.fail_on = set()
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    first = load_embedder(cfg)
    second = load_embedder(cfg)
    assert first is second
    assert len(FakeST.calls) == 1
    assert current_embedder_matches(first)


def current_embedder_matches(model):
    from doc_rag.raglib.model_loader import current_embedder

    return current_embedder() is model
