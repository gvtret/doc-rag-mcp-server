"""Startup model warm-up, honest readiness, and branched search notices.

Covers Task 5 of the offline-model-supply plan:

- the FastAPI lifespan loads the embedder before serving (exercised via
  `with TestClient(app)`; a failed load is recorded in model_loader.status()
  and the service keeps running);
- /health/ready fails (503) in semantic/hybrid mode when the model cannot
  load, and passes in lexical mode without one;
- the readiness reasons include model_missing and index_stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _patch_status(monkeypatch: pytest.MonkeyPatch, *, ready: bool, fingerprint: str | None = None):
    from doc_rag.raglib import model_loader as model_loader

    monkeypatch.setattr(
        model_loader,
        "status",
        lambda: {
            "ready": ready,
            "path": "/models/bge-m3" if ready else None,
            "fingerprint": fingerprint,
            "error": None if ready else "no local model directory and no embeddings.model_name",
            "source": "local" if ready else None,
        },
    )


def _write_manifest(root: Path) -> None:
    (root / "build" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "documents": []}), encoding="utf-8"
    )


def _write_index(root: Path, *, weights_sha256: str = "fp") -> None:
    # The readiness probe only stats the files; stub bytes are enough.
    (root / "build" / "index" / "faiss.index").write_bytes(b"stub")
    (root / "build" / "index" / "index_meta.json").write_text(
        json.dumps({"weights_sha256": weights_sha256}), encoding="utf-8"
    )


def _set_retrieval_mode(tmp_corpus_root: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import yaml

    from doc_rag.server import retrieval as r

    cfg_path = tmp_corpus_root / "config" / "config.yaml"
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("mcp", {})["retrieval_mode"] = mode
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # Cache invalidation: retrieval caches load_config() by (root, mtime);
    # clear it explicitly so same-second writes cannot serve stale config.
    monkeypatch.setattr(r, "_CFG_CACHE", {}, raising=False)


# ---------------------------------------------------------------------------
# _readiness_state
# ---------------------------------------------------------------------------


def test_readiness_fails_without_model(tmp_corpus_root, monkeypatch):
    from doc_rag.server.mcp_http import _readiness_state

    _write_manifest(tmp_corpus_root)
    _write_index(tmp_corpus_root, weights_sha256="fp")
    _patch_status(monkeypatch, ready=False)

    state = _readiness_state()
    assert state["ready"] is False
    assert "model_missing" in state["reasons"]
    assert state["checks"]["model_ready"] is False


def test_readiness_ok_with_model(tmp_corpus_root, monkeypatch):
    from doc_rag.server.mcp_http import _readiness_state

    _write_manifest(tmp_corpus_root)
    _write_index(tmp_corpus_root, weights_sha256="fp")
    _patch_status(monkeypatch, ready=True, fingerprint="fp")

    state = _readiness_state()
    assert state["ready"] is True
    assert state["reasons"] == []
    assert state["checks"]["model_ready"] is True


def test_readiness_ok_without_model_in_lexical_mode(tmp_corpus_root, monkeypatch):
    from doc_rag.server.mcp_http import _readiness_state

    _set_retrieval_mode(tmp_corpus_root, monkeypatch, "lexical")
    _write_manifest(tmp_corpus_root)
    _patch_status(monkeypatch, ready=False)

    state = _readiness_state()
    assert state["ready"] is True
    assert "model_missing" not in state["reasons"]
    # The check itself is still reported (informational).
    assert state["checks"]["model_ready"] is False


def test_readiness_ok_without_model_in_hybrid_mode_not_gated(tmp_corpus_root, monkeypatch):
    """Hybrid mode still needs the model: readiness must fail without it."""
    from doc_rag.server.mcp_http import _readiness_state

    _set_retrieval_mode(tmp_corpus_root, monkeypatch, "hybrid")
    _write_manifest(tmp_corpus_root)
    _patch_status(monkeypatch, ready=False)

    state = _readiness_state()
    assert state["ready"] is False
    assert "model_missing" in state["reasons"]


def test_readiness_reports_stale_index(tmp_corpus_root, monkeypatch):
    from doc_rag.server.mcp_http import _readiness_state

    _write_manifest(tmp_corpus_root)
    _write_index(tmp_corpus_root, weights_sha256="old")
    _patch_status(monkeypatch, ready=True, fingerprint="new")

    state = _readiness_state()
    assert "index_stale" in state["reasons"]


# ---------------------------------------------------------------------------
# /health/ready endpoint + lifespan
# ---------------------------------------------------------------------------


def test_health_ready_endpoint_503_when_model_missing(tmp_corpus_root, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    _write_manifest(tmp_corpus_root)
    _patch_status(monkeypatch, ready=False)

    # `with` runs the lifespan (model warm-up); the patched status keeps the
    # readiness gate deterministic regardless of what the warm-up loaded.
    with TestClient(app) as client:
        r = client.get("/health/ready")

    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert "model_missing" in body["reasons"]


def test_lifespan_records_failed_model_load(tmp_corpus_root):
    """`with TestClient(app)` runs the lifespan: a missing model is recorded."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.raglib import model_loader as model_loader
    from doc_rag.server.mcp_http import app

    with TestClient(app) as client:
        st = model_loader.status()
        r = client.get("/health/live")

    # The test config points at a model that cannot load offline ("test-stub"):
    # the warm-up must fail loudly in status(), not crash the service.
    assert r.status_code == 200
    assert st["ready"] is False
    assert st["error"]


# ---------------------------------------------------------------------------
# Notice texts
# ---------------------------------------------------------------------------


def test_stale_notice_mentions_rebuild_and_fetch_model():
    from doc_rag.server.search_tool import _STALE_NOTICE

    assert "rebuild" in _STALE_NOTICE
    assert "fetch_model.py" in _STALE_NOTICE


def test_model_notice_mentions_fetch_model():
    from doc_rag.server.search_tool import _MODEL_NOTICE

    assert "fetch_model.py" in _MODEL_NOTICE
