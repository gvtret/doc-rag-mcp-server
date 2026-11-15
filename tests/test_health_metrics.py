"""Tests for the /health/{live,ready}, /health, and /metrics endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _client(tmp_corpus_root: Path, monkeypatch: pytest.MonkeyPatch | None = None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    if monkeypatch is not None:
        # /health/ready gates on the model in semantic mode; report a ready
        # model so endpoint-level assertions stay focused on manifest/job.
        from doc_rag.raglib import model_loader as model_loader

        monkeypatch.setattr(
            model_loader,
            "status",
            lambda: {
                "ready": True,
                "path": "/m",
                "fingerprint": "fp",
                "error": None,
                "source": "local",
            },
        )
    return TestClient(app), tmp_corpus_root


def test_health_live_always_200(tmp_corpus_root: Path):
    client, _ = _client(tmp_corpus_root)
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_ready_503_without_manifest(tmp_corpus_root: Path):
    client, root = _client(tmp_corpus_root)
    # tmp_corpus_root deliberately has no manifest yet.
    assert not (root / "build" / "manifest.json").exists()

    r = client.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert "manifest_missing" in body["reasons"]


def test_health_ready_200_with_manifest(tmp_corpus_root: Path, monkeypatch: pytest.MonkeyPatch):
    client, root = _client(tmp_corpus_root, monkeypatch)
    (root / "build" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "documents": []}), encoding="utf-8"
    )
    # Semantic mode also gates on the local index files; stub bytes are
    # enough since the probe only stats them.
    (root / "build" / "index").mkdir(parents=True, exist_ok=True)
    (root / "build" / "index" / "faiss.index").write_bytes(b"stub")
    (root / "build" / "index" / "index_meta.json").write_text(
        json.dumps({"weights_sha256": "fp"}), encoding="utf-8"
    )

    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["reasons"] == []
    # Subsystem checks are informational — present in every response
    assert "checks" in body
    assert "disk_space" in body["checks"]


def test_health_ready_503_when_index_missing(
    tmp_corpus_root: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("fastapi")

    client, root = _client(tmp_corpus_root, monkeypatch)
    (root / "build" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "documents": []}), encoding="utf-8"
    )
    # Manifest present, model ready, but no local index files -> 503.

    r = client.get("/health/ready")

    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert "index_missing" in body["reasons"]


def test_legacy_health_keeps_returning_200(tmp_corpus_root: Path):
    client, _ = _client(tmp_corpus_root)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # New 'ready' field is exposed for clients that want to migrate.
    assert "ready" in body
    assert "reasons" in body


def test_metrics_endpoint_text_when_lib_installed(tmp_corpus_root: Path):
    pytest.importorskip("prometheus_client")
    client, _ = _client(tmp_corpus_root)

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # The Prometheus exposition format is line-oriented "# HELP" + samples.
    assert "doc_rag_mcp_requests_total" in body
    assert "doc_rag_faiss_index_size" in body
