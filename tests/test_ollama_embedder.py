"""Tests for the ollama embedder plumbing: reachability probe, digest
resolution from Ollama manifests, and fail-closed fingerprint semantics.

Covers the reviewer findings: K2 (loader must not report ready when
Ollama is down), I1 (status fingerprint must be populated), I2 (provider
= ollama must never fall back to a local-dir hash), I3 (corrupted
manifest must not raise).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import pytest

from doc_rag.raglib.model_loader import _load_ollama_embedder, _OLLAMA_PROBE_STATE
from doc_rag.raglib.model_paths import _ollama_model_digest, fingerprint_for_cfg
from doc_rag.raglib.ollama_embedder import build_ollama_embedder, ollama_reachable


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def ollama_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated OLLAMA_MODELS dir with a valid bge-m3 manifest."""
    models = tmp_path / "ollama"
    manifest_dir = models / "manifests" / "registry.ollama.ai" / "library" / "bge-m3"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "config": {"digest": "sha256:" + "ab" * 32},
        "layers": [{"digest": "sha256:" + "cd" * 32}],
    }
    (manifest_dir / "latest").write_text(json.dumps(manifest))
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    return models


class _Handler(BaseHTTPRequestHandler):
    answer: dict = {"models": [{"name": "bge-m3:latest"}]}

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/api/tags":
            body = json.dumps(self.answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args: object) -> None:
        pass  # keep test output quiet


@pytest.fixture()
def ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# ------------------------------------------------------------------ probe

def test_ollama_reachable_true_on_http_server(ollama_server: str):
    assert ollama_reachable(ollama_server) is True


def test_ollama_reachable_false_on_dead_port():
    # Port 1 on localhost is effectively guaranteed closed.
    assert ollama_reachable("http://127.0.0.1:1") is False


# ---------------------------------------------------------------- digest

def test_digest_reads_manifest(ollama_home: Path):
    d = _ollama_model_digest("bge-m3", "http://127.0.0.1:11434")
    assert d == "sha256:" + "ab" * 32


def test_digest_unknown_model_returns_none(ollama_home: Path):
    assert _ollama_model_digest("no-such-model", "http://127.0.0.1:11434") is None


def test_digest_corrupted_manifest_returns_none_not_raises(ollama_home: Path, monkeypatch: pytest.MonkeyPatch):
    m = ollama_home / "manifests" / "registry.ollama.ai" / "library" / "bge-m3" / "latest"
    m.write_text('{"config": {"digest": "sha256:broken"')  # truncated JSON
    # Isolate from the real user/system Ollama homes: the lookup chain must
    # not silently fall through to a manifest of a different installation.
    monkeypatch.setattr(
        "doc_rag.raglib.model_paths._OLLAMA_MANIFEST_ROOTS",
        (lambda: os.environ.get("OLLAMA_MODELS"),),
    )
    assert _ollama_model_digest("bge-m3", "http://127.0.0.1:11434") is None


def test_digest_tag_and_namespace_forms(ollama_home: Path):
    d = _ollama_model_digest("bge-m3:latest", "http://127.0.0.1:11434")
    assert d == "sha256:" + "ab" * 32


# ------------------------------------------------- fingerprint fail-closed

def test_fingerprint_ollama_ok(ollama_home: Path):
    cfg = {"embeddings": {"provider": "ollama", "model_name": "bge-m3"}}
    assert fingerprint_for_cfg(cfg) == "ollama:sha256:" + "ab" * 32


def test_fingerprint_ollama_never_falls_back_to_local_hash(ollama_home: Path):
    """Reviewer I2: unknown model must yield None, not a local-dir hash."""
    cfg = {"embeddings": {"provider": "ollama", "model_name": "myorg/nope:q4"}}
    assert fingerprint_for_cfg(cfg) is None


# ---------------------------------------------------------------- loader

def _reset_probe_cache() -> None:
    _OLLAMA_PROBE_STATE.clear()


def test_loader_ready_true_with_live_ollama(ollama_server: str):
    _reset_probe_cache()
    cfg = {"embeddings": {"provider": "ollama", "model_name": "bge-m3",
                          "api_base": ollama_server}}
    emb = _load_ollama_embedder(cfg)
    assert emb is not None
    from doc_rag.raglib.model_loader import status

    st = status()
    assert st["ready"] is True
    assert st["fingerprint"] is not None  # reviewer I1: not None anymore
    assert st["error"] is None


def test_loader_ready_false_with_dead_ollama():
    """Reviewer K2: dead Ollama must give ready=False + error, not ready=True."""
    _reset_probe_cache()
    cfg = {"embeddings": {"provider": "ollama", "model_name": "bge-m3",
                          "api_base": "http://127.0.0.1:1"}}
    emb = _load_ollama_embedder(cfg)
    assert emb is None
    from doc_rag.raglib.model_loader import status

    st = status()
    assert st["ready"] is False
    assert "unreachable" in str(st["error"])


def test_loader_config_error_reported_not_swallowed():
    """Reviewer I5: ollama config error must surface, not be overwritten."""
    _reset_probe_cache()
    cfg = {"embeddings": {"provider": "ollama", "model_name": ""}}
    emb = _load_ollama_embedder(cfg)
    assert emb is None
    from doc_rag.raglib.model_loader import status

    st = status()
    assert st["ready"] is False
    assert "ollama embedder config error" in str(st["error"])


def test_build_ollama_embedder_requires_model_name():
    with pytest.raises(ValueError):
        build_ollama_embedder({"embeddings": {"provider": "ollama"}})


def test_build_ollama_embedder_none_without_provider():
    assert build_ollama_embedder({"embeddings": {"model_name": "bge-m3"}}) is None
