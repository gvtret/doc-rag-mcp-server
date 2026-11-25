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
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from doc_rag.raglib.model_loader import _OLLAMA_PROBE_STATE, _load_ollama_embedder
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


def test_digest_corrupted_manifest_returns_none_not_raises(
    ollama_home: Path, monkeypatch: pytest.MonkeyPatch
):
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
    cfg = {"embeddings": {"provider": "ollama", "model_name": "bge-m3", "api_base": ollama_server}}
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
    cfg = {
        "embeddings": {
            "provider": "ollama",
            "model_name": "bge-m3",
            "api_base": "http://127.0.0.1:1",
        }
    }
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


# ------------------------------------------------ circuit breaker / fail-fast
#
# Incident 2026-11-25: Ollama started without GPU (discovery watchdog raced
# the WSL2 GPU init), bge-m3 ran on CPU, one embedding batch took minutes,
# and every MCP tool call hit the 30 s server budget and returned
# 504 Gateway Timeout. The tests below pin the intended contract:
#   * a timeout is not retried (retrying multiplies the wait),
#   * consecutive timeouts open a breaker that fails fast,
#   * query embeddings use a much shorter timeout than bulk index builds,
#   * degraded state is visible in model_loader.status() and /health.

from doc_rag.raglib.ollama_embedder import (  # noqa: E402
    CircuitBreaker,
    EmbeddingUnavailable,
    OllamaEmbedder,
    ollama_backend_state,
    reset_breakers_for_tests,
)


class _FakeClock:
    """Controllable monotonic clock for breaker cooldown tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _EmbedHandler(BaseHTTPRequestHandler):
    """Fake /api/embed (POST) and /api/ps (GET).

    Class-level knobs are reset by the fixture so tests stay independent.
    """

    requests = 0
    delay = 0.0
    ps_payload: dict = {"models": []}
    body_override: str | None = None
    fail_status: int | None = None

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        type(self).requests += 1
        length = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(length) or b"{}")
        if type(self).delay:
            time.sleep(type(self).delay)
        if type(self).fail_status is not None:
            self.send_error(type(self).fail_status)
            return
        if type(self).body_override is not None:
            body = type(self).body_override.encode()
        else:
            inputs = data.get("input") or []
            body = json.dumps({"embeddings": [[0.1, 0.2, 0.3] for _ in inputs]}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client timed out first; that is exactly what we test

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/api/ps":
            self.send_error(404)
            return
        body = json.dumps(type(self).ps_payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def embed_server():
    _EmbedHandler.requests = 0
    _EmbedHandler.delay = 0.0
    _EmbedHandler.ps_payload = {"models": []}
    _EmbedHandler.body_override = None
    _EmbedHandler.fail_status = None
    reset_breakers_for_tests()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    reset_breakers_for_tests()


def test_circuit_breaker_single_half_open_probe_and_reopen():
    clock = _FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown=60.0, clock=clock)

    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.snapshot()["state"] == "open"
    assert breaker.allow() is False

    # Cooldown elapsed: exactly one probe is allowed; half-open counts as
    # degraded until the probe succeeds.
    clock.now += 61.0
    assert breaker.allow() is True
    assert breaker.allow() is False
    snap = breaker.snapshot()
    assert snap["state"] == "half_open"
    assert snap["degraded"] is True

    # A failed probe re-opens immediately (does not reset to closed).
    breaker.record_failure()
    assert breaker.snapshot()["state"] == "open"

    # After the next cooldown, a successful probe closes it.
    clock.now += 61.0
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.snapshot()["state"] == "closed"
    assert breaker.snapshot()["degraded"] is False


def test_embed_timeout_is_not_retried(embed_server: str):
    _EmbedHandler.delay = 0.4
    embedder = OllamaEmbedder("bge-m3", api_base=embed_server, timeout=0.1)

    started = time.monotonic()
    with pytest.raises(EmbeddingUnavailable):
        embedder.encode(["one"])
    elapsed = time.monotonic() - started

    # One HTTP attempt only: a timeout means the server is wedged, and
    # retrying would multiply the wait instead of recovering.
    assert _EmbedHandler.requests == 1
    assert elapsed < 0.35


def test_breaker_short_circuits_after_threshold(embed_server: str):
    _EmbedHandler.delay = 0.4
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=0.1,
        query_timeout=0.1,
        breaker=CircuitBreaker(threshold=2, cooldown=3600.0),
    )

    for _ in range(2):
        with pytest.raises(EmbeddingUnavailable):
            embedder.encode_query(["one"])
    assert _EmbedHandler.requests == 2

    with pytest.raises(EmbeddingUnavailable) as excinfo:
        embedder.encode_query(["one"])
    assert "circuit" in str(excinfo.value).lower()
    # Fail-fast: no third HTTP request was issued.
    assert _EmbedHandler.requests == 2


def test_bulk_build_is_not_blocked_by_query_breaker(embed_server: str):
    """Two query timeouts must not abort (or be aborted by) an index build."""
    _EmbedHandler.delay = 0.4
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=0.1,
        query_timeout=0.1,
        breaker=CircuitBreaker(threshold=1, cooldown=3600.0),
    )
    with pytest.raises(EmbeddingUnavailable):
        embedder.encode_query(["query"])
    assert embedder.breaker.snapshot()["state"] == "open"

    # Bulk path ignores the interactive breaker (it has its own long timeout)
    # so a rebuild is never killed by query traffic.
    _EmbedHandler.delay = 0.0
    assert embedder.encode(["chunk"]).shape == (1, 3)


def test_bulk_timeout_does_not_open_breaker(embed_server: str):
    """A slow bulk batch must not trip the interactive breaker."""
    _EmbedHandler.delay = 0.4
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=0.1,
        breaker=CircuitBreaker(threshold=1, cooldown=3600.0),
    )
    with pytest.raises(EmbeddingUnavailable):
        embedder.encode(["chunk"])
    assert embedder.breaker.snapshot()["state"] == "closed"


def test_half_open_probe_failing_without_timeout_reopens(monkeypatch: pytest.MonkeyPatch):
    """Connection refused during a recovery probe must reopen the breaker.

    Regression guard for the stuck-half_open defect: before the fix only a
    TimeoutError recorded a verdict, so a probe that hit a closed port during
    an Ollama restart left the breaker half-open forever and every query
    degraded until a process restart.
    """
    import doc_rag.raglib.ollama_embedder as ollama_embedder_module

    monkeypatch.setattr(ollama_embedder_module, "_EMBED_RETRIES", 1)
    clock = _FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown=60.0, clock=clock)
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base="http://127.0.0.1:1",  # connection refused
        timeout=0.05,
        query_timeout=0.05,
        breaker=breaker,
    )

    breaker.record_failure()
    assert breaker.snapshot()["state"] == "open"

    clock.now += 61.0
    with pytest.raises(EmbeddingUnavailable):
        embedder.encode_query(["query"])

    assert breaker.snapshot()["state"] == "open"
    clock.now += 61.0
    assert breaker.allow() is True  # recovery still possible


def test_half_open_probe_unexpected_exception_reopens(embed_server: str):
    """The finally-safety net must verdict even for unexpected exceptions."""
    _EmbedHandler.body_override = json.dumps({"embeddings": [5]})
    clock = _FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown=60.0, clock=clock)
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=0.5,
        query_timeout=0.5,
        breaker=breaker,
    )
    breaker.record_failure()
    clock.now += 61.0

    with pytest.raises(TypeError):
        embedder.encode_query(["query"])

    assert breaker.snapshot()["state"] == "open"


def test_bulk_exhaustion_keeps_batch_halving_retry(embed_server: str, monkeypatch):
    """Bulk failures must stay RuntimeError so VectorIndex halves the batch."""
    import doc_rag.raglib.ollama_embedder as ollama_embedder_module
    import doc_rag.raglib.vector_index as vector_index_module

    monkeypatch.setattr(ollama_embedder_module, "_EMBED_RETRIES", 1)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _EmbedHandler.fail_status = 500

    embedder = OllamaEmbedder("bge-m3", api_base=embed_server, timeout=0.2)
    index = vector_index_module.VectorIndex.__new__(vector_index_module.VectorIndex)
    index._embedder = embedder
    index._cfg = {"embeddings": {"batch_size": 4}}
    index._progress_cb = None

    with pytest.raises(RuntimeError):
        index._encode(["a", "b"], max_retries=3)

    # Three outer attempts x one inner attempt = 3 requests: the batch-halving
    # rescue path was preserved (EmbeddingUnavailable would yield 1).
    assert _EmbedHandler.requests == 3


def test_parallel_queries_are_not_serialized(embed_server: str):
    """N concurrent queries must each take ~one timeout, not N x timeout.

    Regression guard for the shared keep-alive socket serialization: with
    query_timeout=25 s and a 30 s MCP budget, a queue of 4 would turn every
    call after the first into a 504.
    """
    _EmbedHandler.delay = 0.8
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=5.0,
        query_timeout=0.5,
        breaker=CircuitBreaker(threshold=100, cooldown=1.0),
    )
    durations: list[float] = []
    lock = threading.Lock()

    def work() -> None:
        started = time.monotonic()
        with pytest.raises(EmbeddingUnavailable):
            embedder.encode_query(["query"])
        with lock:
            durations.append(time.monotonic() - started)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(durations) == 4
    # Serialized implementation would take ~4 x 0.5 s = 2.0 s for the last
    # caller; the 1.2 s bound keeps a 2.4x margin over a single timeout while
    # still failing on serialization even on a loaded CI runner.
    assert max(durations) < 1.2


def test_query_timeout_is_shorter_than_bulk_timeout(embed_server: str):
    _EmbedHandler.delay = 0.25
    embedder = OllamaEmbedder(
        "bge-m3",
        api_base=embed_server,
        timeout=5.0,
        query_timeout=0.05,
        breaker=CircuitBreaker(threshold=10, cooldown=1.0),
    )

    with pytest.raises(EmbeddingUnavailable):
        embedder.encode_query(["short query"])

    _EmbedHandler.delay = 0.0
    vectors = embedder.encode(["chunk text"])
    assert vectors.shape == (1, 3)


def test_ollama_backend_state_cpu_gpu_unloaded(embed_server: str):
    _EmbedHandler.ps_payload = {"models": [{"name": "bge-m3:latest", "size": 100, "size_vram": 0}]}
    assert ollama_backend_state(embed_server, "bge-m3") == "cpu"

    _EmbedHandler.ps_payload = {
        "models": [{"name": "bge-m3:latest", "size": 100, "size_vram": 100}]
    }
    assert ollama_backend_state(embed_server, "bge-m3") == "gpu"

    # Partial offload must not be reported as a full GPU placement.
    _EmbedHandler.ps_payload = {"models": [{"name": "bge-m3:latest", "size": 100, "size_vram": 40}]}
    assert ollama_backend_state(embed_server, "bge-m3") == "partial"

    _EmbedHandler.ps_payload = {"models": []}
    assert ollama_backend_state(embed_server, "bge-m3") == "unloaded"

    assert ollama_backend_state("http://127.0.0.1:1", "bge-m3", timeout=0.5) == "unreachable"


def test_ollama_backend_state_ignores_other_resident_models(embed_server: str):
    """A resident chat model must not be reported as the embedding model."""
    _EmbedHandler.ps_payload = {"models": [{"name": "qwen2.5:7b", "size": 4000, "size_vram": 4000}]}
    assert ollama_backend_state(embed_server, "bge-m3") == "unloaded"


def test_query_path_skips_sentence_transformers_encode_query(embed_server: str):
    """The short-deadline path must only fire for embedders that opt in.

    sentence-transformers 5.x has its own encode_query(prompt_name="query")
    which would silently change the query representation; the capability
    marker keeps the local-model branch on plain encode().
    """
    from doc_rag.raglib import vector_index as vector_index_module

    class _STLike:
        def encode(self, texts, **kwargs):
            self.used = "encode"
            return np.ones((len(texts), 3), dtype=np.float32)

        def encode_query(self, *args, **kwargs):  # must not be called
            raise AssertionError("SentenceTransformer.encode_query must not be used")

    index = vector_index_module.VectorIndex.__new__(vector_index_module.VectorIndex)
    index._embedder = _STLike()
    index._cfg = {"embeddings": {}}
    index._progress_cb = None

    vectors = index._encode(["query"], max_retries=1, query=True)

    assert index._embedder.used == "encode"
    assert vectors.shape == (1, 3)


def test_loader_sets_current_embedder_for_ollama(ollama_server: str):
    from doc_rag.raglib import model_loader

    model_loader.reset_for_tests()
    _reset_probe_cache()
    cfg = {
        "embeddings": {
            "provider": "ollama",
            "model_name": "bge-m3",
            "api_base": ollama_server,
        }
    }
    emb = _load_ollama_embedder(cfg)
    try:
        assert emb is not None
        assert model_loader.current_embedder() is emb
    finally:
        model_loader.reset_for_tests()


def test_status_reports_degraded_when_breaker_open(monkeypatch: pytest.MonkeyPatch):
    from doc_rag.raglib import model_loader

    embedder = OllamaEmbedder(
        "bge-m3",
        api_base="http://127.0.0.1:1",
        breaker=CircuitBreaker(threshold=1, cooldown=60.0),
    )
    embedder.breaker.record_failure()
    monkeypatch.setattr(model_loader, "_current", embedder)

    status = model_loader.status()
    assert status["degraded"] is True
    assert status["breaker"]["open"] is True


def test_build_ollama_embedder_reads_timeouts_and_breaker_settings():
    emb = build_ollama_embedder(
        {
            "embeddings": {
                "provider": "ollama",
                "model_name": "bge-m3",
                "request_timeout_seconds": 7,
                "query_timeout_seconds": 3,
                "breaker_threshold": 5,
                "breaker_cooldown_seconds": 11,
            }
        }
    )
    assert emb is not None
    assert emb.timeout == 7.0
    assert emb.query_timeout == 3.0
    assert emb.breaker.threshold == 5
    assert emb.breaker.cooldown == 11.0
