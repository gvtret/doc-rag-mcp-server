"""Ollama-backed embedder wrapper.

Implements the sentence-transformers ``encode()`` surface used by
vector_index._encode() (positional texts, batch_size, show_progress_bar,
normalize_embeddings) on top of Ollama's ``POST /api/embed`` endpoint.

Enabled via config:

    embeddings:
      provider: "ollama"
      api_base: "http://127.0.0.1:11434"
      model_name: "bge-m3"

When ``provider`` is absent the loader keeps loading a local
SentenceTransformer exactly as before, so this is fully opt-in.

Notes:
- ``normalize_embeddings=True`` relies on Ollama's bge-m3 output already
  being L2-normalized (the served model normalizes by default); an extra
  explicit normalization is applied anyway to make the contract explicit.
- Input is capped at ``max_seq_length`` *approximate* tokens by character
  budget (4 chars/token heuristic) because Ollama's embed endpoint has no
  truncation parameter; the chunker already sizes chunks to ~512 tokens,
  so the cap only guards degenerate inputs.
- One keep-alive HTTP connection per instance (M6): full rebuilds issue
  ~630 sequential /api/embed calls, and a fresh TCP handshake per batch
  adds measurable latency on localhost.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.parse
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough guard: chunker targets ~512 tokens -> ~2Ki chars
_EMBED_RETRIES = 3
_RETRY_BASE_SLEEP = 0.5  # seconds; exponential with jitter, between attempts only


class EmbeddingUnavailable(RuntimeError):
    """Raised when embeddings cannot be produced quickly by design.

    Two causes: the circuit breaker is open (Ollama has been timing out),
    or a single request hit the timeout. Callers catch it and degrade to
    lexical search instead of waiting out retries and blowing the MCP
    tool budget (30 s) with a 504.
    """


class CircuitBreaker:
    """Thread-safe consecutive-failure breaker with a cooldown window.

    States: ``closed`` (requests allowed), ``open`` (fail fast), and
    ``half_open`` (exactly one probe request in flight after the cooldown).
    A failed probe re-opens the breaker immediately; a successful one
    closes it. ``snapshot()["state"]`` is the authoritative view --
    callers must treat ``half_open`` as degraded, because it is not yet
    proven that the backend recovered.

    threshold: number of consecutive timeouts that opens the breaker.
    cooldown: seconds the breaker stays open before a probe is allowed.
    clock: monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        threshold: int = 2,
        cooldown: float = 60.0,
        clock: Any = time.monotonic,
    ) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown = max(0.0, float(cooldown))
        self._clock = clock
        self._state = "closed"
        self._failures = 0
        self._open_until = 0.0  # monotonic instant while state == "open"
        self._lock = threading.Lock()

    def configure(self, threshold: int, cooldown: float) -> None:
        """Update policy from config without losing current state."""
        with self._lock:
            self.threshold = max(1, int(threshold))
            self.cooldown = max(0.0, float(cooldown))

    def allow(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "half_open":
                return False  # one probe at a time
            if self._clock() < self._open_until:
                return False
            # Cooldown elapsed: transition to half-open and hand out the
            # single probe slot.
            self._state = "half_open"
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            was_probe = self._state == "half_open"
            self._failures += 1
            if was_probe or self._failures >= self.threshold:
                # A failed probe proves the backend is still broken: reopen
                # immediately instead of granting another full threshold.
                self._state = "open"
                self._open_until = self._clock() + self.cooldown

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            return {
                "state": self._state,
                "open": self._state == "open",
                "degraded": self._state != "closed",
                "consecutive_failures": self._failures,
                "retry_in_seconds": (
                    round(max(0.0, self._open_until - now), 2) if self._state == "open" else 0.0
                ),
                "threshold": self.threshold,
                "cooldown_seconds": self.cooldown,
            }


_BREAKERS: dict[tuple[str, str], CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def breaker_for(
    api_base: str,
    model: str,
    threshold: int = 2,
    cooldown: float = 60.0,
) -> CircuitBreaker:
    """Shared breaker per (api_base, model) for *interactive* embeddings.

    Query and build paths create separate embedder instances (the server
    loader vs. a VectorIndex), so sharing here is what makes a timeout on
    one query fail fast on the next. Bulk index builds deliberately bypass
    the breaker (see OllamaEmbedder._post(enforce_breaker=False)): a slow
    large batch must not abort hours of rebuild work, and two query
    timeouts must not kill a rebuild.
    """
    key = (api_base.rstrip("/"), model)
    with _BREAKERS_LOCK:
        breaker = _BREAKERS.get(key)
        if breaker is None:
            breaker = CircuitBreaker(threshold=threshold, cooldown=cooldown)
            _BREAKERS[key] = breaker
        else:
            # Keep the deployment config authoritative across config reloads.
            breaker.configure(threshold, cooldown)
        return breaker


def reset_breakers_for_tests() -> None:
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


class OllamaEmbedder:
    """Minimal sentence-transformers-compatible client for Ollama /api/embed.

    ``supports_query_deadline`` is an explicit capability marker: only this
    class has a per-query short timeout. Call sites must test the marker,
    not ``hasattr(embedder, "encode_query")`` -- sentence-transformers
    itself exposes ``encode_query`` (which injects a prompt and would change
    the query representation silently in the local-model branch).
    """

    supports_query_deadline = True

    def __init__(
        self,
        model: str,
        api_base: str = "http://127.0.0.1:11434",
        normalize: bool = True,
        max_seq_length: int | None = None,
        timeout: float = 120.0,
        dim: int | None = None,
        query_timeout: float | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        # Kept for parity with the sentence-transformers surface; encoding
        # normalization is driven per-call by normalize_embeddings (an
        # explicit L2 pass on the returned matrix), not by this flag.
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        # timeout guards bulk index builds; query_timeout guards interactive
        # query embeddings and must stay below the MCP tool budget.
        self.timeout = float(timeout)
        self.query_timeout = float(query_timeout) if query_timeout is not None else float(timeout)
        self.breaker = breaker if breaker is not None else CircuitBreaker()
        self._dim = dim
        self._conn: http.client.HTTPConnection | None = None
        self._conn_timeout: float | None = None
        # HTTPConnection is not thread-safe and one embedder instance is
        # shared by concurrent tool calls (and by the bulk path). Serialize
        # request/response cycles; the RLock allows _connection() and
        # _drop_connection() to be called while it is held.
        self._io_lock = threading.RLock()
        parsed = urllib.parse.urlsplit(self.api_base)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 11434)

    # -- sentence-transformers compatible surface --------------------------

    @property
    def max_seq_length(self) -> int | None:  # noqa: D401 - property kept simple
        return self._max_seq

    @max_seq_length.setter
    def max_seq_length(self, value: int | None) -> None:
        self._max_seq = int(value) if value else None

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # accepted, ignored (HTTP batches)
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        return self._encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            timeout=self.timeout,
            enforce_breaker=False,  # bulk builds must not be killed by queries
        )

    def encode_query(
        self,
        sentences: list[str],
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        """Encode an interactive query with the short query timeout.

        Used by vector_index search paths only; index builds keep using
        encode() with the long bulk timeout. The shared breaker applies to
        this path only.
        """
        return self._encode(
            sentences,
            batch_size=1,
            normalize_embeddings=normalize_embeddings,
            timeout=self.query_timeout,
            enforce_breaker=True,
        )

    def _encode(
        self,
        sentences: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        timeout: float,
        enforce_breaker: bool,
    ) -> np.ndarray:
        if not sentences:
            # Dimension contract mirrors sentence-transformers: known-dim
            # instances report (0, dim) even for empty input.
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        limit = (self._max_seq or 4096) * _CHARS_PER_TOKEN
        trimmed = [s if len(s) <= limit else s[:limit] for s in sentences]

        out: list[list[float]] = []
        for start in range(0, len(trimmed), batch_size):
            chunk = trimmed[start : start + batch_size]
            out.extend(self._post(chunk, timeout=timeout, enforce_breaker=enforce_breaker))
        vecs = np.asarray(out, dtype=np.float32)
        if normalize_embeddings and vecs.size:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs = vecs / norms
        return vecs

    def health(self) -> dict[str, Any]:
        """Breaker state plus the configured timeouts (for /health)."""
        snapshot = self.breaker.snapshot()
        snapshot.update(
            {
                "model": self.model,
                "api_base": self.api_base,
                "timeout_seconds": self.timeout,
                "query_timeout_seconds": self.query_timeout,
            }
        )
        return snapshot

    # -- internals ----------------------------------------------------------

    def _new_connection(self, timeout: float) -> http.client.HTTPConnection:
        """A dedicated connection for one interactive request (no shared socket)."""
        if self.api_base.startswith("https://"):
            return http.client.HTTPSConnection(self._host, self._port, timeout=timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        """Reuse one keep-alive connection per timeout value.

        Query embeddings use a shorter timeout than bulk builds, so a
        connection created for one timeout cannot be reused for the other
        (the socket timeout is fixed at creation). Switching closes the old
        one; same-timeout calls keep the keep-alive benefit.
        Caller must hold ``_io_lock``.
        """
        if self._conn is not None and self._conn_timeout != timeout:
            self._drop_connection()
        if self._conn is None:
            if self.api_base.startswith("https://"):
                self._conn = http.client.HTTPSConnection(self._host, self._port, timeout=timeout)
            else:
                self._conn = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
            self._conn_timeout = timeout
        return self._conn

    def _drop_connection(self) -> None:
        with self._io_lock:
            conn, self._conn = self._conn, None
            self._conn_timeout = None
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _post(
        self,
        texts: list[str],
        timeout: float | None = None,
        *,
        enforce_breaker: bool = True,
    ) -> list[list[float]]:
        timeout = float(timeout) if timeout is not None else self.timeout
        if enforce_breaker and not self.breaker.allow():
            snapshot = self.breaker.snapshot()
            if snapshot.get("state") == "half_open":
                detail = "recovery probe already in progress"
            else:
                detail = f"retry in {snapshot.get('retry_in_seconds', 0)}s"
            raise EmbeddingUnavailable(
                f"Ollama embedding circuit is open (server is timing out); {detail}"
            )
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        last_exc: Exception | None = None
        decided = False
        try:
            for attempt in range(_EMBED_RETRIES):
                try:
                    if enforce_breaker:
                        # Interactive queries use a dedicated connection:
                        # sharing one serialized socket turned N concurrent
                        # queries into a queue of N x timeout (504 for all
                        # but the first). A fresh localhost connection costs
                        # fractions of a millisecond.
                        conn = self._new_connection(timeout)
                        try:
                            conn.request(
                                "POST",
                                "/api/embed",
                                body=payload,
                                headers={"Content-Type": "application/json"},
                            )
                            resp = conn.getresponse()
                            body = resp.read()
                        finally:
                            try:
                                conn.close()
                            except Exception:
                                pass
                    else:
                        with self._io_lock:
                            conn = self._connection(timeout)
                            conn.request(
                                "POST",
                                "/api/embed",
                                body=payload,
                                headers={"Content-Type": "application/json"},
                            )
                            resp = conn.getresponse()
                            body = resp.read()
                    data = json.loads(body)
                    emb = data.get("embeddings")
                    if not emb:
                        raise ValueError(f"ollama embed returned no vectors: {str(data)[:200]}")
                    if self._dim is None:
                        self._dim = len(emb[0])
                    if len(emb) != len(texts):
                        # A desync here would silently misalign chunk->vector
                        # mapping and surface later as an opaque FAISS size
                        # error; fail loudly at the source instead.
                        raise ValueError(
                            f"ollama embed returned {len(emb)} vectors for {len(texts)} inputs"
                        )
                    if enforce_breaker:
                        self.breaker.record_success()
                        decided = True
                    return emb
                except TimeoutError as e:
                    # Fail fast: a timeout means the server is wedged (e.g. the
                    # model was loaded on CPU). Retrying would multiply the wait
                    # by _EMBED_RETRIES and guarantee a 504 on the MCP layer.
                    if not enforce_breaker:
                        self._drop_connection()
                    if enforce_breaker:
                        self.breaker.record_failure()
                        decided = True
                    logger.warning(
                        "Ollama embed timed out after %.3gs (not retrying): %s", timeout, e
                    )
                    raise EmbeddingUnavailable(f"Ollama embed timed out after {timeout:g}s") from e
                except (OSError, ValueError, http.client.HTTPException) as e:
                    last_exc = e
                    if not enforce_breaker:
                        self._drop_connection()  # broken pipe / closed keep-alive
                    if attempt + 1 < _EMBED_RETRIES:
                        # Deterministic jitter by attempt: spreads retries
                        # without a cryptographically-flagged RNG (S311).
                        sleep_s = _RETRY_BASE_SLEEP * (2**attempt) + 0.1 * (attempt + 1)
                        logger.warning(
                            "Ollama embed attempt %d/%d failed: %s (retrying in %.2fs)",
                            attempt + 1,
                            _EMBED_RETRIES,
                            e,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
            if enforce_breaker:
                # Retries exhausted by a non-timeout error (connection refused
                # during an Ollama restart, malformed response, ...). Record it:
                # otherwise a failed half-open probe would leave the breaker
                # stuck in half_open forever and the service degraded until a
                # process restart.
                self.breaker.record_failure()
                decided = True
                raise EmbeddingUnavailable(
                    f"Ollama embed failed after retries: {last_exc}"
                ) from last_exc
            # Bulk path: keep RuntimeError so VectorIndex._encode still runs
            # its batch-halving retry (the OOM rescue path documented in
            # VectorIndex._encode).
            raise RuntimeError(f"Ollama embed failed after retries: {last_exc}") from last_exc
        finally:
            # Safety net for unexpected exception types escaping the loop.
            if (
                enforce_breaker
                and not decided
                and self.breaker.snapshot().get("state") == "half_open"
            ):
                self.breaker.record_failure()


def ollama_reachable(api_base: str, timeout: float = 2.0) -> bool:
    """Cheap reachability probe: GET /api/tags on the Ollama server.

    Used by the model loader so the service reports ready=False (and
    doc_search degrades to lexical results) when Ollama is down, instead
    of failing every search after retries.
    """
    try:
        parsed = urllib.parse.urlsplit(api_base.rstrip("/"))
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 11434)
        if parsed.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=timeout
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/api/tags")
            resp = conn.getresponse()
            resp.read()
            return 200 <= resp.status < 500  # any sane HTTP answer = alive
        finally:
            conn.close()
    except Exception:
        return False


def _ollama_host_port(api_base: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(api_base.rstrip("/"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 11434)
    return host, port


def ollama_backend_state(api_base: str, model: str, timeout: float = 1.0) -> str:
    """Report where Ollama is running the requested embedding model.

    Values: "gpu" (fully on GPU), "partial" (0 < size_vram < size, partial
    offload), "cpu" (model resident, all in RAM), "unloaded" (the requested
    model is not resident), "unreachable". A different resident model is
    never reported as the requested one: that would mask exactly the
    incident this field exists for (requested model not resident, a chat
    model in VRAM -> a false "gpu"). This is the direct signal for the
    2026-11-25 incident: Ollama registered a CPU-only inference backend
    and embeddings became hundreds of times slower while /api/tags stayed
    perfectly healthy.
    """
    try:
        host, port = _ollama_host_port(api_base)
        if api_base.startswith("https://"):
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=timeout
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/api/ps")
            resp = conn.getresponse()
            body = resp.read()
            if not (200 <= resp.status < 300):
                return "unreachable"
        finally:
            conn.close()
        data = json.loads(body or b"{}")
        models = data.get("models") or []
        want = model.split(":")[0]
        chosen = next(
            (m for m in models if str(m.get("name") or "").split(":")[0] == want),
            None,
        )
        if chosen is None:
            return "unloaded"
        size_vram = int(chosen.get("size_vram") or 0)
        size = int(chosen.get("size") or 0)
        if size_vram > 0:
            return "partial" if size_vram < size else "gpu"
        if size > 0:
            return "cpu"
        return "unloaded"
    except Exception:
        return "unreachable"


def build_ollama_embedder(cfg: dict[str, Any]) -> OllamaEmbedder | None:
    """Construct an OllamaEmbedder from cfg, or None if not configured."""
    e = cfg.get("embeddings") or {}
    if str(e.get("provider") or "").strip().lower() != "ollama":
        return None
    model = str(e.get("model_name") or "").strip()
    if not model:
        raise ValueError("embeddings.provider=ollama requires embeddings.model_name")
    api_base = str(e.get("api_base") or "http://127.0.0.1:11434")

    def _positive_float(key: str, default: float) -> float:
        try:
            value = float(e.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _positive_int(key: str, default: int) -> int:
        try:
            value = int(e.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return OllamaEmbedder(
        model=model,
        api_base=api_base,
        normalize=bool(e.get("normalize", True)),
        max_seq_length=e.get("max_seq_length"),
        timeout=_positive_float("request_timeout_seconds", 120.0),
        query_timeout=_positive_float("query_timeout_seconds", 20.0),
        breaker=breaker_for(
            api_base,
            model,
            threshold=_positive_int("breaker_threshold", 2),
            cooldown=_positive_float("breaker_cooldown_seconds", 60.0),
        ),
    )
