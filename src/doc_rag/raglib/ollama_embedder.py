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
import time
import urllib.parse
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough guard: chunker targets ~512 tokens -> ~2Ki chars
_EMBED_RETRIES = 3
_RETRY_BASE_SLEEP = 0.5  # seconds; exponential with jitter, between attempts only


class OllamaEmbedder:
    """Minimal sentence-transformers-compatible client for Ollama /api/embed."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://127.0.0.1:11434",
        normalize: bool = True,
        max_seq_length: int | None = None,
        timeout: float = 120.0,
        dim: int | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        # Kept for parity with the sentence-transformers surface; encoding
        # normalization is driven per-call by normalize_embeddings (an
        # explicit L2 pass on the returned matrix), not by this flag.
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.timeout = timeout
        self._dim = dim
        self._conn: http.client.HTTPConnection | None = None
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
        if not sentences:
            # Dimension contract mirrors sentence-transformers: known-dim
            # instances report (0, dim) even for empty input.
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        limit = (self._max_seq or 4096) * _CHARS_PER_TOKEN
        trimmed = [s if len(s) <= limit else s[:limit] for s in sentences]

        out: list[list[float]] = []
        for start in range(0, len(trimmed), batch_size):
            chunk = trimmed[start : start + batch_size]
            out.extend(self._post(chunk))
        vecs = np.asarray(out, dtype=np.float32)
        if normalize_embeddings and vecs.size:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs = vecs / norms
        return vecs

    # -- internals ----------------------------------------------------------

    def _connection(self) -> http.client.HTTPConnection:
        """Reuse one keep-alive connection; rebuild silently when the
        server closed it (Ollama keeps connections alive, but idle timeouts
        and restarts happen)."""
        if self._conn is None:
            if self.api_base.startswith("https://"):
                self._conn = http.client.HTTPSConnection(
                    self._host, self._port, timeout=self.timeout
                )
            else:
                self._conn = http.client.HTTPConnection(
                    self._host, self._port, timeout=self.timeout
                )
        return self._conn

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        last_exc: Exception | None = None
        for attempt in range(_EMBED_RETRIES):
            try:
                conn = self._connection()
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
                return emb
            except (OSError, ValueError) as e:
                last_exc = e
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
        raise RuntimeError(f"Ollama embed failed after retries: {last_exc}")


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


def build_ollama_embedder(cfg: dict[str, Any]) -> OllamaEmbedder | None:
    """Construct an OllamaEmbedder from cfg, or None if not configured."""
    e = cfg.get("embeddings") or {}
    if str(e.get("provider") or "").strip().lower() != "ollama":
        return None
    model = str(e.get("model_name") or "").strip()
    if not model:
        raise ValueError("embeddings.provider=ollama requires embeddings.model_name")
    return OllamaEmbedder(
        model=model,
        api_base=str(e.get("api_base") or "http://127.0.0.1:11434"),
        normalize=bool(e.get("normalize", True)),
        max_seq_length=e.get("max_seq_length"),
    )
