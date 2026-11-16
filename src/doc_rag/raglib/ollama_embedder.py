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
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough guard: chunker targets ~512 tokens -> ~2Ki chars


class OllamaEmbedder:
    """Minimal sentence-transformers-compatible client for Ollama /api/embed."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://127.0.0.1:11434",
        normalize: bool = True,
        max_seq_length: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.timeout = timeout

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

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self.api_base}/api/embed",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                emb = data.get("embeddings")
                if not emb:
                    raise ValueError(f"ollama embed returned no vectors: {str(data)[:200]}")
                if self._dim is None:
                    self._dim = len(emb[0])
                return emb
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                last_exc = e
                logger.warning("Ollama embed attempt %d failed: %s", attempt + 1, e)
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama embed failed after retries: {last_exc}")

    _dim: int | None = None


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
