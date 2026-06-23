"""Abstract vector-store interface for pluggable backends.

Every backend (FAISS, Qdrant, pgvector) implements the `VectorStore`
protocol. The pipeline and retrieval code interact only with this
interface, never with backend-specific APIs directly.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol that all vector-store backends must satisfy."""

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add vectors with their chunk IDs."""
        ...

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, list[str]]:
        """Search for nearest neighbors. Returns (distances, chunk_ids)."""
        ...

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by chunk IDs. Returns count deleted."""
        ...

    def save(self) -> None:
        """Persist current state to disk/remote."""
        ...

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        ...

    def reset(self) -> None:
        """Clear all vectors and metadata."""
        ...
