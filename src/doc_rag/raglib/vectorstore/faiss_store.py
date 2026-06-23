"""FAISS vector-store backend.

Wraps the existing FAISS-specific code from indexer.py into the
VectorStore protocol. Retains full compatibility with the current
on-disk format (faiss.index + index_meta.json).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexMeta:
    protocol_version: str
    model_name: str
    metric: str
    normalize: bool
    dim: int
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "model_name": self.model_name,
            "metric": self.metric,
            "normalize": self.normalize,
            "dim": self.dim,
            "chunk_ids": self.chunk_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMeta:
        return cls(
            protocol_version=str(data.get("protocol_version", "1.0.0")),
            model_name=str(data.get("model_name", "")),
            metric=str(data.get("metric", "ip")),
            normalize=bool(data.get("normalize", True)),
            dim=int(data.get("dim", 0)),
            chunk_ids=list(data.get("chunk_ids", [])),
        )


class FaissVectorStore:
    """FAISS-backed vector store implementing the VectorStore protocol."""

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        metric: str = "ip",
    ) -> None:
        self._index_path = index_path
        self._meta_path = meta_path
        self._metric = metric
        self._index: Any = None
        self._meta: IndexMeta | None = None
        self._id_to_pos: dict[str, int] = {}

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is not installed. Install with: uv sync --extra faiss"
            ) from exc

        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, encoding="utf-8") as f:
                self._meta = IndexMeta.from_dict(json.load(f))
            self._id_to_pos = {cid: i for i, cid in enumerate(self._meta.chunk_ids)}
            logger.info(
                "faiss: loaded %d vectors (dim=%d) from %s",
                self._meta.dim,
                len(self._meta.chunk_ids),
                self._index_path,
            )
        else:
            self._meta = IndexMeta(
                protocol_version="1.0.0",
                model_name="",
                metric=self._metric,
                normalize=True,
                dim=0,
            )
            self._index = None
            self._id_to_pos = {}

    def init(self, dim: int, model_name: str, normalize: bool = True) -> None:
        """Initialize a new empty index with the given dimensionality."""
        import faiss

        self._dim = dim
        self._metric_type = "ip" if self._metric == "ip" else "l2"
        if self._metric_type == "ip":
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)
        self._meta = IndexMeta(
            protocol_version="1.0.0",
            model_name=model_name,
            metric=self._metric,
            normalize=normalize,
            dim=dim,
        )
        self._id_to_pos = {}
        logger.info("faiss: initialized empty %s index (dim=%d)", self._metric_type, dim)

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        self._ensure_loaded()
        if self._index is None:
            raise RuntimeError("Index not initialized. Call init() first.")
        n = vectors.shape[0]
        if n != len(ids):
            raise ValueError(f"vectors count ({n}) != ids count ({len(ids)})")
        self._index.add(vectors)
        for i, cid in enumerate(ids):
            self._id_to_pos[cid] = len(self._meta.chunk_ids) + i
            self._meta.chunk_ids.append(cid)
        logger.info("faiss: added %d vectors (total=%d)", n, self._index.ntotal)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, list[str]]:
        self._ensure_loaded()
        if self._index is None or self._index.ntotal == 0:
            return np.array([], dtype=np.float32), []
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query, k)
        ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self._meta.chunk_ids):
                ids.append(self._meta.chunk_ids[idx])
            else:
                ids.append("")
        return distances[0][:k], ids

    def delete(self, ids: list[str]) -> int:
        self._ensure_loaded()
        if self._index is None or not ids:
            return 0
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is not installed") from exc

        keep_positions = []
        keep_ids = []
        removed = 0
        for i, cid in enumerate(self._meta.chunk_ids):
            if cid in ids:
                removed += 1
            else:
                keep_positions.append(i)
                keep_ids.append(cid)

        if removed == 0:
            return 0

        if not keep_positions:
            self._index = None
            self._meta.chunk_ids = []
            self._id_to_pos = {}
            return removed

        vecs = np.vstack([self._index.reconstruct(p) for p in keep_positions]).astype(np.float32)

        if self._meta.metric == "ip":
            self._index = faiss.IndexFlatIP(self._meta.dim)
        else:
            self._index = faiss.IndexFlatL2(self._meta.dim)
        self._index.add(vecs)

        self._meta.chunk_ids = keep_ids
        self._id_to_pos = {cid: i for i, cid in enumerate(keep_ids)}
        logger.info("faiss: deleted %d vectors (remaining=%d)", removed, self._index.ntotal)
        return removed

    def save(self) -> None:
        self._ensure_loaded()
        if self._index is None:
            return
        import faiss

        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta.to_dict(), f, indent=2)
        logger.info("faiss: saved %d vectors to %s", self._index.ntotal, self._index_path)

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return self._index.ntotal if self._index else 0

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._meta.dim if self._meta else 0

    def reset(self) -> None:
        self._index = None
        self._meta = None
        self._id_to_pos = {}
        for p in (self._index_path, self._meta_path):
            if os.path.exists(p):
                os.remove(p)
        logger.info("faiss: reset (deleted %s)", self._index_path)

    def reconstruct(self, chunk_id: str) -> np.ndarray | None:
        """Get the raw vector for a chunk ID."""
        self._ensure_loaded()
        if self._index is None:
            return None
        pos = self._id_to_pos.get(chunk_id)
        if pos is None:
            return None
        return self._index.reconstruct(pos)

    def get_meta(self) -> IndexMeta | None:
        self._ensure_loaded()
        return self._meta

    def set_meta(self, meta: IndexMeta) -> None:
        self._meta = meta
