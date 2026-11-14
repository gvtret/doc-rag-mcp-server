"""FAISS vector-store backend.

Holds the FAISS-specific code that used to live in raglib/indexer.py,
behind the VectorStore protocol. Retains full compatibility with the
current on-disk format (faiss.index + index_meta.json).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from doc_rag.raglib.atomic import atomic_write_text, fsync_dir

logger = logging.getLogger(__name__)


@dataclass
class IndexMeta:
    protocol_version: str
    model_name: str
    metric: str
    normalize: bool
    dim: int
    chunk_ids: list[str] = field(default_factory=list)
    weights_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "model_name": self.model_name,
            "metric": self.metric,
            "normalize": self.normalize,
            "dim": self.dim,
            "chunk_ids": self.chunk_ids,
            "weights_sha256": self.weights_sha256,
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
            weights_sha256=str(data.get("weights_sha256", "")),
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
            index = faiss.read_index(self._index_path)
            with open(self._meta_path, encoding="utf-8") as f:
                meta = IndexMeta.from_dict(json.load(f))
            if int(index.ntotal) != len(meta.chunk_ids):
                # Fail closed (crash window between the index replace and the
                # meta write in save()): a positional mapping built from this
                # pair would serve wrong chunks. Degrade instead of serving.
                logger.error(
                    "faiss: index integrity check failed: ntotal=%d != len(chunk_ids)=%d "
                    "for %s -- degrading to unavailable (rebuild the index)",
                    int(index.ntotal),
                    len(meta.chunk_ids),
                    self._index_path,
                )
                self._index = None
                self._meta = None
                self._id_to_pos = {}
            else:
                self._index = index
                self._meta = meta
                self._id_to_pos = {cid: i for i, cid in enumerate(meta.chunk_ids)}
                logger.info(
                    "faiss: loaded %d vectors (dim=%d) from %s",
                    meta.dim,
                    len(meta.chunk_ids),
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

    def init(
        self, dim: int, model_name: str, normalize: bool = True, weights_sha256: str = ""
    ) -> None:
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
            weights_sha256=weights_sha256,
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
        # Position base must be captured before appending: computing it as
        # len(chunk_ids) inside the loop yields 0, 2, 4, ... after the first
        # element lands in chunk_ids.
        base = len(self._meta.chunk_ids)
        for i, cid in enumerate(ids):
            self._id_to_pos[cid] = base + i
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

        ids_set = set(ids)
        keep_positions = []
        keep_ids = []
        removed = 0
        for i, cid in enumerate(self._meta.chunk_ids):
            if cid in ids_set:
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

        # Reconstruct straight into the final array instead of building a
        # Python list of per-vector arrays and `np.vstack`-ing it — that
        # briefly doubles peak memory (list + the vstack copy) for a large
        # index, which matters on a CPU-only, memory-constrained server.
        vecs = np.empty((len(keep_positions), self._meta.dim), dtype=np.float32)
        for out_i, p in enumerate(keep_positions):
            vecs[out_i] = self._index.reconstruct(p)

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

        index_dir = os.path.dirname(self._index_path) or "."
        os.makedirs(index_dir, exist_ok=True)
        # Atomic sequence: the index is written to a temp file in the same
        # directory, fsync'ed, then renamed over the target; meta is then
        # written via atomic_write_text. Each file on disk is always a
        # complete, untruncated version (old or new), never a torn write.
        #
        # Known window: a crash AFTER os.replace(index) but BEFORE the meta
        # write completes leaves a NEW index with the OLD meta (e.g. ntotal
        # counts vectors that meta's chunk_ids do not list). The pair is not
        # guaranteed mutually consistent in that case; detecting and
        # repairing such drift at load time is out of scope here (tracked
        # as a separate consistency-check task). A crash before the replace
        # leaves the old index file completely intact.
        tmp_index_path = self._index_path + ".tmp"
        try:
            faiss.write_index(self._index, tmp_index_path)
            # faiss.write_index writes and closes internally, so flush+fsync
            # on the handle is impossible; fsync via a fresh read fd of the
            # tmp file so the rename is crash-safe on the data level.
            fd = os.open(tmp_index_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_index_path, self._index_path)
        except BaseException:
            try:
                os.unlink(tmp_index_path)
            except OSError:
                pass
            raise
        meta_json = json.dumps(self._meta.to_dict(), indent=2)
        atomic_write_text(self._meta_path, meta_json)
        fsync_dir(index_dir)
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
