"""High-level vector index — wraps VectorStore + embedder + metadata.

Replaces the raw FAISS calls in indexer.py, retrieval.py, and
pipeline.py with a backend-agnostic interface that works across
FAISS, Qdrant, and pgvector.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from doc_rag.raglib.vectorstore import VectorStore
from doc_rag.raglib.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)


def _load_embedder(cfg: dict[str, Any]) -> Any:
    """Load sentence-transformers embedder (same logic as indexer.py)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    model_name = cfg.get("embeddings", {}).get("model_name", "BAAI/bge-large-en-v1.5")
    device = cfg.get("embeddings", {}).get("device", "cpu")
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None


class VectorIndex:
    """Backend-agnostic vector index.

    Manages: VectorStore instance, embedder, metadata (chunk_ids),
    and build/update/delete lifecycle.
    """

    def __init__(self, cfg: dict[str, Any], collection: str = "default") -> None:
        self._cfg = cfg
        self._collection = collection
        self._store: VectorStore | None = None
        self._embedder: Any = None
        self._chunk_ids: list[str] = []
        self._initialized = False

    def _load_existing_ids(self) -> None:
        """Load existing chunk IDs from the store metadata if available."""
        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

            store = self._ensure_store()
            if isinstance(store, FaissVectorStore):
                store._ensure_loaded()
                if store._meta and store._meta.chunk_ids:
                    self._chunk_ids = list(store._meta.chunk_ids)
        except Exception:
            pass

    def _ensure_store(self) -> VectorStore:
        if self._store is None:
            self._store = create_vector_store(self._cfg, self._collection)
        return self._store

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = _load_embedder(self._cfg)
        return self._embedder

    def _normalize(self) -> bool:
        return bool(self._cfg.get("embeddings", {}).get("normalize", True))

    def _batch_size(self) -> int:
        return int(self._cfg.get("embeddings", {}).get("batch_size", 32))

    def _encode(self, texts: list[str]) -> np.ndarray:
        embedder = self._ensure_embedder()
        if embedder is None:
            raise RuntimeError("sentence-transformers not installed")
        return embedder.encode(
            texts,
            batch_size=self._batch_size(),
            show_progress_bar=False,
            normalize_embeddings=self._normalize(),
        ).astype(np.float32)

    def build(self, chunks: list[dict[str, Any]], force: bool = False, log=print) -> bool:
        """Build or update the index from chunk dicts.

        Returns True if index is available after build.
        """
        store = self._ensure_store()
        all_ids = [str(c.get("chunk_id", "")) for c in chunks if c.get("chunk_id")]
        if not all_ids:
            log("[doc-rag][index] No chunks with chunk_id -> skipping")
            return False

        if not force and store.size > 0:
            existing = set(self._chunk_ids) if self._chunk_ids else set()
            new_ids = [cid for cid in all_ids if cid not in existing]
            if not new_ids:
                log("[doc-rag][index] Index up to date")
                return True
            log(f"[doc-rag][index] Incremental update: +{len(new_ids)} chunks")
            new_chunks = [c for c in chunks if c.get("chunk_id") in set(new_ids)]
            texts = [str(c.get("text", "")) for c in new_chunks]
            vecs = self._encode(texts)
            store.add(vecs, new_ids)
            self._chunk_ids.extend(new_ids)
            store.save()
            log(f"[doc-rag][index] Updated: total={store.size}")
            return True

        # Full build
        log("[doc-rag][index] Building index from scratch...")
        texts = [str(c.get("text", "")) for c in chunks]
        vecs = self._encode(texts)
        dim = int(vecs.shape[1])
        model_name = str(self._cfg.get("embeddings", {}).get("model_name", ""))
        store.init(dim=dim, model_name=model_name, normalize=self._normalize())
        store.add(vecs, all_ids)
        self._chunk_ids = list(all_ids)
        store.save()
        log(f"[doc-rag][index] Built: ntotal={store.size} dim={dim}")
        return True

    def search(self, query: str, top_k: int = 6) -> tuple[np.ndarray, list[str]]:
        """Search the index. Returns (distances, chunk_ids)."""
        store = self._ensure_store()
        if store.size == 0:
            return np.array([], dtype=np.float32), []
        vec = self._encode([query])
        return store.search(vec, top_k)

    def delete(self, doc_ids: set[str]) -> int:
        """Delete all chunks belonging to the given doc_ids."""
        store = self._ensure_store()
        self._load_existing_ids()
        to_delete = [cid for cid in self._chunk_ids if cid.split(":")[0] in doc_ids]
        if not to_delete:
            return 0
        removed = store.delete(to_delete)
        self._chunk_ids = [cid for cid in self._chunk_ids if cid not in set(to_delete)]
        if not self._chunk_ids:
            store.reset()
        else:
            store.save()
        logger.info("vectorindex: deleted %d chunks for %d docs", removed, len(doc_ids))
        return removed

    def reset(self) -> None:
        store = self._ensure_store()
        store.reset()
        self._chunk_ids = []
        logger.info("vectorindex: reset")

    @property
    def size(self) -> int:
        return self._ensure_store().size

    @property
    def dim(self) -> int:
        return self._ensure_store().dim

    def is_available(self) -> bool:
        """Check if the index is loaded and has vectors."""
        try:
            store = self._ensure_store()
            return store.size > 0
        except Exception:
            return False
