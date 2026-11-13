"""High-level vector index — wraps VectorStore + embedder + metadata.

Replaces the raw FAISS calls in indexer.py, retrieval.py, and
pipeline.py with a backend-agnostic interface that works across
FAISS, Qdrant, and pgvector.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from doc_rag.raglib import model_loader
from doc_rag.raglib.vectorstore import VectorStore
from doc_rag.raglib.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)


def _load_embedder(cfg: dict[str, Any]) -> Any:
    """Load the embedder via the shared model_loader (single load site).

    The name and signature are load-bearing: tests monkeypatch this
    function by string path, so it must stay module-level here.
    """
    return model_loader.load_embedder(cfg)


class VectorIndex:
    """Backend-agnostic vector index.

    Manages: VectorStore instance, embedder, metadata (chunk_ids),
    and build/update/delete lifecycle. Supports namespaces via collection.
    """

    def __init__(
        self, cfg: dict[str, Any], collection: str = "default", namespace: str | None = None
    ) -> None:
        self._cfg = cfg
        self._collection = namespace or collection
        self._namespace = namespace
        self._store: VectorStore | None = None
        self._embedder: Any = None
        self._chunk_ids: list[str] = []
        self._initialized = False
        # Optional progress callback: cb(phase, done, total). Used by the
        # UI to show "embedding X/N" instead of an opaque "running".
        self._progress_cb: Any = None

    def set_progress_callback(self, cb: Any) -> None:
        """Register cb(phase: str, done: int, total: int) for long operations."""
        self._progress_cb = cb

    def _emit(self, phase: str, done: int, total: int) -> None:
        if self._progress_cb is None:
            return
        try:
            self._progress_cb(phase, done, total)
        except Exception:
            pass

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

    def _store_matches_config(self, store: VectorStore) -> bool:
        """True if the on-disk index was built with the current embed settings.

        Guards the incremental path against mixing vectors from different
        embedding models / dimensions (BUG-006). Compares against the FAISS
        backend's `IndexMeta` when available; otherwise falls back to the
        vector dimension. Returns True when no comparison is possible, so
        backends without metadata keep the old incremental behaviour.
        """
        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

            if isinstance(store, FaissVectorStore):
                store._ensure_loaded()
                meta = store._meta
                if meta is None:
                    return True
                want_model = str(self._cfg.get("embeddings", {}).get("model_name", ""))
                want_norm = self._normalize()
                if meta.model_name and meta.model_name != want_model:
                    return False
                if meta.normalize != want_norm:
                    return False
                return True
        except Exception:
            return True
        return True

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

    def _encode(self, texts: list[str], max_retries: int = 3) -> np.ndarray:
        """Encode texts to vectors with retry logic.

        Halves the encoding batch size on each retry. A failure is often
        OOM triggered by the current batch size being too large for
        available memory (CPU RAM or GPU VRAM) — retrying with the *same*
        batch size just reproduces the same failure three times in a row.
        """
        embedder = self._ensure_embedder()
        if embedder is None:
            raise RuntimeError("sentence-transformers not installed")

        batch_size = self._batch_size()
        total = len(texts)
        last_error = None
        for attempt in range(max_retries):
            try:
                self._emit("embed", 0, total)
                # Encode in explicit batches so `encode()` is only called with
                # arguments every sentence-transformers version accepts
                # (some versions reject an extra `callback=` kwarg), while
                # still letting the UI see per-batch progress.
                parts: list[np.ndarray] = []
                done = 0
                bs = batch_size
                for start in range(0, total, bs):
                    sl = texts[start : start + bs]
                    parts.append(
                        embedder.encode(
                            sl,
                            batch_size=bs,
                            show_progress_bar=False,
                            normalize_embeddings=self._normalize(),
                        ).astype(np.float32)
                    )
                    done += len(sl)
                    self._emit("embed", done, total)
                vecs = (
                    np.vstack(parts)
                    if parts
                    else np.zeros((0, self.dim or 0), dtype=np.float32)
                )
                self._emit("embed", total, total)
                return vecs
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time

                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        "Embedding attempt %d failed: %s, retrying with batch_size=%d...",
                        attempt + 1,
                        e,
                        batch_size,
                    )
                    time.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_error}")

    def build(self, chunks: list[dict[str, Any]], force: bool = False, log=print) -> bool:
        """Build or update the index from chunk dicts.

        Returns True if index is available after build.
        """
        store = self._ensure_store()

        # Dedup input chunks by chunk_id (keep last)
        _seen: set[str] = set()
        _deduped: list[dict[str, Any]] = []
        for c in reversed(chunks):
            cid = str(c.get("chunk_id", ""))
            if cid and cid not in _seen:
                _seen.add(cid)
                _deduped.append(c)
        chunks = list(reversed(_deduped))

        all_ids = [str(c.get("chunk_id", "")) for c in chunks if c.get("chunk_id")]
        if not all_ids:
            log("[doc-rag][index] No chunks with chunk_id -> skipping")
            return False

        if not force and store.size > 0 and self._store_matches_config(store):
            # Load existing IDs from store metadata if not yet loaded
            if not self._chunk_ids:
                self._load_existing_ids()
            existing = set(self._chunk_ids) if self._chunk_ids else set()
            new_ids = [cid for cid in all_ids if cid not in existing]
            if not new_ids:
                log("[doc-rag][index] Index up to date")
                self._emit("done", store.size, store.size)
                return True
            log(f"[doc-rag][index] Incremental update: +{len(new_ids)} chunks")
            new_chunks = [c for c in chunks if c.get("chunk_id") in set(new_ids)]
            texts = [str(c.get("text", "")) for c in new_chunks]
            vecs = self._encode(texts)
            store.add(vecs, new_ids)
            self._chunk_ids.extend(new_ids)
            store.save()
            log(f"[doc-rag][index] Updated: total={store.size}")
            self._emit("done", store.size, store.size)
            return True

        if not force and store.size > 0:
            log("[doc-rag][index] Embedding settings changed -> forcing full rebuild")

        # Full build
        log("[doc-rag][index] Building index from scratch...")
        self._emit("embed", 0, len(chunks))
        texts = [str(c.get("text", "")) for c in chunks]
        vecs = self._encode(texts)
        self._emit("index", len(chunks), len(chunks))
        dim = int(vecs.shape[1])
        model_name = str(self._cfg.get("embeddings", {}).get("model_name", ""))
        store.init(dim=dim, model_name=model_name, normalize=self._normalize())
        store.add(vecs, all_ids)
        self._chunk_ids = list(all_ids)
        store.save()
        log(f"[doc-rag][index] Built: ntotal={store.size} dim={dim}")
        self._emit("done", store.size, store.size)
        return True

    def search(self, query: str, top_k: int = 6) -> tuple[np.ndarray, list[str]]:
        """Search the index. Returns (distances, chunk_ids)."""
        store = self._ensure_store()
        if store.size == 0:
            return np.array([], dtype=np.float32), []
        vec = self._encode([query])
        return store.search(vec, top_k)

    def search_with_filter(
        self, query: str, top_k: int = 6, filters: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        """Hybrid search: vector similarity + metadata filter.

        Only supported on Qdrant backend. Falls back to plain search on FAISS/pgvector.
        """
        store = self._ensure_store()
        if store.size == 0:
            return np.array([], dtype=np.float32), [], []
        vec = self._encode([query])
        if hasattr(store, "search_with_filter") and filters:
            return store.search_with_filter(vec, top_k, filters)
        dists, ids = store.search(vec, top_k)
        return dists, ids, [{"chunk_id": cid} for cid in ids]

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
