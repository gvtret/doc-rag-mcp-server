"""High-level vector index — wraps VectorStore + embedder + metadata.

Replaces the raw FAISS calls that used to live in the legacy indexer.py
(now doc_rag.raglib.vectorstore.faiss_store), retrieval.py, and
pipeline.py with a backend-agnostic interface that works across
FAISS, Qdrant, and pgvector.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from doc_rag.raglib import model_loader, model_paths
from doc_rag.raglib.chunking_meta import config_fingerprint
from doc_rag.raglib.embedding_text import build_embedding_text
from doc_rag.raglib.vectorstore import VectorStore
from doc_rag.raglib.vectorstore.factory import create_vector_store

if TYPE_CHECKING:
    from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)

# Metadata pre-filter threshold: below this many allowed vectors the query is
# scored exactly against their reconstructed vectors; above it an oversampled
# ANN search with post-filtering is cheaper.
_EXACT_SCAN_LIMIT = 5000


def _supported_filter_keys() -> set[str]:
    return {"doc_id", "doc_ids", "section_path", "is_table"}


def _validate_filter_keys(filters: dict[str, Any]) -> None:
    """Fail loud on unknown filter keys: a typo like table_only would
    otherwise be silently ignored and the client would receive unfiltered
    results believing the filter applied."""
    unknown = set(filters) - _supported_filter_keys()
    if unknown:
        raise ValueError(
            f"unsupported filter keys: {sorted(unknown)}; "
            f"supported: {sorted(_supported_filter_keys())}"
        )


def _chunk_matches_filters(chunk_id: str, meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """True if a chunk passes the metadata filters.

    Keys: doc_id (str equality), doc_ids (list membership), section_path
    (prefix match), is_table (bool: True -> table chunks only, False ->
    non-table chunks only). doc_id falls back to the chunk_id prefix when
    the metadata lacks an explicit doc_id. Unknown keys are rejected by
    _validate_filter_keys before this runs.
    """
    doc_id = str(meta.get("doc_id", "") or chunk_id.split(":")[0])
    if "doc_id" in filters and doc_id != str(filters["doc_id"]):
        return False
    if "doc_ids" in filters:
        want = filters["doc_ids"]
        want_list = [str(d) for d in want] if isinstance(want, (list, tuple, set)) else [str(want)]
        if doc_id not in want_list:
            return False
    if "section_path" in filters:
        sec = str(meta.get("section_path", "") or "")
        if not sec.startswith(str(filters["section_path"])):
            return False
    if "is_table" in filters:
        if bool(meta.get("is_table")) != bool(filters["is_table"]):
            return False
    return True


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
        """Load existing chunk IDs from the store metadata if available.

        Fail-closed policy: when the store metadata exists but cannot be
        read (corrupt JSON, unreadable file), existing ids are NOT silently
        treated as empty -- that would turn a rebuild-worthy index into a
        duplicate-vector incremental append. The incremental path is
        guarded by _store_matches_config, which already refuses such stores
        (full rebuild); here we only log at the right severity. A missing
        meta file is the legacy pre-meta layout and stays functional.
        """
        store: Any = None
        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

            store = self._ensure_store()
            if isinstance(store, FaissVectorStore):
                store._ensure_loaded()
                meta = store._meta
                if meta is not None and meta.chunk_ids:
                    self._chunk_ids = list(meta.chunk_ids)
                elif meta is None or not meta.chunk_ids:
                    # Legacy pre-meta layout (or nothing built yet): WARN,
                    # ids stay empty, incremental/rebuild proceeds.
                    logger.warning(
                        "no readable index metadata for %s: treating existing ids as empty",
                        getattr(store, "_meta_path", "<unknown>"),
                    )
        except Exception as e:
            meta_path = getattr(store, "_meta_path", None)
            if meta_path and os.path.exists(meta_path):
                logger.error(
                    "could not read index metadata %s (%s): refusing to treat "
                    "existing ids as empty; full rebuild required",
                    meta_path,
                    e,
                )
            else:
                logger.warning("no readable index metadata (%s): empty id list", e)

    def _store_matches_config(self, store: VectorStore) -> bool:
        """True if the on-disk index was built with the current embed settings.

        Guard against mixing vectors from different embedding models /
        dimensions / weight revisions (BUG-006). Compares against the
        FAISS backend's `IndexMeta`; backends without metadata keep the
        old incremental behaviour (True). Errors reading the store
        metadata or computing the fingerprint fail closed (False): an
        unverifiable index must be rebuilt, not incrementally mixed.
        Documented compromise: when the model resolves via the shared
        HF cache (hf_cache deployments) there is no local model directory
        to fingerprint, so `fingerprint_for_cfg` returns None and this
        weights check is skipped for such deployments.

        Both fingerprint gates (weights sha256 and the chunking/text
        config fingerprint) are FAISS-only: non-FAISS backends return
        True and keep the legacy incremental behaviour; the other
        stores accept `config_fingerprint` in init() but do not
        persist it.
        """
        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

            is_faiss = isinstance(store, FaissVectorStore)
        except Exception:
            # Backend check itself failed: keep the legacy fail-open behaviour.
            return True
        if not is_faiss:
            return True
        try:
            store._ensure_loaded()
            meta = store._meta
        except Exception:
            # Unreadable store metadata: cannot verify settings -> rebuild.
            logger.warning("could not read index metadata: refusing incremental update")
            return False
        if meta is None:
            # No meta at all: either nothing built yet, or a legacy
            # pre-meta index. Only the latter reaches here with vectors
            # on disk; the ntotal/chunk_ids integrity check lives in
            # _ensure_loaded, so a consistent-looking store with a
            # missing meta still goes through the full-rebuild gate
            # below via the model fingerprint check.
            return True
        want_model = str(self._cfg.get("embeddings", {}).get("model_name", ""))
        want_norm = self._normalize()
        if meta.model_name and meta.model_name != want_model:
            return False
        if meta.normalize != want_norm:
            return False
        try:
            want_fp = model_paths.fingerprint_for_cfg(self._cfg) or ""
        except Exception as e:
            # Weights unreadable: cannot prove the index matches them.
            logger.warning(
                "could not compute weights fingerprint (%s): refusing incremental update",
                e,
            )
            return False
        try:
            want_cfg_fp = config_fingerprint(self._cfg)
        except Exception as e:
            logger.warning(
                "could not compute config fingerprint (%s): refusing incremental update", e
            )
            return False
        have_cfg_fp = str(meta.config_fingerprint or "")
        if want_cfg_fp:
            if not have_cfg_fp:
                # Legacy index built before config fingerprints existed:
                # its vectors/texts may come from any chunking scheme.
                logger.warning("index predates config fingerprint: full rebuild required")
                return False
            if have_cfg_fp != want_cfg_fp:
                logger.warning(
                    "index config fingerprint mismatch (have=%s want=%s): full rebuild required",
                    have_cfg_fp[:12],
                    want_cfg_fp[:12],
                )
                return False
        have_fp = str(meta.weights_sha256 or "")
        if want_fp:
            if not have_fp:
                # Legacy index built before fingerprints existed:
                # its vectors may come from any revision of the
                # weights, so incremental mixing is unsafe.
                logger.warning("index has no weights fingerprint: full rebuild required")
                return False
            if have_fp != want_fp:
                logger.warning(
                    "index weights fingerprint mismatch (have=%s want=%s): full rebuild required",
                    have_fp[:12],
                    want_fp[:12],
                )
                return False
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
                vecs = np.vstack(parts) if parts else np.zeros((0, self.dim or 0), dtype=np.float32)
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

    def build(
        self,
        chunks: list[dict[str, Any]],
        force: bool = False,
        log=print,
        title_hints: dict[str, str] | None = None,
    ) -> bool:
        """Build or update the index from chunk dicts.

        title_hints maps doc_id -> document title hint and feeds the
        TITLE part of the embedding context prefix: the EMBEDDING text of
        each chunk becomes "{title} > {section_path}" (via
        build_embedding_text). title_hints=None only disables the title
        part -- the section_path part of the prefix still applies (that
        is the recall fix itself, independent of titles). Stored chunk
        text and served results are NOT changed -- only the encoded
        input. Queries are encoded without a prefix.

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

        # Read the store size defensively for FAISS stores: a corrupt
        # index_meta.json makes plain `store.size` raise. That must NOT
        # skip the index update with a WARN (pipeline would log and move
        # on) -- it must fall through to a full rebuild, same as any other
        # unverifiable-index condition. Non-FAISS stores keep the direct
        # call: transient size errors there should surface, not silently
        # trigger a full re-embed.
        needs_full_rebuild = False
        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

            is_faiss = isinstance(store, FaissVectorStore)
        except Exception:
            is_faiss = False
        if is_faiss:
            try:
                store_size = store.size
            except Exception as e:
                logger.warning("could not read index size (%s): forcing full rebuild", e)
                needs_full_rebuild = True
        else:
            store_size = store.size

        if (
            not needs_full_rebuild
            and not force
            and store_size > 0
            and self._store_matches_config(store)
        ):
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
            new_id_set = set(new_ids)
            new_chunks = [c for c in chunks if c.get("chunk_id") in new_id_set]
            texts = [self._embedding_text(c, title_hints) for c in new_chunks]
            vecs = self._encode(texts)
            store.add(vecs, new_ids)
            self._chunk_ids.extend(new_ids)
            store.save()
            log(f"[doc-rag][index] Updated: total={store.size}")
            self._emit("done", store.size, store.size)
            return True

        if not needs_full_rebuild and not force and store_size > 0:
            log("[doc-rag][index] Embedding settings changed -> forcing full rebuild")

        self._run_full_build(chunks, log, title_hints)
        return True

    @staticmethod
    def _embedding_text(chunk: dict[str, Any], title_hints: dict[str, str] | None) -> str:
        """Text to encode for `chunk`: raw text or a context-prefixed one.

        title_hints maps doc_id -> title (fallback: the doc_id's chunk_id
        prefix before ":" -- the pipeline always sets doc_id, but direct
        callers may pass chunks without it).
        """
        title = ""
        if title_hints:
            doc_id = str(chunk.get("doc_id", "") or str(chunk.get("chunk_id", "")).split(":")[0])
            title = title_hints.get(doc_id, "") or ""
        return build_embedding_text(chunk, title)

    def _run_full_build(
        self,
        chunks: list[dict[str, Any]],
        log: Any,
        title_hints: dict[str, str] | None = None,
    ) -> None:
        """Rebuild the index from scratch over `chunks`.

        Extracted from build() so the corrupt-meta fallthrough reuses the
        exact full-build sequence (fingerprint-first, same log lines).
        """
        store = self._ensure_store()
        all_ids = [str(c.get("chunk_id", "")) for c in chunks if c.get("chunk_id")]
        log("[doc-rag][index] Building index from scratch...")
        # Fingerprint FIRST (fail-closed early): if fetch_model.py swaps the
        # weights while a 30-min encode is running, meta must record the
        # fingerprint of the weights the vectors were actually built with.
        # Computing it after _encode would legitimize mixing old vectors with
        # a new fingerprint.
        model_name = str(self._cfg.get("embeddings", {}).get("model_name", ""))
        weights_sha256 = model_paths.fingerprint_for_cfg(self._cfg) or ""
        cfg_fingerprint = config_fingerprint(self._cfg) or ""
        self._emit("embed", 0, len(chunks))
        texts = [self._embedding_text(c, title_hints) for c in chunks]
        vecs = self._encode(texts)
        self._emit("index", len(chunks), len(chunks))
        dim = int(vecs.shape[1])
        store.init(
            dim=dim,
            model_name=model_name,
            normalize=self._normalize(),
            weights_sha256=weights_sha256,
            config_fingerprint=cfg_fingerprint,
        )
        store.add(vecs, all_ids)
        self._chunk_ids = list(all_ids)
        store.save()
        log(f"[doc-rag][index] Built: ntotal={store.size} dim={dim}")
        self._emit("done", store.size, store.size)

    def search(self, query: str, top_k: int = 6) -> tuple[np.ndarray, list[str]]:
        """Search the index. Returns (distances, chunk_ids)."""
        store = self._ensure_store()
        if store.size == 0:
            return np.array([], dtype=np.float32), []
        vec = self._encode([query])
        return store.search(vec, top_k)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
        chunk_meta: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        """Search with metadata filters applied.

        Supported filter keys: doc_id (str), doc_ids (list), section_path
        (prefix match), is_table (bool).

        Backends with native filter support (Qdrant) handle filters
        server-side. FAISS has no payloads, so the caller passes chunk_meta
        (chunk_id -> {"doc_id", "section_path", "is_table"}): allowed chunk
        ids are pre-filtered by metadata, then scored exactly against their
        vectors (reconstruct + dot scan, up to 5000 allowed), or via
        oversampled search + post-filter above that.

        Fail-loud policy: filters given without chunk_meta raise ValueError
        instead of silently returning unfiltered results; unknown filter
        keys raise ValueError listing the supported keys. is_table=False
        is honored symmetrically (non-table chunks only). With no filters
        the old behavior is kept (plain search, synthetic payloads).
        """
        store = self._ensure_store()
        if store.size == 0:
            return np.array([], dtype=np.float32), [], []
        if not filters:
            vec = self._encode([query])
            dists, ids = store.search(vec, top_k)
            return dists, ids, [{"chunk_id": cid} for cid in ids]

        _validate_filter_keys(filters)

        if hasattr(store, "search_with_filter"):
            return store.search_with_filter(self._encode([query]), top_k, filters)

        if not chunk_meta:
            # None AND {}: both mean "no metadata available" and must not
            # silently degrade into unfiltered or empty results.
            raise ValueError(
                "filters require chunk metadata (chunk_meta) on this backend; "
                "refusing to silently ignore the filters"
            )

        allowed_ids = [
            cid for cid, meta in chunk_meta.items() if _chunk_matches_filters(cid, meta, filters)
        ]
        if not allowed_ids:
            return np.array([], dtype=np.float32), [], []

        try:
            from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore
        except Exception:
            FaissVectorStore = None  # type: ignore[misc,assignment]
        if FaissVectorStore is not None and isinstance(store, FaissVectorStore):
            return self._faiss_filtered_search(store, query, top_k, allowed_ids)

        # Non-FAISS backend without native filter support: oversample and
        # post-filter so the filters are still enforced, not ignored.
        vec = self._encode([query])
        oversample = min(max(top_k * 20, top_k), store.size)
        dists, ids = store.search(vec, oversample)
        allowed = set(allowed_ids)
        kept = [(d, cid) for d, cid in zip(dists, ids, strict=False) if cid in allowed]
        kept = kept[:top_k]
        out_d = np.array([d for d, _ in kept], dtype=np.float32)
        return out_d, [cid for _, cid in kept], [{"chunk_id": cid} for _, cid in kept]

    def _faiss_filtered_search(
        self, store: FaissVectorStore, query: str, top_k: int, allowed_ids: list[str]
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        """Score the query against the allowed vectors (FAISS).

        Exact metric-aware scan when the allowed set is small enough
        (<= _EXACT_SCAN_LIMIT): IP -> inner product, higher is better;
        L2 -> squared euclidean distance, LOWER is better (scores match
        plain store.search distances, so both orders and magnitudes agree
        with the unfiltered path). Above the limit an oversampled ANN
        search with post-filtering runs. Returns at most top_k results:
        exact min(top_k, len(allowed)) on the exact branch, at most top_k
        on the oversample branch (an ANN slice may miss allowed chunks).
        """
        vec = self._encode([query])
        positions = store.positions_for(allowed_ids)
        if not positions:
            return np.array([], dtype=np.float32), [], []
        dim = int(store.dim)
        if dim <= 0:
            raise RuntimeError("faiss store has no dimension; cannot scan")
        if len(positions) <= _EXACT_SCAN_LIMIT:
            sub = np.empty((len(positions), dim), dtype=np.float32)
            for i, p in enumerate(positions):
                sub[i] = store._index.reconstruct(p)
            metric = str(getattr(store.get_meta(), "metric", "ip") or "ip").lower()
            if metric == "l2":
                # Squared L2 distance via |q-v|^2 = |q|^2 + |v|^2 - 2*q.v,
                # computed in full so the scores EQUAL the distances plain
                # IndexFlatL2.search returns (same order and magnitudes as
                # the unfiltered path); lower distance = better.
                q_sq = float(np.square(vec[0]).sum())
                scores = np.square(sub).sum(axis=1) - 2.0 * (sub @ vec[0]) + q_sq
                order = np.argsort(scores)[:top_k]  # lower distance = better
            else:
                scores = sub @ vec[0]
                order = np.argsort(-scores)[:top_k]  # higher IP = better
            ids = [store._meta.chunk_ids[positions[i]] for i in order]
            out_d = np.asarray(scores[order], dtype=np.float32)
            return out_d, ids, [{"chunk_id": cid} for cid in ids]
        oversample = min(max(top_k * 20, top_k), int(store._index.ntotal))
        dists, ids = store.search(vec, oversample)
        allowed = set(allowed_ids)
        kept = [(d, cid) for d, cid in zip(dists, ids, strict=False) if cid in allowed][:top_k]
        out_d = np.asarray([d for d, _ in kept], dtype=np.float32)
        return out_d, [cid for _, cid in kept], [{"chunk_id": cid} for _, cid in kept]

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
