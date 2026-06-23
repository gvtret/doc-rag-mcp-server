"""Vector store migration tool.

Migrates vectors from one backend to another, or from a single-collection
FAISS store to a multi-namespace layout.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from doc_rag.raglib.vector_index import VectorIndex
from doc_rag.raglib.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)


def migrate_vectors(
    cfg: dict[str, Any],
    source_backend: str = "faiss",
    target_backend: str = "qdrant",
    collection: str = "default",
) -> dict[str, Any]:
    """Migrate vectors from one backend to another.

    Args:
        cfg: Parsed config dict.
        source_backend: Source backend name (faiss, qdrant, pgvector).
        target_backend: Target backend name.
        collection: Target collection name.

    Returns:
        Migration stats dict.
    """
    stats = {
        "source_backend": source_backend,
        "target_backend": target_backend,
        "vectors_migrated": 0,
        "success": False,
    }

    # Create source store
    source_cfg = dict(cfg)
    source_cfg.setdefault("index", {})["backend"] = source_backend
    source_store = create_vector_store(source_cfg, collection)

    # Create target store
    target_cfg = dict(cfg)
    target_cfg.setdefault("index", {})["backend"] = target_backend
    target_store = create_vector_store(target_cfg, collection)

    # Load source metadata
    source_vi = VectorIndex(source_cfg, collection)
    source_vi._load_existing_ids()

    if not source_vi._chunk_ids:
        stats["message"] = "No chunks found in source"
        return stats

    # Initialize target with same dimension
    dim = source_store.dim if hasattr(source_store, "dim") else 0
    if dim == 0:
        # Try to get dim from source by reading a vector
        try:
            if hasattr(source_store, "_index") and source_store._index is not None:
                dim = source_store._index.d
        except Exception:
            pass

    if dim == 0:
        stats["message"] = "Cannot determine source dimension"
        return stats

    # Initialize target
    embedder_cfg = cfg.get("embeddings", {})
    model_name = embedder_cfg.get("model_name", "")
    normalize = embedder_cfg.get("normalize", True)
    target_store.init(dim=dim, model_name=model_name, normalize=normalize)

    # Load embeddings model
    from doc_rag.raglib.vector_index import _load_embedder

    embedder = _load_embedder(cfg)
    if embedder is None:
        stats["message"] = "Embeddings model not available"
        return stats

    # Load chunks
    root = cfg.get("_root", ".")
    chunks_path = os.path.join(root, cfg["paths"]["chunks_dir"], "chunks.jsonl")
    if not os.path.exists(chunks_path):
        stats["message"] = f"Chunks file not found: {chunks_path}"
        return stats

    import json

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # Filter chunks to only those in source
    chunk_id_set = set(source_vi._chunk_ids)
    chunks_to_migrate = [c for c in chunks if c.get("chunk_id") in chunk_id_set]

    if not chunks_to_migrate:
        stats["message"] = "No matching chunks found"
        return stats

    # Encode all texts
    texts = [str(c.get("text", "")) for c in chunks_to_migrate]
    ids = [str(c.get("chunk_id", "")) for c in chunks_to_migrate]

    batch_size = int(embedder_cfg.get("batch_size", 32))
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    # Add to target
    target_store.add(vecs, ids)
    target_store.save()

    stats["vectors_migrated"] = len(ids)
    stats["success"] = True
    logger.info(
        "migrate: %d vectors migrated from %s to %s (collection=%s)",
        len(ids),
        source_backend,
        target_backend,
        collection,
    )
    return stats


def add_namespace_to_cfg(cfg: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create a config copy scoped to a specific namespace/collection."""
    cfg_copy = dict(cfg)
    cfg_copy["_namespace"] = namespace
    return cfg_copy
