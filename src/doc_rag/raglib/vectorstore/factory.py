"""Vector-store factory — creates the right backend from config."""

from __future__ import annotations

import logging
from typing import Any

from doc_rag.raglib.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def create_vector_store(
    cfg: dict[str, Any],
    collection: str = "default",
    **kwargs: Any,
) -> VectorStore:
    """Create a VectorStore backend based on config.

    Args:
        cfg: Parsed config dict (must contain `index.backend`).
        collection: Collection/namespace name.
        **kwargs: Extra args forwarded to the backend constructor.

    Returns:
        A VectorStore implementation.
    """
    index_cfg = cfg.get("index") or {}
    backend = str(index_cfg.get("backend", "faiss")).lower()

    if backend == "faiss":
        from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

        root = cfg.get("_root", ".")
        index_dir = cfg.get("paths", {}).get("index_dir", "build/index")
        import os

        base = (
            os.path.join(root, index_dir, collection)
            if collection != "default"
            else os.path.join(root, index_dir)
        )
        return FaissVectorStore(
            index_path=os.path.join(base, "faiss.index"),
            meta_path=os.path.join(base, "index_meta.json"),
            metric=str(index_cfg.get("metric", "ip")),
            **kwargs,
        )

    if backend == "qdrant":
        from doc_rag.raglib.vectorstore.qdrant_store import QdrantVectorStore

        qdrant_cfg = index_cfg.get("qdrant") or {}
        return QdrantVectorStore(
            collection_name=qdrant_cfg.get("collection", collection),
            url=qdrant_cfg.get("url", "http://localhost:6333"),
            api_key=qdrant_cfg.get("api_key"),
            **kwargs,
        )

    if backend == "pgvector":
        from doc_rag.raglib.vectorstore.pgvector_store import PgvectorVectorStore

        pg_cfg = index_cfg.get("pgvector") or {}
        return PgvectorVectorStore(
            table_name=pg_cfg.get("table", f"vectors_{collection}"),
            dsn=pg_cfg.get("dsn", "postgresql://localhost:5432/doc_rag"),
            **kwargs,
        )

    raise ValueError(f"Unknown vector store backend: {backend!r}. Valid: faiss, qdrant, pgvector")
