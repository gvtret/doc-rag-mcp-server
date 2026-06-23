"""Qdrant vector-store backend.

Optional dependency: `qdrant-client`. Install with:
    uv sync --extra qdrant

Qdrant runs as a separate service (Docker or native). The backend
connects via HTTP and manages vectors + payload in a collection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PAYLOAD_KEY = "chunk_id"


def _import_qdrant():  # type: ignore[no-untyped-def]
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            FieldCondition,
            Filter,
            MatchValue,
            PointIdsList,
            PointStruct,
            VectorParams,
        )

        return (
            QdrantClient,
            Distance,
            FieldCondition,
            Filter,
            MatchValue,
            PointIdsList,
            PointStruct,
            VectorParams,
        )
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install with: uv sync --extra qdrant"
        ) from exc


class QdrantVectorStore:
    """Qdrant-backed vector store implementing the VectorStore protocol."""

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        self._collection = collection_name
        self._url = url
        self._api_key = api_key
        self._client: Any = None
        self._dim: int = 0

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        QdrantClient = _import_qdrant()[0]
        kwargs: dict[str, Any] = {"url": self._url}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = QdrantClient(**kwargs)
        return self._client

    def _ensure_collection(self) -> None:
        client = self._ensure_client()
        QdrantClient, Distance, _, _, _, _, _, VectorParams = _import_qdrant()  # noqa: F841
        try:
            collections = client.get_collections().collections
            names = [c.name for c in collections]
            if self._collection in names:
                info = client.get_collection(self._collection)
                self._dim = info.config.params.vectors.size
                return
        except Exception:
            pass
        if self._dim == 0:
            raise RuntimeError(
                f"Qdrant collection '{self._collection}' not found. "
                "Create it first or set index.qdrant.collection in config."
            )

    def init(self, dim: int, model_name: str, normalize: bool = True) -> None:
        client = self._ensure_client()
        QdrantClient, Distance, _, _, _, _, _, VectorParams = _import_qdrant()  # noqa: F841
        self._dim = dim
        distance = Distance.COSINE if normalize else Distance.EUCLID
        try:
            client.get_collection(self._collection)
            client.delete_collection(self._collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        logger.info("qdrant: created collection '%s' (dim=%d)", self._collection, dim)

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        client = self._ensure_client()
        QdrantClient, _, _, _, _, _, PointStruct, _ = _import_qdrant()  # noqa: F841
        points = []
        for _i, (vec, cid) in enumerate(zip(vectors, ids, strict=False)):
            point_id = abs(hash(cid)) % (2**63)
            points.append(
                PointStruct(id=point_id, vector=vec.tolist(), payload={_PAYLOAD_KEY: cid})
            )
        client.upsert(collection_name=self._collection, points=points)
        logger.info("qdrant: added %d vectors to '%s'", len(points), self._collection)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, list[str]]:
        client = self._ensure_client()
        QdrantClient, _, _, _, _, _, _, _ = _import_qdrant()  # noqa: F841
        results = client.query_points(
            collection_name=self._collection,
            query=query[0].tolist(),
            limit=top_k,
        )
        distances = []
        ids = []
        for r in results.points:
            distances.append(r.score)
            ids.append(r.payload.get(_PAYLOAD_KEY, ""))
        return np.array(distances, dtype=np.float32), ids

    def delete(self, ids: list[str]) -> int:
        client = self._ensure_client()
        QdrantClient, _, _, _, _, _, _, _ = _import_qdrant()  # noqa: F841
        point_ids = [abs(hash(cid)) % (2**63) for cid in ids]
        client.delete(
            collection_name=self._collection,
            points_selector=point_ids,
        )
        logger.info("qdrant: deleted %d vectors from '%s'", len(ids), self._collection)
        return len(ids)

    def save(self) -> None:
        # Qdrant persists automatically; no-op.
        pass

    @property
    def size(self) -> int:
        client = self._ensure_client()
        try:
            info = client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    @property
    def dim(self) -> int:
        return self._dim

    def reset(self) -> None:
        client = self._ensure_client()
        try:
            client.delete_collection(self._collection)
            logger.info("qdrant: deleted collection '%s'", self._collection)
        except Exception:
            pass
