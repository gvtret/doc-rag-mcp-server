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
_PAYLOAD_DOC_ID = "doc_id"
_PAYLOAD_SECTION = "section_path"
_PAYLOAD_IS_TABLE = "is_table"
_PAYLOAD_TEXT = "text"


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

    def add_with_metadata(
        self,
        vectors: np.ndarray,
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors with rich metadata payload for hybrid search."""
        client = self._ensure_client()
        QdrantClient, _, _, _, _, _, PointStruct, _ = _import_qdrant()  # noqa: F841
        points = []
        for i, (vec, cid) in enumerate(zip(vectors, ids, strict=False)):
            point_id = abs(hash(cid)) % (2**63)
            payload: dict[str, Any] = {_PAYLOAD_KEY: cid}
            if metadata and i < len(metadata):
                md = metadata[i]
                if md.get("doc_id"):
                    payload[_PAYLOAD_DOC_ID] = md["doc_id"]
                if md.get("section_path"):
                    payload[_PAYLOAD_SECTION] = md["section_path"]
                if md.get("is_table"):
                    payload[_PAYLOAD_IS_TABLE] = True
                if md.get("text"):
                    payload[_PAYLOAD_TEXT] = md["text"][:500]
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
        client.upsert(collection_name=self._collection, points=points)
        logger.info("qdrant: added %d vectors with metadata to '%s'", len(points), self._collection)

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

    def search_with_filter(
        self,
        query: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        """Hybrid search: vector similarity + metadata filter.

        Args:
            query: Query vector (1, dim).
            top_k: Number of results.
            filters: Metadata filters. Supports:
                - doc_id: str — filter by document ID prefix
                - section_path: str — filter by section path prefix
                - is_table: bool — filter to only tables

        Returns:
            (distances, chunk_ids, payloads) — payloads contain full metadata.
        """
        client = self._ensure_client()
        QdrantClient, _, FieldCondition, Filter, MatchValue, _, _, _ = _import_qdrant()  # noqa: F841

        query_filter = None
        if filters:
            conditions = []
            if "doc_id" in filters:
                conditions.append(
                    FieldCondition(
                        key=_PAYLOAD_DOC_ID,
                        match=MatchValue(value=filters["doc_id"]),
                    )
                )
            if "section_path" in filters:
                conditions.append(
                    FieldCondition(
                        key=_PAYLOAD_SECTION,
                        match=MatchValue(value=filters["section_path"]),
                    )
                )
            if "is_table" in filters and filters["is_table"]:
                conditions.append(
                    FieldCondition(
                        key=_PAYLOAD_IS_TABLE,
                        match=MatchValue(value=True),
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        results = client.query_points(
            collection_name=self._collection,
            query=query[0].tolist(),
            limit=top_k,
            query_filter=query_filter,
        )
        distances = []
        ids = []
        payloads = []
        for r in results.points:
            distances.append(r.score)
            ids.append(r.payload.get(_PAYLOAD_KEY, ""))
            payloads.append(dict(r.payload))
        return np.array(distances, dtype=np.float32), ids, payloads

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
