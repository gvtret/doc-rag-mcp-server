"""pgvector vector-store backend.

Optional dependency: `psycopg[binary]` + `pgvector`. Install with:
    uv sync --extra pgvector

Connects to an existing PostgreSQL instance with the pgvector extension.
Each collection is a separate table.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_psycopg():  # type: ignore[no-untyped-def]
    try:
        import psycopg

        return psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Install with: uv sync --extra pgvector"
        ) from exc


class PgvectorVectorStore:
    """PostgreSQL + pgvector backed vector store."""

    def __init__(
        self,
        table_name: str,
        dsn: str = "postgresql://localhost:5432/doc_rag",
    ) -> None:
        self._table = table_name
        self._dsn = dsn
        self._conn: Any = None
        self._dim: int = 0

    def _ensure_conn(self) -> Any:
        if self._conn is not None and not self._conn.closed:
            return self._conn
        psycopg = _import_psycopg()
        self._conn = psycopg.connect(self._dsn)
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                raise RuntimeError("pgvector extension not installed in PostgreSQL")
            cur.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{self._table}' AND column_name = 'embedding'"
            )
            if cur.fetchone():
                cur.execute(f"SELECT vector_dims(embedding) FROM {self._table} LIMIT 1")
                row = cur.fetchone()
                if row:
                    self._dim = row[0]

    def init(self, dim: int, model_name: str, normalize: bool = True) -> None:
        conn = self._ensure_conn()
        self._dim = dim
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table}")
            cur.execute(f"""
                CREATE TABLE {self._table} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    embedding vector({dim})
                )
            """)
            cur.execute(
                f"CREATE INDEX ON {self._table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        conn.commit()
        logger.info("pgvector: created table '%s' (dim=%d)", self._table, dim)

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            for vec, cid in zip(vectors, ids, strict=False):
                cur.execute(
                    f"INSERT INTO {self._table} (chunk_id, embedding) VALUES (%s, %s::vector)",
                    (cid, vec.tolist()),
                )
        conn.commit()
        logger.info("pgvector: added %d vectors to '%s'", len(ids), self._table)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, list[str]]:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self._table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (query[0].tolist(), query[0].tolist(), top_k),
            )
            rows = cur.fetchall()
        distances = []
        ids = []
        for cid, score in rows:
            distances.append(float(score))
            ids.append(cid)
        return np.array(distances, dtype=np.float32), ids

    def delete(self, ids: list[str]) -> int:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s)",
                (ids,),
            )
            deleted = cur.rowcount
        conn.commit()
        logger.info("pgvector: deleted %d vectors from '%s'", deleted, self._table)
        return deleted

    def save(self) -> None:
        # PostgreSQL persists automatically; commit any pending work.
        if self._conn and not self._conn.closed:
            self._conn.commit()

    @property
    def size(self) -> int:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

    @property
    def dim(self) -> int:
        return self._dim

    def reset(self) -> None:
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table}")
        conn.commit()
        logger.info("pgvector: dropped table '%s'", self._table)
