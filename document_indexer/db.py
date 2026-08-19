"""PostgreSQL + pgvector storage and semantic search."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from document_indexer.exceptions import DatabaseConnectionError, InvalidEmbeddingError

TABLE_NAME = "document_chunks"
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(table_name: str) -> str:
    if not _SAFE_IDENTIFIER_RE.match(table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    return table_name


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk ready to be stored."""

    chunk_text: str
    embedding: list[float]
    filename: str
    split_strategy: str


@dataclass(frozen=True)
class SearchResult:
    """One semantic search hit, ordered by ascending cosine distance."""

    id: int
    chunk_text: str
    filename: str
    split_strategy: str
    created_at: datetime
    distance: float


@contextlib.contextmanager
def connect(postgres_url: str) -> Iterator["psycopg2.extensions.connection"]:
    """Open a database connection with the pgvector adapter registered.

    Raises:
        DatabaseConnectionError: if the connection cannot be established.
    """
    try:
        conn = psycopg2.connect(postgres_url)
    except psycopg2.OperationalError as exc:
        raise DatabaseConnectionError(f"Could not connect to the database: {exc}") from exc

    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema(conn, dimensions: int, table_name: str = TABLE_NAME) -> None:
    """Create the pgvector extension, the chunks table, and its similarity
    index if they don't already exist. Safe to call on every run."""
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError(f"dimensions must be a positive integer, got: {dimensions!r}")
    table_name = _validate_table_name(table_name)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({dimensions}) NOT NULL,
                filename TEXT NOT NULL,
                split_strategy TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
            ON {table_name} USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def insert_chunks(conn, records: list[ChunkRecord], table_name: str = TABLE_NAME) -> None:
    """Batch-insert chunk records. No-op for an empty list.

    Raises:
        InvalidEmbeddingError: if pgvector rejects a record (wrong embedding
            dimension, NaN/Infinity values, or a NULL embedding).
    """
    if not records:
        return
    table_name = _validate_table_name(table_name)

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO {table_name} (chunk_text, embedding, filename, split_strategy) VALUES %s",
                [(r.chunk_text, r.embedding, r.filename, r.split_strategy) for r in records],
            )
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise InvalidEmbeddingError(f"Could not insert chunk(s): {exc}") from exc


def search(
    conn, query_embedding: list[float], top_k: int = 5, table_name: str = TABLE_NAME
) -> list[SearchResult]:
    """Return the top_k chunks closest to query_embedding by cosine distance.

    Rows whose distance comes out as NaN (a stored zero vector, against which
    cosine distance is mathematically undefined) are excluded rather than
    sorted arbitrarily. This is otherwise unreachable from normal data since
    pgvector itself rejects NaN/Infinity vector values at insert time.

    Raises:
        InvalidEmbeddingError: if query_embedding's dimension doesn't match
            the table's embedding column.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got: {top_k}")
    table_name = _validate_table_name(table_name)

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, chunk_text, filename, split_strategy, created_at, distance
                FROM (
                    SELECT id, chunk_text, filename, split_strategy, created_at,
                           embedding <=> %s::vector AS distance
                    FROM {table_name}
                ) scored
                WHERE distance = distance  -- excludes NaN (NaN is never equal to itself)
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, top_k),
            )
            rows = cur.fetchall()
    except psycopg2.Error as exc:
        conn.rollback()
        raise InvalidEmbeddingError(f"Search query failed: {exc}") from exc

    return [
        SearchResult(
            id=row[0],
            chunk_text=row[1],
            filename=row[2],
            split_strategy=row[3],
            created_at=row[4],
            distance=row[5],
        )
        for row in rows
    ]
