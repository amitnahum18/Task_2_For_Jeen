"""PostgreSQL + pgvector storage and semantic search."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from document_indexer.exceptions import DatabaseConnectionError

TABLE_NAME = "document_chunks"


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


def init_schema(conn, dimensions: int) -> None:
    """Create the pgvector extension, the chunks table, and its similarity
    index if they don't already exist. Safe to call on every run."""
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError(f"dimensions must be a positive integer, got: {dimensions!r}")

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
            ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def insert_chunks(conn, records: list[ChunkRecord]) -> None:
    """Batch-insert chunk records. No-op for an empty list."""
    if not records:
        return

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {TABLE_NAME} (chunk_text, embedding, filename, split_strategy) VALUES %s",
            [(r.chunk_text, r.embedding, r.filename, r.split_strategy) for r in records],
        )
    conn.commit()


def search(conn, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
    """Return the top_k chunks closest to query_embedding by cosine distance."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got: {top_k}")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, chunk_text, filename, split_strategy, created_at,
                   embedding <=> %s::vector AS distance
            FROM {TABLE_NAME}
            ORDER BY distance
            LIMIT %s
            """,
            (query_embedding, top_k),
        )
        rows = cur.fetchall()

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
