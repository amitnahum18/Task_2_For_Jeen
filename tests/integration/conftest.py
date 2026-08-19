"""Fixtures for tests that require a real PostgreSQL + pgvector connection.

Run with `pytest -m integration`. The default `pytest` (or explicit
`pytest -m "not integration"`) skips this directory entirely, so a database
is only needed when these tests are run on purpose.
"""

from __future__ import annotations

import uuid

import pytest

from document_indexer.config import load_config
from document_indexer.db import connect, init_schema

TEST_DIMENSIONS = 3  # small and easy to reason about geometrically; pgvector's
# constraint/error behavior does not depend on the actual dimension size.


@pytest.fixture
def db_conn():
    config = load_config()
    with connect(config.postgres_url) as conn:
        yield conn


@pytest.fixture
def test_table(db_conn):
    """A freshly created, uniquely named table, dropped after the test."""
    table_name = f"_test_chunks_{uuid.uuid4().hex[:8]}"
    init_schema(db_conn, dimensions=TEST_DIMENSIONS, table_name=table_name)
    yield table_name
    with db_conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    db_conn.commit()
