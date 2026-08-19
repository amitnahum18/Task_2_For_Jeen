"""Unit tests for document_indexer.db. psycopg2 is fully mocked - no real
database connection. Real-database behavior (constraints, pgvector edge
cases) is covered separately under tests/integration."""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg2
import pytest

from document_indexer.db import ChunkRecord, SearchResult, connect, init_schema, insert_chunks, search
from document_indexer.exceptions import DatabaseConnectionError, InvalidEmbeddingError


@pytest.fixture
def mock_conn(mocker):
    conn = mocker.MagicMock()
    cursor = mocker.MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


class TestConnect:
    def test_wraps_operational_error(self, mocker) -> None:
        mocker.patch("document_indexer.db.psycopg2.connect", side_effect=psycopg2.OperationalError("refused"))

        with pytest.raises(DatabaseConnectionError):
            with connect("postgresql://bad"):
                pass

    def test_closes_connection_on_success(self, mocker) -> None:
        fake_conn = mocker.MagicMock()
        mocker.patch("document_indexer.db.psycopg2.connect", return_value=fake_conn)
        mocker.patch("document_indexer.db.register_vector")

        with connect("postgresql://ok"):
            pass

        fake_conn.close.assert_called_once()

    def test_closes_connection_even_if_body_raises(self, mocker) -> None:
        fake_conn = mocker.MagicMock()
        mocker.patch("document_indexer.db.psycopg2.connect", return_value=fake_conn)
        mocker.patch("document_indexer.db.register_vector")

        with pytest.raises(RuntimeError):
            with connect("postgresql://ok"):
                raise RuntimeError("boom")

        fake_conn.close.assert_called_once()

    def test_registers_vector_adapter(self, mocker) -> None:
        fake_conn = mocker.MagicMock()
        mocker.patch("document_indexer.db.psycopg2.connect", return_value=fake_conn)
        register_vector = mocker.patch("document_indexer.db.register_vector")

        with connect("postgresql://ok"):
            pass

        register_vector.assert_called_once_with(fake_conn)


class TestInitSchema:
    def test_creates_extension_table_and_index(self, mock_conn) -> None:
        conn, cursor = mock_conn

        init_schema(conn, dimensions=768)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        assert any("CREATE EXTENSION" in s for s in statements)
        assert any("CREATE TABLE" in s for s in statements)
        assert any("hnsw" in s for s in statements)
        conn.commit.assert_called_once()

    def test_interpolates_dimensions_into_vector_column(self, mock_conn) -> None:
        conn, cursor = mock_conn

        init_schema(conn, dimensions=1536)

        statements = " ".join(call.args[0] for call in cursor.execute.call_args_list)
        assert "VECTOR(1536)" in statements

    def test_rejects_non_positive_dimensions(self, mock_conn) -> None:
        conn, _ = mock_conn

        with pytest.raises(ValueError):
            init_schema(conn, dimensions=0)

    def test_rejects_non_integer_dimensions(self, mock_conn) -> None:
        conn, _ = mock_conn

        with pytest.raises(ValueError):
            init_schema(conn, dimensions="768")


class TestInsertChunks:
    def test_empty_list_is_a_noop(self, mock_conn) -> None:
        conn, cursor = mock_conn

        insert_chunks(conn, [])

        cursor.execute.assert_not_called()
        conn.commit.assert_not_called()

    def test_batch_inserts_all_records(self, mock_conn, mocker) -> None:
        conn, cursor = mock_conn
        execute_values = mocker.patch("document_indexer.db.execute_values")
        records = [
            ChunkRecord("first chunk", [0.1, 0.2], "doc.pdf", "paragraph"),
            ChunkRecord("second chunk", [0.3, 0.4], "doc.pdf", "paragraph"),
        ]

        insert_chunks(conn, records)

        passed_cursor, sql, values = execute_values.call_args.args
        assert passed_cursor is cursor
        assert "INSERT INTO document_chunks" in sql
        assert values == [
            ("first chunk", [0.1, 0.2], "doc.pdf", "paragraph"),
            ("second chunk", [0.3, 0.4], "doc.pdf", "paragraph"),
        ]
        conn.commit.assert_called_once()

    def test_db_error_is_wrapped_and_rolled_back(self, mock_conn, mocker) -> None:
        conn, cursor = mock_conn
        mocker.patch(
            "document_indexer.db.execute_values",
            side_effect=psycopg2.errors.NotNullViolation("null value in column embedding"),
        )
        records = [ChunkRecord("chunk", [0.1, 0.2], "doc.pdf", "paragraph")]

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(conn, records)

        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()


class TestSearch:
    def test_maps_rows_into_search_results(self, mock_conn) -> None:
        conn, cursor = mock_conn
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        cursor.fetchall.return_value = [
            (1, "chunk text", "doc.pdf", "paragraph", now, 0.05),
        ]

        results = search(conn, [0.1, 0.2], top_k=5)

        assert results == [
            SearchResult(
                id=1,
                chunk_text="chunk text",
                filename="doc.pdf",
                split_strategy="paragraph",
                created_at=now,
                distance=0.05,
            )
        ]

    def test_empty_results_returns_empty_list(self, mock_conn) -> None:
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []

        assert search(conn, [0.1, 0.2], top_k=5) == []

    def test_passes_query_embedding_and_top_k_as_params(self, mock_conn) -> None:
        conn, cursor = mock_conn
        cursor.fetchall.return_value = []

        search(conn, [0.1, 0.2, 0.3], top_k=7)

        _, params = cursor.execute.call_args.args
        assert params == ([0.1, 0.2, 0.3], 7)

    def test_rejects_non_positive_top_k(self, mock_conn) -> None:
        conn, _ = mock_conn

        with pytest.raises(ValueError):
            search(conn, [0.1, 0.2], top_k=0)

    def test_db_error_is_wrapped_and_rolled_back(self, mock_conn) -> None:
        conn, cursor = mock_conn
        cursor.execute.side_effect = psycopg2.errors.DataException("different vector dimensions 2 and 3")

        with pytest.raises(InvalidEmbeddingError):
            search(conn, [0.1, 0.2], top_k=5)

        conn.rollback.assert_called_once()
