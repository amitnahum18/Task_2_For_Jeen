"""Integration tests against a real PostgreSQL + pgvector database.

This is the rigor suite for invalid vector data: what pgvector actually does
(not what we assume it does) with NULL, wrong-dimension, NaN, and Infinity
embeddings, and how search() behaves when the table already contains a
problematic row. Mocks cannot stand in for this - the guarantees being
tested are pgvector's own type/constraint enforcement.

Run with: pytest -m integration
"""

from __future__ import annotations

import math

import pytest

from document_indexer.db import ChunkRecord, insert_chunks, search
from document_indexer.exceptions import InvalidEmbeddingError

pytestmark = pytest.mark.integration


class TestInvalidVectorValues:
    def test_null_embedding_is_rejected(self, db_conn, test_table) -> None:
        record = ChunkRecord("some text", None, "corrupt.pdf", "paragraph")

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(db_conn, [record], table_name=test_table)

        assert search(db_conn, [1.0, 0.0, 0.0], table_name=test_table) == []

    def test_wrong_dimension_embedding_is_rejected(self, db_conn, test_table) -> None:
        record = ChunkRecord("some text", [1.0, 2.0, 3.0, 4.0, 5.0], "corrupt.pdf", "paragraph")

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(db_conn, [record], table_name=test_table)

        assert search(db_conn, [1.0, 0.0, 0.0], table_name=test_table) == []

    def test_nan_embedding_is_rejected(self, db_conn, test_table) -> None:
        record = ChunkRecord("some text", [1.0, math.nan, 3.0], "corrupt.pdf", "paragraph")

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(db_conn, [record], table_name=test_table)

    def test_infinite_embedding_is_rejected(self, db_conn, test_table) -> None:
        record = ChunkRecord("some text", [1.0, math.inf, 3.0], "corrupt.pdf", "paragraph")

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(db_conn, [record], table_name=test_table)

    def test_rejected_insert_does_not_affect_other_records_in_the_batch(
        self, db_conn, test_table
    ) -> None:
        """execute_values sends one batch statement, so one bad record fails
        the whole batch rather than silently partially inserting - verify
        that failure mode explicitly rather than assume it."""
        records = [
            ChunkRecord("good one", [1.0, 0.0, 0.0], "mixed.pdf", "paragraph"),
            ChunkRecord("bad one", [1.0, math.nan, 0.0], "mixed.pdf", "paragraph"),
        ]

        with pytest.raises(InvalidEmbeddingError):
            insert_chunks(db_conn, records, table_name=test_table)

        assert search(db_conn, [1.0, 0.0, 0.0], table_name=test_table) == []

    def test_search_query_with_wrong_dimension_is_rejected(self, db_conn, test_table) -> None:
        insert_chunks(
            db_conn,
            [ChunkRecord("text", [1.0, 0.0, 0.0], "doc.pdf", "paragraph")],
            table_name=test_table,
        )

        with pytest.raises(InvalidEmbeddingError):
            search(db_conn, [1.0, 0.0], table_name=test_table)  # 2 dims, table is 3

    def test_a_stored_zero_vector_does_not_break_search_or_appear_in_results(
        self, db_conn, test_table
    ) -> None:
        """A zero vector is valid and storable, but cosine distance against it
        is mathematically undefined (NaN) - the one realistic "corrupted-but-
        present" row, since pgvector rejects NULL/NaN/Infinity at insert time
        and can never store them in the first place."""
        insert_chunks(
            db_conn,
            [
                ChunkRecord("normal row", [1.0, 0.0, 0.0], "mixed.pdf", "paragraph"),
                ChunkRecord("zero row", [0.0, 0.0, 0.0], "mixed.pdf", "paragraph"),
            ],
            table_name=test_table,
        )

        results = search(db_conn, [1.0, 0.0, 0.0], top_k=5, table_name=test_table)

        assert [r.chunk_text for r in results] == ["normal row"]

    def test_search_on_empty_table_returns_empty_list(self, db_conn, test_table) -> None:
        assert search(db_conn, [1.0, 0.0, 0.0], table_name=test_table) == []


class TestSearchRanking:
    def test_results_are_ordered_by_ascending_cosine_distance(self, db_conn, test_table) -> None:
        insert_chunks(
            db_conn,
            [
                ChunkRecord("opposite direction", [-1.0, 0.0, 0.0], "doc.pdf", "paragraph"),
                ChunkRecord("same direction", [1.0, 0.0, 0.0], "doc.pdf", "paragraph"),
                ChunkRecord("perpendicular", [0.0, 1.0, 0.0], "doc.pdf", "paragraph"),
            ],
            table_name=test_table,
        )

        results = search(db_conn, [1.0, 0.0, 0.0], top_k=10, table_name=test_table)

        assert [r.chunk_text for r in results] == [
            "same direction",
            "perpendicular",
            "opposite direction",
        ]
        assert results[0].distance == pytest.approx(0.0, abs=1e-6)
        assert results[1].distance == pytest.approx(1.0, abs=1e-6)
        assert results[2].distance == pytest.approx(2.0, abs=1e-6)

    def test_top_k_limits_result_count(self, db_conn, test_table) -> None:
        insert_chunks(
            db_conn,
            [ChunkRecord(f"chunk {i}", [float(i), 0.0, 0.0], "doc.pdf", "paragraph") for i in range(1, 6)],
            table_name=test_table,
        )

        results = search(db_conn, [1.0, 0.0, 0.0], top_k=2, table_name=test_table)

        assert len(results) == 2

    def test_round_trip_preserves_metadata(self, db_conn, test_table) -> None:
        insert_chunks(
            db_conn,
            [ChunkRecord("exact text", [1.0, 0.0, 0.0], "report.docx", "sentence")],
            table_name=test_table,
        )

        [result] = search(db_conn, [1.0, 0.0, 0.0], table_name=test_table)

        assert result.chunk_text == "exact text"
        assert result.filename == "report.docx"
        assert result.split_strategy == "sentence"
        assert result.created_at is not None
