"""Unit tests for document_indexer.embeddings. The Gemini client is mocked
throughout - no real API calls or network access."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from document_indexer.embeddings import embed_chunks, embed_query
from document_indexer.exceptions import EmbeddingGenerationError


def _fake_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


@pytest.fixture
def mock_client(mocker):
    client = mocker.MagicMock()
    mocker.patch("document_indexer.embeddings.genai.Client", return_value=client)
    return client


class TestEmbedChunks:
    def test_returns_one_vector_per_input_text_in_order(self, mock_client) -> None:
        mock_client.models.embed_content.return_value = _fake_response([[0.1, 0.2], [0.3, 0.4]])

        result = embed_chunks(["chunk one", "chunk two"], api_key="key", dimensions=2)

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_uses_retrieval_document_task_type(self, mock_client) -> None:
        mock_client.models.embed_content.return_value = _fake_response([[0.1, 0.2]])

        embed_chunks(["chunk"], api_key="key", dimensions=2)

        kwargs = mock_client.models.embed_content.call_args.kwargs
        assert kwargs["model"] == "gemini-embedding-001"
        assert kwargs["contents"] == ["chunk"]
        assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
        assert kwargs["config"].output_dimensionality == 2

    def test_empty_input_returns_empty_without_calling_api(self, mock_client) -> None:
        result = embed_chunks([], api_key="key", dimensions=768)

        assert result == []
        mock_client.models.embed_content.assert_not_called()

    def test_api_failure_raises_embedding_generation_error(self, mock_client) -> None:
        mock_client.models.embed_content.side_effect = RuntimeError("quota exceeded")

        with pytest.raises(EmbeddingGenerationError):
            embed_chunks(["chunk"], api_key="key", dimensions=2)

    def test_dimension_mismatch_raises_embedding_generation_error(self, mock_client) -> None:
        mock_client.models.embed_content.return_value = _fake_response([[0.1, 0.2, 0.3]])

        with pytest.raises(EmbeddingGenerationError):
            embed_chunks(["chunk"], api_key="key", dimensions=2)


class TestEmbedQuery:
    def test_returns_a_single_vector_not_a_list_of_vectors(self, mock_client) -> None:
        mock_client.models.embed_content.return_value = _fake_response([[0.5, 0.6]])

        result = embed_query("search text", api_key="key", dimensions=2)

        assert result == [0.5, 0.6]

    def test_uses_retrieval_query_task_type(self, mock_client) -> None:
        mock_client.models.embed_content.return_value = _fake_response([[0.5, 0.6]])

        embed_query("search text", api_key="key", dimensions=2)

        kwargs = mock_client.models.embed_content.call_args.kwargs
        assert kwargs["contents"] == ["search text"]
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_api_failure_raises_embedding_generation_error(self, mock_client) -> None:
        mock_client.models.embed_content.side_effect = RuntimeError("network error")

        with pytest.raises(EmbeddingGenerationError):
            embed_query("search text", api_key="key", dimensions=2)
