"""Unit tests for search.py's CLI wiring. document_indexer calls are mocked -
see tests/integration for real search behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import search as search_cli
from document_indexer.config import Config
from document_indexer.db import SearchResult
from document_indexer.exceptions import DatabaseConnectionError


def _fake_config() -> Config:
    return Config(gemini_api_key="key", postgres_url="postgresql://x", embedding_dim=3)


@pytest.fixture
def mock_pipeline(mocker):
    mocker.patch("search.load_config", return_value=_fake_config())
    mocker.patch("search.embed_query", return_value=[0.1, 0.2, 0.3])

    mock_conn = mocker.MagicMock()
    mock_connect = mocker.patch("search.connect")
    mock_connect.return_value.__enter__.return_value = mock_conn

    return {"search": mocker.patch("search.search"), "conn": mock_conn}


class TestMainWithResults:
    def test_prints_formatted_results_and_returns_zero(self, mock_pipeline, capsys) -> None:
        mock_pipeline["search"].return_value = [
            SearchResult(1, "matching chunk text", "doc.pdf", "paragraph", datetime.now(timezone.utc), 0.1234),
        ]

        exit_code = search_cli.main(["--query", "login issue"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "0.1234" in out
        assert "doc.pdf" in out
        assert "matching chunk text" in out

    def test_truncates_long_chunk_text(self, mock_pipeline, capsys) -> None:
        long_text = "x" * 500
        mock_pipeline["search"].return_value = [
            SearchResult(1, long_text, "doc.pdf", "paragraph", datetime.now(timezone.utc), 0.1),
        ]

        search_cli.main(["--query", "anything"])

        out = capsys.readouterr().out
        assert "..." in out
        assert long_text not in out

    def test_passes_top_k_through(self, mock_pipeline) -> None:
        mock_pipeline["search"].return_value = []

        search_cli.main(["--query", "anything", "--top-k", "3"])

        assert mock_pipeline["search"].call_args.kwargs["top_k"] == 3


class TestMainEmptyResults:
    def test_prints_no_results_message_and_returns_zero(self, mock_pipeline, capsys) -> None:
        mock_pipeline["search"].return_value = []

        exit_code = search_cli.main(["--query", "nothing matches this"])

        assert exit_code == 0
        assert "No results found." in capsys.readouterr().out


class TestMainErrors:
    def test_database_connection_failure_returns_1(self, mocker, capsys) -> None:
        mocker.patch("search.load_config", return_value=_fake_config())
        mocker.patch("search.embed_query", return_value=[0.1, 0.2, 0.3])
        mocker.patch("search.connect", side_effect=DatabaseConnectionError("refused"))

        exit_code = search_cli.main(["--query", "anything"])

        assert exit_code == 1
        assert "Error" in capsys.readouterr().err
