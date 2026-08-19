"""Unit tests for index_documents.py's CLI wiring: argument handling and
error-path plumbing. Every document_indexer call is mocked - pipeline
correctness itself is covered by each module's own tests."""

from __future__ import annotations

import pytest

import index_documents
from document_indexer.config import Config
from document_indexer.exceptions import (
    EmbeddingGenerationError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)


def _fake_config() -> Config:
    return Config(gemini_api_key="key", postgres_url="postgresql://x", embedding_dim=3)


@pytest.fixture
def mock_pipeline(mocker):
    mocker.patch("index_documents.load_config", return_value=_fake_config())
    mocker.patch("index_documents.extract_text", return_value="some extracted text")
    mocker.patch("index_documents.chunk_text", return_value=["chunk one", "chunk two"])
    mocker.patch(
        "index_documents.embed_chunks", return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    mock_conn = mocker.MagicMock()
    mock_connect = mocker.patch("index_documents.connect")
    mock_connect.return_value.__enter__.return_value = mock_conn

    return {
        "init_schema": mocker.patch("index_documents.init_schema"),
        "insert_chunks": mocker.patch("index_documents.insert_chunks"),
        "conn": mock_conn,
    }


class TestMainSuccess:
    def test_indexes_successfully_and_returns_zero(self, mock_pipeline, capsys) -> None:
        exit_code = index_documents.main(["--file", "doc.pdf", "--strategy", "paragraph"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "2 chunks" in out
        assert "Indexed 2 chunks" in out

    def test_passes_records_with_basename_filename_only(self, mock_pipeline) -> None:
        index_documents.main(["--file", "/some/dir/doc.pdf", "--strategy", "paragraph"])

        records = mock_pipeline["insert_chunks"].call_args.args[1]
        assert all(r.filename == "doc.pdf" for r in records)

    def test_fixed_size_strategy_forwards_overlap(self, mock_pipeline, mocker) -> None:
        chunk_text = mocker.patch("index_documents.chunk_text", return_value=["a"])

        index_documents.main(
            ["--file", "doc.pdf", "--strategy", "fixed_size", "--chunk-size", "50", "--overlap", "10"]
        )

        assert chunk_text.call_args.kwargs == {"chunk_size": 50, "overlap": 10}

    def test_paragraph_strategy_does_not_forward_overlap(self, mock_pipeline, mocker) -> None:
        chunk_text = mocker.patch("index_documents.chunk_text", return_value=["a"])

        index_documents.main(["--file", "doc.pdf", "--strategy", "paragraph"])

        assert "overlap" not in chunk_text.call_args.kwargs

    def test_initializes_schema_before_inserting(self, mock_pipeline) -> None:
        index_documents.main(["--file", "doc.pdf", "--strategy", "paragraph"])

        mock_pipeline["init_schema"].assert_called_once_with(mock_pipeline["conn"], 3)


class TestMainErrors:
    def test_missing_file_returns_1_and_prints_to_stderr(self, mocker, capsys) -> None:
        mocker.patch("index_documents.load_config", return_value=_fake_config())
        mocker.patch("index_documents.extract_text", side_effect=FileNotFoundError("no such file"))

        exit_code = index_documents.main(["--file", "missing.pdf", "--strategy", "paragraph"])

        assert exit_code == 1
        assert "Error" in capsys.readouterr().err

    def test_unsupported_file_type_returns_1(self, mocker) -> None:
        mocker.patch("index_documents.load_config", return_value=_fake_config())
        mocker.patch(
            "index_documents.extract_text", side_effect=UnsupportedFileTypeError("bad type")
        )

        assert index_documents.main(["--file", "doc.txt", "--strategy", "paragraph"]) == 1

    def test_no_extractable_text_returns_1(self, mocker) -> None:
        mocker.patch("index_documents.load_config", return_value=_fake_config())
        mocker.patch("index_documents.extract_text", side_effect=NoExtractableTextError("empty"))

        assert index_documents.main(["--file", "doc.pdf", "--strategy", "paragraph"]) == 1

    def test_embedding_failure_returns_1(self, mock_pipeline, mocker) -> None:
        mocker.patch("index_documents.embed_chunks", side_effect=EmbeddingGenerationError("quota"))

        assert index_documents.main(["--file", "doc.pdf", "--strategy", "paragraph"]) == 1

    def test_invalid_strategy_choice_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit):
            index_documents.main(["--file", "doc.pdf", "--strategy", "not_a_real_strategy"])
