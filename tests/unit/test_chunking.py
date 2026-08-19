"""Unit tests for document_indexer.chunking."""

from __future__ import annotations

import pytest

from document_indexer.chunking import (
    chunk_by_paragraphs,
    chunk_by_sentences,
    chunk_fixed_size,
    chunk_text,
    split_sentences,
)


class TestChunkFixedSize:
    def test_splits_into_expected_number_of_chunks(self) -> None:
        text = "x" * 2500
        chunks = chunk_fixed_size(text, chunk_size=1000, overlap=200)

        # step = 800; starts at 0, 800, 1600 -> 3 chunks covering 2500 chars
        assert len(chunks) == 3

    def test_consecutive_chunks_share_the_overlap(self) -> None:
        text = "".join(f"{i:04d}" for i in range(1000))  # deterministic, indexable content
        chunks = chunk_fixed_size(text, chunk_size=1000, overlap=200)

        for i in range(len(chunks) - 1):
            tail_of_current = chunks[i][-200:]
            head_of_next = chunks[i + 1][:200]
            assert tail_of_current == head_of_next

    def test_text_shorter_than_chunk_size_returns_single_chunk(self) -> None:
        chunks = chunk_fixed_size("short text", chunk_size=1000, overlap=200)

        assert chunks == ["short text"]

    def test_empty_text_returns_no_chunks(self) -> None:
        assert chunk_fixed_size("", chunk_size=1000, overlap=200) == []

    def test_whitespace_only_text_returns_no_chunks(self) -> None:
        assert chunk_fixed_size("   \n\t  ", chunk_size=1000, overlap=200) == []

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", chunk_size=0, overlap=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", chunk_size=100, overlap=-1)

    def test_rejects_overlap_not_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", chunk_size=100, overlap=100)


class TestSplitSentences:
    def test_splits_on_sentence_terminators(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"

        sentences = split_sentences(text)

        assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]

    def test_does_not_split_on_common_abbreviations(self) -> None:
        text = "Contact Dr. Smith for details. He is available on Monday."

        sentences = split_sentences(text)

        assert sentences == [
            "Contact Dr. Smith for details.",
            "He is available on Monday.",
        ]

    def test_empty_text_returns_no_sentences(self) -> None:
        assert split_sentences("") == []

    def test_single_sentence_with_no_terminator(self) -> None:
        assert split_sentences("just one clause with no period") == [
            "just one clause with no period"
        ]


class TestChunkBySentences:
    def test_groups_sentences_up_to_chunk_size(self) -> None:
        text = "One. Two. Three. Four. Five."

        chunks = chunk_by_sentences(text, chunk_size=15)

        assert chunks == ["One. Two.", "Three. Four.", "Five."]
        assert all(len(c) <= 15 for c in chunks)

    def test_single_short_text_returns_one_chunk(self) -> None:
        assert chunk_by_sentences("Just one sentence.", chunk_size=1000) == [
            "Just one sentence."
        ]

    def test_empty_text_returns_no_chunks(self) -> None:
        assert chunk_by_sentences("", chunk_size=1000) == []

    def test_sentence_longer_than_chunk_size_becomes_its_own_chunk(self) -> None:
        long_sentence = "word " * 100 + "end."
        chunks = chunk_by_sentences(long_sentence, chunk_size=10)

        assert len(chunks) == 1
        assert chunks[0] == long_sentence.strip()

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_by_sentences("text.", chunk_size=0)


class TestChunkByParagraphs:
    def test_splits_on_blank_lines(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = chunk_by_paragraphs(text)

        assert chunks == ["Paragraph one.", "Paragraph two.", "Paragraph three."]

    def test_single_paragraph_returns_one_chunk(self) -> None:
        assert chunk_by_paragraphs("Just one paragraph, no breaks.") == [
            "Just one paragraph, no breaks."
        ]

    def test_empty_text_returns_no_chunks(self) -> None:
        assert chunk_by_paragraphs("") == []

    def test_extra_blank_lines_do_not_create_empty_chunks(self) -> None:
        text = "First.\n\n\n\nSecond.\n\n   \n\nThird."

        chunks = chunk_by_paragraphs(text)

        assert chunks == ["First.", "Second.", "Third."]


class TestChunkTextDispatcher:
    def test_dispatches_to_fixed_size(self) -> None:
        assert chunk_text("x" * 50, "fixed_size", chunk_size=20, overlap=5) == chunk_fixed_size(
            "x" * 50, chunk_size=20, overlap=5
        )

    def test_dispatches_to_sentence(self) -> None:
        text = "One. Two. Three."
        assert chunk_text(text, "sentence", chunk_size=50) == chunk_by_sentences(
            text, chunk_size=50
        )

    def test_dispatches_to_paragraph(self) -> None:
        text = "A.\n\nB."
        assert chunk_text(text, "paragraph") == chunk_by_paragraphs(text)

    def test_unknown_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("text", "not_a_real_strategy")
