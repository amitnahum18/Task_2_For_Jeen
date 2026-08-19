"""Unit tests for document_indexer.extraction.

PDF/DOCX fixtures are generated on the fly (not loaded from docs/) so each
test controls exactly the input it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as DocxDocument
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet

from document_indexer.exceptions import NoExtractableTextError, UnsupportedFileTypeError
from document_indexer.extraction import _normalize_whitespace, extract_text

PARAGRAPHS = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn "
    "every single morning without fail, rain or shine.",
    "Second paragraph covers billing and subscription details, including "
    "renewal dates, proration rules, and refund windows for customers.",
    "Third paragraph discusses technical troubleshooting steps for sync "
    "failures, cache clearing, and how to report persistent errors.",
]


def _make_pdf_with_paragraphs(path: Path, paragraphs: list[str]) -> None:
    """One paragraph per page - see extraction._extract_pdf_text for why PDF
    paragraph-chunking is page-grained rather than true paragraph-grained."""
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []
    for i, para in enumerate(paragraphs):
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(para, styles["BodyText"]))
    doc.build(story)


def _make_blank_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.showPage()
    c.save()


def _make_docx_with_paragraphs(path: Path, paragraphs: list[str]) -> None:
    document = DocxDocument()
    for para in paragraphs:
        document.add_paragraph(para)
    document.save(str(path))


class TestExtractTextPdf:
    def test_extracts_all_paragraph_text(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.pdf"
        _make_pdf_with_paragraphs(pdf_path, PARAGRAPHS)

        text = extract_text(pdf_path)
        flattened = " ".join(text.split())  # pypdf hard-wraps long lines with '\n'

        for para in PARAGRAPHS:
            assert para.split(".")[0] in flattened

    def test_page_breaks_become_paragraph_breaks(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.pdf"
        _make_pdf_with_paragraphs(pdf_path, PARAGRAPHS)

        text = extract_text(pdf_path)
        blocks = [b for b in text.split("\n\n") if b.strip()]

        assert len(blocks) == len(PARAGRAPHS)

    def test_blank_pdf_raises_no_extractable_text(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "blank.pdf"
        _make_blank_pdf(pdf_path)

        with pytest.raises(NoExtractableTextError):
            extract_text(pdf_path)

    def test_uppercase_extension_is_accepted(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.PDF"
        _make_pdf_with_paragraphs(pdf_path, PARAGRAPHS[:1])

        text = extract_text(pdf_path)

        assert text.strip()


class TestExtractTextDocx:
    def test_extracts_all_paragraph_text(self, tmp_path: Path) -> None:
        docx_path = tmp_path / "sample.docx"
        _make_docx_with_paragraphs(docx_path, PARAGRAPHS)

        text = extract_text(docx_path)

        for para in PARAGRAPHS:
            assert para in text

    def test_preserves_paragraph_boundaries(self, tmp_path: Path) -> None:
        docx_path = tmp_path / "sample.docx"
        _make_docx_with_paragraphs(docx_path, PARAGRAPHS)

        text = extract_text(docx_path)
        blocks = [b for b in text.split("\n\n") if b.strip()]

        assert blocks == PARAGRAPHS

    def test_empty_docx_raises_no_extractable_text(self, tmp_path: Path) -> None:
        docx_path = tmp_path / "empty.docx"
        _make_docx_with_paragraphs(docx_path, [])

        with pytest.raises(NoExtractableTextError):
            extract_text(docx_path)

    def test_whitespace_only_docx_raises_no_extractable_text(self, tmp_path: Path) -> None:
        docx_path = tmp_path / "whitespace.docx"
        _make_docx_with_paragraphs(docx_path, ["   ", "\t"])

        with pytest.raises(NoExtractableTextError):
            extract_text(docx_path)


class TestExtractTextErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "does_not_exist.pdf")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("plain text file", encoding="utf-8")

        with pytest.raises(UnsupportedFileTypeError):
            extract_text(txt_path)


class TestNormalizeWhitespace:
    def test_collapses_repeated_spaces_and_tabs(self) -> None:
        assert _normalize_whitespace("a   b\t\tc") == "a b c"

    def test_collapses_three_or_more_newlines_to_one_blank_line(self) -> None:
        assert _normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert _normalize_whitespace("  \n hello \n  ") == "hello"

    def test_normalizes_windows_line_endings(self) -> None:
        assert _normalize_whitespace("a\r\nb\r\nc") == "a\nb\nc"
