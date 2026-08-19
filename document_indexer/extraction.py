"""Extract clean text from PDF and DOCX documents."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from document_indexer.exceptions import NoExtractableTextError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# pypdf logs parse warnings (e.g. "invalid pdf header") straight to stderr on
# a corrupted file; we already turn that failure into a clean
# NoExtractableTextError, so the raw warning would just be confusing noise.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_text(file_path: str | Path) -> str:
    """Extract clean text from a PDF or DOCX file.

    Paragraph breaks (blank lines) are preserved where the source format makes
    them available, so downstream paragraph-based chunking has something real
    to split on. Whitespace is otherwise normalized.

    Raises:
        FileNotFoundError: if the file does not exist.
        UnsupportedFileTypeError: if the extension is not .pdf or .docx.
        NoExtractableTextError: if the file has no usable text, or can't be
            parsed at all (e.g. a scanned PDF with no text layer, an empty
            file, or one that's corrupted / not actually a PDF or DOCX
            despite its extension).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        raw_text = _extract_pdf_text(path) if extension == ".pdf" else _extract_docx_text(path)
    except Exception as exc:
        # Intentionally broad: pypdf/python-docx each raise several different
        # exception types (and don't share one common base) for a corrupted,
        # truncated, or otherwise unparseable file - all of them mean the
        # same thing to a caller, so all of them become the same error here.
        raise NoExtractableTextError(f"Could not read '{path}': {exc}") from exc

    clean_text = _normalize_whitespace(raw_text)

    if not clean_text.strip():
        raise NoExtractableTextError(f"No extractable text found in: {path}")

    return clean_text


def _extract_pdf_text(path: Path) -> str:
    """Join each page's text, treating page boundaries as paragraph breaks.

    PDFs have no reliable, generator-independent signal for where a paragraph
    ends versus where a line merely wrapped: font metrics and internal text
    layout vary too much between PDF producers to guess this safely from
    position or spacing alone. Page boundaries are the one paragraph-like
    break PDFs expose reliably, so PDF paragraph-based chunking ends up
    page-grained; DOCX gets true paragraph-grained chunking via python-docx's
    real paragraph objects.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _extract_docx_text(path: Path) -> str:
    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n\n".join(paragraphs)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and blank lines while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()
