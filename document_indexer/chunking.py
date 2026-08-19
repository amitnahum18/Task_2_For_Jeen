"""Split extracted document text into chunks using three strategies:
fixed-size with overlap, sentence-based, and paragraph-based.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

STRATEGIES = ("fixed_size", "sentence", "paragraph")

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "eg", "ie", "inc", "ltd", "co", "approx", "st", "u.s", "u.k",
}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def chunk_text(text: str, strategy: str, **kwargs) -> list[str]:
    """Chunk text using the named strategy.

    Args:
        text: the text to split.
        strategy: one of STRATEGIES ("fixed_size", "sentence", "paragraph").
        **kwargs: forwarded to the underlying strategy function
            (e.g. chunk_size, overlap).

    Raises:
        ValueError: if strategy is not recognized.
    """
    if strategy == "fixed_size":
        return chunk_fixed_size(text, **kwargs)
    if strategy == "sentence":
        return chunk_by_sentences(text, **kwargs)
    if strategy == "paragraph":
        return chunk_by_paragraphs(text)
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")


def chunk_fixed_size(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Split text into fixed-size character windows, each overlapping the
    previous one by `overlap` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_by_sentences(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Group sentences into chunks of up to `chunk_size` characters each,
    never splitting in the middle of a sentence."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _pack_units(split_sentences(text), chunk_size)


def chunk_by_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (paragraph breaks). Each paragraph is one chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation and capitalization cues.

    This is a lightweight heuristic, not a trained model: it guards against
    the most common abbreviations (Mr., e.g., etc.) but can still mis-split
    on unusual ones. For chunking purposes an occasional extra split just
    yields a slightly smaller chunk, which is harmless.
    """
    text = text.strip()
    if not text:
        return []

    pieces = _SENTENCE_BOUNDARY_RE.split(text)
    sentences: list[str] = []
    buffer = ""
    for piece in pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        words = buffer.split()
        last_word = re.sub(r"[.!?]+$", "", words[-1]).lower() if words else ""
        if last_word in _ABBREVIATIONS:
            continue  # false split (e.g. "Dr." before a capitalized name) - keep accumulating
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def _pack_units(units: list[str], chunk_size: int) -> list[str]:
    """Greedily pack text units (e.g. sentences) into chunks no larger than
    chunk_size, without splitting a unit across chunks."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        added_len = len(unit) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + added_len > chunk_size:
            chunks.append(" ".join(current))
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks
