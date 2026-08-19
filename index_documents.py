#!/usr/bin/env python
"""CLI: extract, chunk, embed, and index a PDF/DOCX document.

Usage:
    python index_documents.py --file ./docs/example.pdf --strategy paragraph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from document_indexer.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, STRATEGIES, chunk_text
from document_indexer.config import load_config
from document_indexer.db import ChunkRecord, connect, init_schema, insert_chunks
from document_indexer.embeddings import embed_chunks
from document_indexer.exceptions import DocumentIndexerError
from document_indexer.extraction import extract_text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Path to a .pdf or .docx file")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES, help="Chunking strategy")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {DEFAULT_CHUNK_SIZE}); "
        "unused for the paragraph strategy",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Character overlap between chunks (default: {DEFAULT_OVERLAP}); "
        "fixed_size strategy only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config()

        text = extract_text(args.file)

        chunk_kwargs = {"chunk_size": args.chunk_size}
        if args.strategy == "fixed_size":
            chunk_kwargs["overlap"] = args.overlap
        chunks = chunk_text(text, args.strategy, **chunk_kwargs)
        print(f"Extracted {len(text)} characters -> {len(chunks)} chunks ({args.strategy})")

        vectors = embed_chunks(chunks, config.gemini_api_key, config.embedding_dim)
        filename = Path(args.file).name
        records = [
            ChunkRecord(chunk, vector, filename, args.strategy)
            for chunk, vector in zip(chunks, vectors)
        ]

        with connect(config.postgres_url) as conn:
            init_schema(conn, config.embedding_dim)
            insert_chunks(conn, records)

        print(f"Indexed {len(records)} chunks from '{filename}' into the database.")
        return 0

    except (DocumentIndexerError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
