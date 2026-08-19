#!/usr/bin/env python
"""CLI: semantic search over previously indexed document chunks.

Usage:
    python search.py --query "login issue"
"""

from __future__ import annotations

import argparse
import sys

from document_indexer.config import load_config
from document_indexer.db import SearchResult, connect, search
from document_indexer.embeddings import embed_query
from document_indexer.exceptions import DocumentIndexerError

CHUNK_PREVIEW_LENGTH = 200


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Search text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    return parser.parse_args(argv)


def _format_result(rank: int, result: SearchResult) -> str:
    preview = result.chunk_text
    if len(preview) > CHUNK_PREVIEW_LENGTH:
        preview = preview[:CHUNK_PREVIEW_LENGTH].rstrip() + "..."
    return (
        f"{rank}. distance={result.distance:.4f}  {result.filename} ({result.split_strategy})\n"
        f"   {preview}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config()
        query_embedding = embed_query(args.query, config.gemini_api_key, config.embedding_dim)

        with connect(config.postgres_url) as conn:
            results = search(conn, query_embedding, top_k=args.top_k)

        if not results:
            print("No results found.")
            return 0

        for rank, result in enumerate(results, start=1):
            print(_format_result(rank, result))
        return 0

    except (DocumentIndexerError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
