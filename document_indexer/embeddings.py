"""Generate text embeddings via the Gemini API."""

from __future__ import annotations

from google import genai
from google.genai import types

from document_indexer.exceptions import EmbeddingGenerationError

EMBEDDING_MODEL = "gemini-embedding-001"
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
MAX_BATCH_SIZE = 100  # Gemini's embedContent API rejects batches larger than this


def embed_chunks(texts: list[str], api_key: str, dimensions: int) -> list[list[float]]:
    """Embed document chunks for indexing, one vector per input text, in order."""
    return _embed(texts, api_key, dimensions, TASK_TYPE_DOCUMENT)


def embed_query(text: str, api_key: str, dimensions: int) -> list[float]:
    """Embed a single search query."""
    return _embed([text], api_key, dimensions, TASK_TYPE_QUERY)[0]


def _embed(texts: list[str], api_key: str, dimensions: int, task_type: str) -> list[list[float]]:
    if not texts:
        return []

    client = genai.Client(api_key=api_key)
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dimensions)

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[start : start + MAX_BATCH_SIZE]
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch, config=config)
        except Exception as exc:
            # Intentionally broad: this is the external API boundary, and every
            # failure mode (auth, network, quota, malformed input) should surface
            # to callers as the same domain-specific error.
            raise EmbeddingGenerationError(f"Gemini embedding request failed: {exc}") from exc
        embeddings.extend(item.values for item in response.embeddings)

    for embedding in embeddings:
        if len(embedding) != dimensions:
            raise EmbeddingGenerationError(
                f"Expected {dimensions}-dimensional embedding, got {len(embedding)}"
            )
    return embeddings
