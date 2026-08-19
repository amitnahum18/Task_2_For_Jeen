"""Environment-based configuration loading for the document indexer."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from document_indexer.exceptions import ConfigError

load_dotenv()

DEFAULT_EMBEDDING_DIM = 768


@dataclass(frozen=True)
class Config:
    """Resolved application configuration.

    ``__repr__`` is overridden so secrets never end up in logs, tracebacks,
    or an accidental ``print(config)``.
    """

    gemini_api_key: str
    postgres_url: str
    embedding_dim: int

    def __repr__(self) -> str:
        return f"Config(gemini_api_key='***', postgres_url='***', embedding_dim={self.embedding_dim})"


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Raises:
        ConfigError: if a required variable is missing or malformed.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    postgres_url = os.getenv("POSTGRES_URL")
    embedding_dim_raw = os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM))

    missing = [
        name
        for name, value in (("GEMINI_API_KEY", gemini_api_key), ("POSTGRES_URL", postgres_url))
        if not value
    ]
    if missing:
        raise ConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )

    try:
        embedding_dim = int(embedding_dim_raw)
    except ValueError as exc:
        raise ConfigError(f"EMBEDDING_DIM must be an integer, got: {embedding_dim_raw!r}") from exc

    if embedding_dim <= 0:
        raise ConfigError(f"EMBEDDING_DIM must be a positive integer, got: {embedding_dim}")

    return Config(
        gemini_api_key=gemini_api_key,
        postgres_url=postgres_url,
        embedding_dim=embedding_dim,
    )
