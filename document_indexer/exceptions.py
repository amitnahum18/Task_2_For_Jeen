"""Custom exception hierarchy for the document indexing and retrieval pipeline."""


class DocumentIndexerError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(DocumentIndexerError):
    """Raised when required configuration (environment variables) is missing or invalid."""


class UnsupportedFileTypeError(DocumentIndexerError):
    """Raised when the input file extension is not one of the supported types (.pdf, .docx)."""


class NoExtractableTextError(DocumentIndexerError):
    """Raised when a document contains no usable text after extraction and cleaning."""


class EmbeddingGenerationError(DocumentIndexerError):
    """Raised when the Gemini embedding API call fails (network, quota, invalid input, etc.)."""


class DatabaseConnectionError(DocumentIndexerError):
    """Raised when a connection to the PostgreSQL database cannot be established."""


class InvalidEmbeddingError(DocumentIndexerError):
    """Raised when an embedding vector is rejected as malformed - wrong
    dimension, NULL, or non-finite (NaN/Infinity) values - whether caught
    before sending it to the database or by pgvector itself."""
