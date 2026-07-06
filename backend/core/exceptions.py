from __future__ import annotations


class RAGError(Exception):
    """Base class for all application errors."""


class IngestionError(RAGError):
    """Raised when a document cannot be loaded or ingested."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGError):
    """Raised for vector-store read/write failures."""


class LLMError(RAGError):
    """Raised when the LLM provider fails or is misconfigured."""


class TranslationError(RAGError):
    """Raised when language detection or translation fails."""