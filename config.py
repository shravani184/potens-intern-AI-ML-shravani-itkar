from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Literal["groq", "gemini"] = Field(
        default="groq",
        description="Which LLM backend to use for generation.",
    )
    groq_api_key: str = Field(default="", description="Groq API key.")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    google_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-1.5-flash")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model used for embeddings.",
    )

    chroma_persist_dir: str = Field(default=str(BASE_DIR / "data" / "chroma"))
    collection_name: str = Field(default="rag_semantic_chunks")

    documents_dir: str = Field(default=str(BASE_DIR / "data" / "documents"))

    semantic_similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    max_chunk_chars: int = Field(default=1200, gt=0)
    min_chunk_chars: int = Field(default=120, ge=0)

    top_k: int = Field(default=5, gt=0, description="Chunks retrieved per query.")

    rerank_candidate_pool: int = Field(default=15, gt=0)
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    min_retrieval_similarity: float = Field(default=0.25, ge=0.0, le=1.0)

    human_review_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_base_url: str = Field(default="http://localhost:8000")
    log_level: str = Field(default="INFO")

    insufficient_context_message: str = Field(
        default="The provided documents do not contain enough information to answer this question.",
    )

    def ensure_dirs(self) -> None:
        """Create persistence directories if they do not yet exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    s = Settings()
    s.ensure_dirs()
    return s

settings = get_settings()