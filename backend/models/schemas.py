"""Pydantic models that define the public API contract.

These schemas are the single source of truth for request/response shapes and
are reused by FastAPI (for validation + OpenAPI docs) and by the Streamlit UI.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

class Citation(BaseModel):
    """A single evidence citation returned alongside an answer."""

    source: str = Field(..., description="Source filename the chunk came from.")
    page: int = Field(..., description="Page number of the supporting chunk.")
    chunk_id: str = Field(..., description="Unique identifier of the chunk.")
    snippet: str = Field(..., description="Supporting text snippet from the chunk.")
    similarity: float = Field(
        ..., description="Cosine similarity of the chunk to the query (0-1)."
    )


class RetrievedChunk(BaseModel):
    """A retrieved chunk exposed to the UI for transparency."""

    chunk_id: str
    source: str
    page: int
    text: str
    similarity: float

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: Optional[int] = Field(
        default=None, gt=0, description="Override the number of chunks retrieved."
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="If set, restrict retrieval to a single source filename.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What is semantic chunking and why is it used?"}
        }
    }


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str
    detected_language: str = Field(..., description="ISO code of the query language.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool = Field(
        ..., description="True when confidence is below the human-review threshold."
    )
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)


class ContradictRequest(BaseModel):
    """Request body for the /contradict endpoint."""

    document_1: str = Field(..., description="Filename of the first document.")
    document_2: str = Field(..., description="Filename of the second document.")
    topic: str = Field(..., min_length=1, description="Topic to compare across docs.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_1": "policy_v1.pdf",
                "document_2": "policy_v2.pdf",
                "topic": "remote work eligibility",
            }
        }
    }


class ContradictResponse(BaseModel):
    """Response body for the /contradict endpoint."""

    conflict: bool
    reasoning: str
    document_1: str
    document_2: str
    topic: str
    citations_document_1: List[Citation] = Field(default_factory=list)
    citations_document_2: List[Citation] = Field(default_factory=list)

class IngestResponse(BaseModel):
    """Response after ingesting one or more documents."""

    ingested_files: List[str]
    total_chunks: int
    message: str


class DocumentInfo(BaseModel):
    """Summary of an indexed document."""

    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """List of documents currently in the vector store."""

    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    llm_provider: str
    embedding_model: str
    collection: str
    indexed_chunks: int