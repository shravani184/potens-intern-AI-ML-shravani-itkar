from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.logging_config import configure_logging, get_logger
from backend.routes import ask, contradict, ingest
from config import settings

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Semantic RAG API",
    version="1.0.0",
    description=(
        "Retrieval-Augmented Generation over PDF documents using semantic "
        "chunking, ChromaDB, Sentence-Transformers embeddings and an LLM. "
        "Answers are grounded strictly in retrieved context with multilingual "
        "support and contradiction detection."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router)
app.include_router(contradict.router)
app.include_router(ingest.router)


@app.get("/", tags=["Meta"])
def root() -> dict:
    """Root endpoint with a short service summary."""
    return {
        "service": "Semantic RAG API",
        "version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "docs": "/docs",
        "endpoints": ["/ask", "/contradict", "/ingest", "/documents", "/health"],
    }


@app.on_event("startup")
def _startup() -> None:
    logger.info(
        "Semantic RAG API starting (provider=%s, collection=%s).",
        settings.llm_provider,
        settings.collection_name,
    )
