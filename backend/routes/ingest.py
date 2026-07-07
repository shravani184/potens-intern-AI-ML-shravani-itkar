from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.exceptions import RAGError
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
)
from backend.modules.pipeline import ingest_directory, ingest_file
from backend.modules.vectorstore import get_vector_store
from config import settings

logger = get_logger(__name__)
router = APIRouter(tags=["Documents"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    ingested: List[str] = []
    total_chunks = 0
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {upload.filename}")
        dest = docs_dir / Path(upload.filename).name
        try:
            with dest.open("wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
        finally:
            await upload.close()
        try:
            result = ingest_file(dest, replace=True)
            ingested.append(result.source)
            total_chunks += result.chunks
        except RAGError as exc:
            logger.error("Ingestion failed for %s: %s", dest.name, exc)
            raise HTTPException(status_code=422, detail=f"{dest.name}: {exc}") from exc

    return IngestResponse(
        ingested_files=ingested,
        total_chunks=total_chunks,
        message=f"Ingested {len(ingested)} file(s) into '{settings.collection_name}'.",
    )


@router.post("/ingest/directory", response_model=IngestResponse)
def ingest_dir() -> IngestResponse:
    try:
        results = ingest_directory(settings.documents_dir, replace=True)
    except RAGError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(
        ingested_files=[r.source for r in results],
        total_chunks=sum(r.chunks for r in results),
        message=f"Ingested {len(results)} file(s) from documents directory.",
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    store = get_vector_store()
    sources = store.list_sources()
    docs = [DocumentInfo(source=s, chunk_count=c) for s, c in sorted(sources.items())]
    return DocumentsResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=sum(sources.values()),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
        collection=settings.collection_name,
        indexed_chunks=store.count(),
    )
