from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.core.exceptions import IngestionError
from backend.core.logging_config import get_logger
from backend.modules.chunking import Chunk, semantic_chunk
from backend.modules.embeddings import get_embedding_model
from backend.modules.ingestion import extract_sentences, list_pdf_files
from backend.modules.vectorstore import get_vector_store

logger = get_logger(__name__)


@dataclass
class IngestResult:

    source: str
    chunks: int


def ingest_file(path: str | Path, replace: bool = True) -> IngestResult:
    
    path = Path(path)
    embedder = get_embedding_model()
    store = get_vector_store()

    sentences = extract_sentences(path)
    if not sentences:
        raise IngestionError(f"No sentences extracted from '{path.name}'.")

    chunks: List[Chunk] = semantic_chunk(sentences, embedder=embedder)
    if not chunks:
        raise IngestionError(f"No chunks produced for '{path.name}'.")

    if replace:
        store.delete_source(path.name)
    store.add_chunks(chunks)

    logger.info("Ingested '%s' -> %d chunks.", path.name, len(chunks))
    return IngestResult(source=path.name, chunks=len(chunks))


def ingest_directory(directory: str | Path, replace: bool = True) -> List[IngestResult]:
    files = list_pdf_files(directory)
    if not files:
        raise IngestionError(f"No PDF files found in '{directory}'.")

    results: List[IngestResult] = []
    for file in files:
        try:
            results.append(ingest_file(file, replace=replace))
        except IngestionError as exc:
            logger.error("Skipping '%s': %s", file.name, exc)
    if not results:
        raise IngestionError("No documents were successfully ingested.")
    return results
