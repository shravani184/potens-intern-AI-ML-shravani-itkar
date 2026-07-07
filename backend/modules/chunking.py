from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from backend.core.logging_config import get_logger
from backend.modules.embeddings import EmbeddingModel, get_embedding_model
from backend.modules.ingestion import Sentence
from config import settings

logger = get_logger(__name__)


@dataclass
class Chunk:

    chunk_id: str
    source: str
    page: int  # primary page (where the chunk starts)
    page_start: int
    page_end: int
    text: str
    embedding: np.ndarray = field(repr=False)

    def metadata(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "page": int(self.page),
            "page_start": int(self.page_start),
            "page_end": int(self.page_end),
            "chunk_id": self.chunk_id,
        }


def _make_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()[:10]
    stem = source.rsplit(".", 1)[0]
    return f"{stem}::c{index:04d}::{digest}"


def _finalise_chunk(
    sentences: List[Sentence],
    embedder: EmbeddingModel,
    source: str,
    index: int,
) -> Chunk:

    text = " ".join(s.text for s in sentences).strip()
    pages = [s.page for s in sentences]

    embedding = embedder.embed_text(text)
    return Chunk(
        chunk_id=_make_chunk_id(source, index, text),
        source=source,
        page=min(pages),
        page_start=min(pages),
        page_end=max(pages),
        text=text,
        embedding=embedding,
    )


def semantic_chunk(
    sentences: List[Sentence],
    embedder: EmbeddingModel | None = None,
    similarity_threshold: float | None = None,
    max_chunk_chars: int | None = None,
    min_chunk_chars: int | None = None,
) -> List[Chunk]:
    
    if not sentences:
        return []

    embedder = embedder or get_embedding_model()
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.semantic_similarity_threshold
    )
    max_chars = max_chunk_chars if max_chunk_chars is not None else settings.max_chunk_chars
    min_chars = min_chunk_chars if min_chunk_chars is not None else settings.min_chunk_chars

    source = sentences[0].source
    sent_embeddings = embedder.embed_texts([s.text for s in sentences])

    groups: List[List[Sentence]] = []
    current: List[Sentence] = [sentences[0]]
    centroid = sent_embeddings[0].copy()
    current_len = len(sentences[0].text)

    for i in range(1, len(sentences)):
        emb = sent_embeddings[i]
        similarity = float(np.dot(centroid / (np.linalg.norm(centroid) or 1.0), emb))
        sent_len = len(sentences[i].text)

        topic_change = similarity < threshold
        would_overflow = current_len + sent_len > max_chars

        if topic_change or would_overflow:
            groups.append(current)
            current = [sentences[i]]
            centroid = emb.copy()
            current_len = sent_len
        else:
            current.append(sentences[i])
            n = len(current)
            centroid = centroid * (n - 1) / n + emb / n
            current_len += sent_len + 1

    groups.append(current)

    merged: List[List[Sentence]] = []
    for group in groups:
        group_len = sum(len(s.text) for s in group)
        if merged and group_len < min_chars:
            merged[-1].extend(group)
        else:
            merged.append(group)

    chunks = [
        _finalise_chunk(group, embedder, source, idx)
        for idx, group in enumerate(merged)
    ]
    logger.info(
        "Semantic chunking of '%s': %d sentences -> %d chunks (threshold=%.2f).",
        source,
        len(sentences),
        len(chunks),
        threshold,
    )
    return chunks
