from __future__ import annotations

from typing import Dict, List

from backend.core.logging_config import get_logger
from backend.models.schemas import Citation, ContradictResponse
from backend.modules.llm import get_llm_client
from backend.modules.retrieval import retrieve
from config import settings

logger = get_logger(__name__)


def _context_block(chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[{i}] (page: {ch['page']}, chunk_id: {ch['chunk_id']})\n{ch['text']}"
        for i, ch in enumerate(chunks, start=1)
    )


def _snippet(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _citations(chunks: List[Dict], passages: List[int]) -> List[Citation]:
    indices = [p - 1 for p in passages if isinstance(p, int)]
    if not indices:
        indices = list(range(len(chunks)))
    citations: List[Citation] = []
    seen = set()
    for pos in indices:
        if 0 <= pos < len(chunks) and pos not in seen:
            seen.add(pos)
            ch = chunks[pos]
            citations.append(
                Citation(
                    source=ch["source"],
                    page=ch["page"],
                    chunk_id=ch["chunk_id"],
                    snippet=_snippet(ch["text"]),
                    similarity=round(ch["similarity"], 4),
                )
            )
    return citations


def check_contradiction(document_1: str, document_2: str, topic: str) -> ContradictResponse:
    top_k = settings.top_k

    chunks_1 = retrieve(topic, top_k=top_k, source_filter=document_1)
    chunks_2 = retrieve(topic, top_k=top_k, source_filter=document_2)

    if not chunks_1 or not chunks_2:
        missing = document_1 if not chunks_1 else document_2
        logger.info("No relevant evidence in '%s' for topic '%s'.", missing, topic)
        return ContradictResponse(
            conflict=False,
            reasoning=(
                f"Not enough relevant information was found in '{missing}' about "
                f"'{topic}' to assess a contradiction."
            ),
            document_1=document_1,
            document_2=document_2,
            topic=topic,
            citations_document_1=_citations(chunks_1, []) if chunks_1 else [],
            citations_document_2=_citations(chunks_2, []) if chunks_2 else [],
        )

    result = get_llm_client().detect_contradiction(
        topic=topic,
        doc1=document_1,
        doc2=document_2,
        context1=_context_block(chunks_1),
        context2=_context_block(chunks_2),
    )

    conflict = bool(result.get("conflict", False))
    reasoning = str(result.get("reasoning", "")).strip() or "No reasoning provided."
    passages_1 = [int(p) for p in (result.get("document_1_passages") or []) if str(p).isdigit()]
    passages_2 = [int(p) for p in (result.get("document_2_passages") or []) if str(p).isdigit()]

    return ContradictResponse(
        conflict=conflict,
        reasoning=reasoning,
        document_1=document_1,
        document_2=document_2,
        topic=topic,
        citations_document_1=_citations(chunks_1, passages_1),
        citations_document_2=_citations(chunks_2, passages_2),
    )
