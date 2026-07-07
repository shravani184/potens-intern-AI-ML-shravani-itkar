from __future__ import annotations

from typing import Dict, List

from backend.core.logging_config import get_logger
from backend.models.schemas import AskResponse, Citation, RetrievedChunk
from backend.modules import translation
from backend.modules.llm import get_llm_client
from backend.modules.retrieval import retrieve
from config import settings

logger = get_logger(__name__)


def _build_context(chunks: List[Dict]) -> str:
    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (source: {ch['source']}, page: {ch['page']}, "
            f"chunk_id: {ch['chunk_id']})\n{ch['text']}"
        )
    return "\n\n".join(lines)


def _snippet(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _to_citations(chunks: List[Dict], used_indices: List[int]) -> List[Citation]:
    citations: List[Citation] = []
    seen = set()
    for idx in used_indices:
        pos = idx - 1  # LLM passages are 1-indexed
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


def _retrieved_models(chunks: List[Dict]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=ch["chunk_id"],
            source=ch["source"],
            page=ch["page"],
            text=ch["text"],
            similarity=round(ch["similarity"], 4),
        )
        for ch in chunks
    ]


def answer_question(
    question: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> AskResponse:

    detected_lang = translation.detect_language(question)
    english_query = (
        question
        if detected_lang == "en"
        else translation.to_english(question, source_lang=detected_lang).text
    )
    logger.info("Question language=%s | english_query=%r", detected_lang, english_query)

    chunks = retrieve(english_query, top_k=top_k, source_filter=source_filter)
    retrieved_models = _retrieved_models(chunks)

    top_similarity = chunks[0]["similarity"] if chunks else 0.0

    if not chunks or top_similarity < settings.min_retrieval_similarity:
        fallback = settings.insufficient_context_message
        localized = translation.from_english(fallback, detected_lang)
        logger.info(
            "Confidence gate triggered (top_sim=%.3f < %.3f); skipping LLM.",
            top_similarity,
            settings.min_retrieval_similarity,
        )
        return AskResponse(
            answer=localized,
            detected_language=detected_lang,
            confidence=round(top_similarity, 4),
            needs_human_review=True,
            citations=[],
            retrieved_chunks=retrieved_models,
        )

    context = _build_context(chunks)
    result = get_llm_client().generate_answer(english_query, context)

    english_answer = str(result.get("answer", "")).strip()
    insufficient = bool(result.get("insufficient_context", False))
    used_passages = result.get("used_passages") or []
    if not isinstance(used_passages, list):
        used_passages = []

    if insufficient or not english_answer:
        english_answer = settings.insufficient_context_message
        citations: List[Citation] = []
        confidence = min(top_similarity, settings.human_review_threshold)
    else:
        citations = _to_citations(chunks, [int(i) for i in used_passages if str(i).isdigit()])
        if not citations:
            citations = _to_citations(chunks, [1])
        confidence = top_similarity

    localized_answer = translation.from_english(english_answer, detected_lang)

    needs_review = confidence < settings.human_review_threshold

    return AskResponse(
        answer=localized_answer,
        detected_language=detected_lang,
        confidence=round(confidence, 4),
        needs_human_review=needs_review,
        citations=citations,
        retrieved_chunks=retrieved_models,
    )
