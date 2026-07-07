from __future__ import annotations

from typing import Dict, List, Optional

from backend.core.logging_config import get_logger
from backend.modules.embeddings import get_embedding_model
from backend.modules.vectorstore import get_vector_store
from config import settings

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    source_filter: Optional[str] = None,
    use_reranker: Optional[bool] = None,
) -> List[Dict]:
    
    top_k = top_k or settings.top_k
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker

    embedder = get_embedding_model()
    store = get_vector_store()

    query_embedding = embedder.embed_text(query)

    if use_reranker:
        pool_size = max(settings.rerank_candidate_pool, top_k)
        candidates = store.query(query_embedding, top_k=pool_size, source_filter=source_filter)
        if not candidates:
            return []
        try:
            from backend.modules.reranker import get_reranker

            results = get_reranker().rerank(query, candidates, top_k=top_k)
            logger.info("Reranked %d candidates -> top %d.", len(candidates), len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker unavailable (%s); using vector scores.", exc)
            return candidates[:top_k]

    results = store.query(query_embedding, top_k=top_k, source_filter=source_filter)
    logger.info("Retrieved %d chunks for query.", len(results))
    return results
