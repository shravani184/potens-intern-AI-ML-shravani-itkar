from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List

from backend.core.logging_config import get_logger
from config import settings

logger = get_logger(__name__)


class Reranker:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
            cand["similarity"] = 1.0 / (1.0 + math.exp(-float(score)))  # sigmoid

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model)
