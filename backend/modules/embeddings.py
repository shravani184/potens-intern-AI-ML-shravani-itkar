from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np

from backend.core.exceptions import EmbeddingError
from backend.core.logging_config import get_logger
from config import settings

logger = get_logger(__name__)


class EmbeddingModel:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # loaded lazily

    @property
    def model(self):
        if self._model is None:
            try:
                # Imported here so the (heavy) dependency is only required when
                # embeddings are actually used.
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run `pip install -r requirements.txt`."
                ) from exc

            logger.info("Loading embedding model '%s'...", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc
            logger.info("Embedding model loaded (dim=%d).", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return vectors.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(settings.embedding_model)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)