from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.core.exceptions import VectorStoreError
from backend.core.logging_config import get_logger
from backend.modules.chunking import Chunk
from config import settings

logger = get_logger(__name__)


class VectorStore:

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:  # pragma: no cover
                raise VectorStoreError("chromadb is not installed.") from exc
            try:
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not open ChromaDB: {exc}") from exc
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        try:
            self.collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=[c.embedding.tolist() for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata() for c in chunks],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to write chunks: {exc}") from exc
        logger.info("Upserted %d chunks into '%s'.", len(chunks), self.collection_name)
        return len(chunks)

    def delete_source(self, source: str) -> None:
        try:
            self.collection.delete(where={"source": source})
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete source '{source}': {exc}") from exc

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
    
        where = {"source": source_filter} if source_filter else None
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Query failed: {exc}") from exc

        return self._format_results(result)

    @staticmethod
    def _format_results(result: Dict) -> List[Dict]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        formatted: List[Dict] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            formatted.append(
                {
                    "chunk_id": meta.get("chunk_id", cid),
                    "source": meta.get("source", "unknown"),
                    "page": int(meta.get("page", 0)),
                    "text": doc,
                    "similarity": similarity,
                }
            )
        return formatted

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:  # noqa: BLE001
            return 0

    def list_sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to list sources: {exc}") from exc
        for meta in data.get("metadatas") or []:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = None


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings.chroma_persist_dir, settings.collection_name)
