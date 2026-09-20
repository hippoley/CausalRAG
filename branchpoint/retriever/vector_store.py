from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from branchpoint.embeddings import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger(__name__)


class VectorStoreRetriever:
    """Vector retrieval without a hard dependency on a local ML runtime."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        backend: str = "memory",
        dimension: Optional[int] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_provider_name: str = "openai",
        api_key: Optional[str] = None,
    ) -> None:
        self.embedding_provider = embedding_provider or create_embedding_provider(
            provider=embedding_provider_name,
            model=model_name,
            api_key=api_key,
            dimensions=dimension,
        )
        self.model_name = self.embedding_provider.model_name
        self.embedding_provider_name = embedding_provider_name
        self.dimension = dimension or self.embedding_provider.dimension
        self.backend = backend.lower()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.passages: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors = np.empty((0, self.dimension or 0), dtype=np.float32)
        self.index = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if self.backend == "faiss" and self.dimension:
            self._ensure_faiss(self.dimension)
        elif self.backend not in {"memory", "faiss"}:
            logger.warning("Unsupported backend %s; falling back to memory", self.backend)
            self.backend = "memory"

    def _ensure_faiss(self, dimension: int) -> None:
        try:
            import faiss
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "FAISS backend requested but faiss is not installed. Install "
                "with: pip install 'branchpoint[faiss]'"
            ) from exc
        if self.index is None:
            self.index = faiss.IndexFlatIP(int(dimension))

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def index_corpus(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        store_original: bool = True,
    ) -> Dict[str, Any]:
        if not texts:
            return {"indexed": 0, "dimension": self.dimension}

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")
        metadata = metadata or [
            {"id": str(i), "position": i} for i in range(len(texts))
        ]
        ids = ids or [str(i) for i in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batches.append(self.embedding_provider.embed(texts[start : start + self.batch_size]))
        embeddings = np.vstack(batches).astype(np.float32)
        self.dimension = int(embeddings.shape[1])

        if store_original:
            self.passages = list(texts)
            self.metadata = list(metadata)

        self._add_to_backend(embeddings)
        if self.cache_dir:
            self._cache_vectors(embeddings, ids, metadata)

        return {
            "indexed": len(texts),
            "dimension": self.dimension,
            "provider": self.embedding_provider_name,
            "model": self.model_name,
            "backend": self.backend,
        }

    def _add_to_backend(self, embeddings: np.ndarray) -> None:
        normalized = self._normalize_rows(embeddings)
        if self.backend == "faiss":
            self._ensure_faiss(normalized.shape[1])
            self.index.add(normalized)
        else:
            self.vectors = normalized

    def _cache_vectors(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(os.path.join(self.cache_dir, "vectors.npy"), embeddings)
        with open(os.path.join(self.cache_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "ids": ids,
                    "metadata": metadata,
                    "model": self.model_name,
                    "provider": self.embedding_provider_name,
                    "dimension": self.dimension,
                },
                handle,
                ensure_ascii=False,
            )

    def load_cached(self, cache_dir: Optional[str] = None) -> bool:
        directory = cache_dir or self.cache_dir
        if not directory:
            return False
        vectors_path = os.path.join(directory, "vectors.npy")
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(vectors_path) or not os.path.exists(metadata_path):
            return False

        try:
            embeddings = np.load(vectors_path).astype(np.float32)
            with open(metadata_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            cached_dimension = int(data.get("dimension") or embeddings.shape[1])
            if self.dimension is not None and cached_dimension != self.dimension:
                logger.warning(
                    "Cached dimension %s differs from provider dimension %s",
                    cached_dimension,
                    self.dimension,
                )
            self.dimension = cached_dimension
            self.metadata = data.get("metadata", [])
            self._add_to_backend(embeddings)
            return True
        except Exception as exc:
            logger.error("Error loading cached vectors: %s", exc)
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        include_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        if not self.passages:
            return []
        query_vector = self._normalize_rows(self.embedding_provider.embed([query]))
        if self.backend == "faiss":
            return self._search_faiss(query_vector, top_k, threshold, include_scores)
        return self._search_memory(query_vector[0], top_k, threshold, include_scores)

    def _search_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: Optional[float],
        include_scores: bool,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        output = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.passages):
                continue
            score = float(score)
            if threshold is not None and score < threshold:
                continue
            item = self.passages[index]
            output.append((item, score) if include_scores else item)
        return output

    def _search_memory(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: Optional[float],
        include_scores: bool,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        if self.vectors.size == 0:
            return []
        similarities = self.vectors @ query_vector
        ranked = np.argsort(similarities)[::-1][:top_k]
        output = []
        for index in ranked:
            score = float(similarities[index])
            if threshold is not None and score < threshold:
                continue
            item = self.passages[int(index)]
            output.append((item, score) if include_scores else item)
        return output

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        results = self.search(query, top_k, threshold, include_scores=True)
        output = []
        for rank, (passage, score) in enumerate(results, start=1):
            try:
                index = self.passages.index(passage)
            except ValueError:
                index = -1
            output.append(
                {
                    "passage": passage,
                    "score": score,
                    "metadata": self.metadata[index] if 0 <= index < len(self.metadata) else {},
                    "rank": rank,
                }
            )
        return output
