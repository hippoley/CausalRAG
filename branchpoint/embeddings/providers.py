from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from openai import OpenAI


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal provider contract shared by vector and graph retrieval."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> Optional[int]:
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 array for the supplied texts."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        ...


def _as_matrix(values: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


class OpenAIEmbeddingProvider:
    """Lightweight hosted embedding provider.

    This is the default for the optional retrieval layer so installing retrieval no
    longer implies installing PyTorch. The client can be injected for tests or
    OpenAI-compatible gateways.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._model_name = model
        self._dimension = dimensions
        self.client = client or OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            width = self._dimension or 0
            return np.empty((0, width), dtype=np.float32)
        kwargs = {"model": self._model_name, "input": list(texts)}
        if self._dimension is not None:
            kwargs["dimensions"] = self._dimension
        response = self.client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        matrix = _as_matrix(item.embedding for item in ordered)
        if self._dimension is None and matrix.size:
            self._dimension = int(matrix.shape[1])
        return matrix

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider loaded only when explicitly requested."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Local embeddings require sentence-transformers/PyTorch. Install "
                "them with: pip install 'branchpoint[local-embeddings]'"
            ) from exc
        self._model_name = model
        self.encoder = SentenceTransformer(model)
        self._dimension = int(self.encoder.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.asarray(
            self.encoder.encode(list(texts), convert_to_numpy=True), dtype=np.float32
        )

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def create_embedding_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
    client: Optional[Any] = None,
) -> EmbeddingProvider:
    name = provider.lower().replace("_", "-")
    if name in {"openai", "hosted"}:
        return OpenAIEmbeddingProvider(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            dimensions=dimensions,
            client=client,
        )
    if name in {"local", "sentence-transformers", "sentence-transformer"}:
        return SentenceTransformerEmbeddingProvider(model or "all-MiniLM-L6-v2")
    raise ValueError(f"Unsupported embedding provider: {provider}")
