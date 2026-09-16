from types import SimpleNamespace

import numpy as np

from causalrag.causal_graph.builder import CausalGraphBuilder
from causalrag.causal_graph.retriever import CausalPathRetriever
from causalrag.embeddings import OpenAIEmbeddingProvider
from causalrag.retriever.vector_store import VectorStoreRetriever


class FakeEmbeddingProvider:
    model_name = "fake-embedding"
    dimension = 3

    def embed(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("co2" in lowered or "carbon" in lowered),
                    float("window" in lowered or "ventilation" in lowered),
                    float("occupant" in lowered or "people" in lowered),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeEmbeddingsEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index, _text in enumerate(kwargs["input"])
            ]
        )


def test_openai_embedding_provider_uses_embeddings_api():
    endpoint = FakeEmbeddingsEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small", client=client
    )

    matrix = provider.embed(["a", "b"])

    assert matrix.shape == (2, 2)
    assert provider.dimension == 2
    assert endpoint.calls[0]["model"] == "text-embedding-3-small"


def test_vector_retriever_accepts_injected_provider_without_torch():
    provider = FakeEmbeddingProvider()
    retriever = VectorStoreRetriever(
        embedding_provider=provider,
        embedding_provider_name="fake",
        backend="memory",
    )
    retriever.index_corpus(
        [
            "Opening the window improves ventilation",
            "More occupants can raise CO2",
        ]
    )

    result = retriever.search("window ventilation", top_k=1)

    assert result == ["Opening the window improves ventilation"]


def test_graph_builder_and_path_retriever_share_provider():
    provider = FakeEmbeddingProvider()
    builder = CausalGraphBuilder(
        embedding_provider=provider,
        embedding_provider_name="fake",
        extractor_method="rule",
    )
    builder.add_triples(
        [
            ("opening window", "ventilation", 0.9),
            ("ventilation", "lower CO2", 0.9),
        ]
    )
    retriever = CausalPathRetriever(builder)

    nodes = retriever.retrieve_nodes("window ventilation", threshold=0.1)
    paths = retriever.retrieve_paths("window ventilation", max_paths=3)

    assert nodes
    assert any("opening window" in path for path in paths)
    assert retriever.embedding_provider is provider
