# pipeline.py
# Legacy one-shot retrieval interface. New applications should prefer create_agent().

import json
import os
from typing import Any, Optional

from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever
from .embeddings import EmbeddingProvider, create_embedding_provider
from .generator.llm_interface import LLMInterface
from .generator.prompt_builder import build_prompt
from .reranker.causal_path import CausalPathReranker
from .retriever.hybrid import HybridRetriever
from .retriever.vector_store import VectorStoreRetriever


class BranchpointPipeline:
    """Backward-compatible one-shot causal retrieval pipeline."""

    def __init__(
        self,
        model_name: str = "gpt-5.6-terra",
        embedding_model: str = "text-embedding-3-small",
        graph_path: Optional[str] = None,
        index_path: Optional[str] = None,
        config_path: Optional[str] = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        extractor_method: str = "rule",
        embedding_provider_name: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_backend: str = "memory",
    ) -> None:
        self.index_path = index_path
        self.llm = LLMInterface(model=model_name, provider=provider, api_key=api_key)
        llm_extractor = self.llm if extractor_method in ("llm", "hybrid") else None

        provider_name = embedding_provider_name or (
            "local" if provider == "local" else "openai"
        )
        shared_embeddings = embedding_provider or create_embedding_provider(
            provider=provider_name,
            model=embedding_model,
            api_key=embedding_api_key or api_key,
        )

        self.graph_builder = CausalGraphBuilder(
            model_name=embedding_model,
            extractor_method=extractor_method,
            llm_interface=llm_extractor,
            embedding_provider=shared_embeddings,
            embedding_provider_name=provider_name,
            embedding_api_key=embedding_api_key or api_key,
        )
        if graph_path:
            self.graph_builder.load(graph_path)

        self.vector_retriever = VectorStoreRetriever(
            model_name=embedding_model,
            backend=vector_backend,
            cache_dir=index_path,
            embedding_provider=shared_embeddings,
            embedding_provider_name=provider_name,
            api_key=embedding_api_key or api_key,
        )
        if index_path and self.vector_retriever.load_cached(index_path):
            passages_path = os.path.join(index_path, "passages.json")
            if os.path.exists(passages_path):
                with open(passages_path, "r", encoding="utf-8") as handle:
                    self.vector_retriever.passages = json.load(handle)

        self.graph_retriever = CausalPathRetriever(self.graph_builder)
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, self.graph_retriever
        )
        self.reranker = CausalPathReranker(self.graph_retriever)

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        return None

    def index(self, documents):
        """Build causal graph and vector index from documents."""
        documents = list(documents)
        graph_result = self.graph_builder.index_documents(documents)
        vector_result = self.vector_retriever.index_corpus(documents)
        if self.index_path:
            os.makedirs(self.index_path, exist_ok=True)
            with open(
                os.path.join(self.index_path, "passages.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(documents, handle, ensure_ascii=False)
            self.graph_builder.save(os.path.join(self.index_path, "causal_graph.json"))
        return {"graph": graph_result, "vectors": vector_result}

    def run(self, query: str, top_k: int = 5):
        """Retrieve, causally rerank, and generate a structured answer."""
        candidates = self.hybrid_retriever.retrieve(query, top_k=top_k)
        reranked = self.reranker.rerank(query, candidates)
        context = [passage for passage, _score in reranked[:top_k]]
        causal_paths = self.graph_retriever.retrieve_paths(query, max_paths=5)
        causal_nodes = self.graph_retriever.retrieve_path_nodes(query)
        prompt = build_prompt(query, context, causal_path=causal_nodes)
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "context": context,
            "causal_paths": causal_paths,
        }
