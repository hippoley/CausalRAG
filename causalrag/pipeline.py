# pipeline.py
# Legacy one-shot retrieval interface. New applications should prefer create_agent().

from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever
from .reranker.causal_path import CausalPathReranker
from .retriever.vector_store import VectorStoreRetriever
from .retriever.hybrid import HybridRetriever
from .generator.prompt_builder import build_prompt
from .generator.llm_interface import LLMInterface


class CausalRAGPipeline:
    """Backward-compatible one-shot causal RAG pipeline."""

    def __init__(
        self,
        model_name="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",
        graph_path=None,
        index_path=None,
        config_path=None,
        provider="openai",
        api_key=None,
    ):
        self.graph_builder = CausalGraphBuilder(graph_path=graph_path)
        self.vector_retriever = VectorStoreRetriever(
            model_name=embedding_model,
            cache_dir=index_path,
        )
        if index_path:
            self.vector_retriever.load_cached(index_path)
        self.graph_retriever = CausalPathRetriever(self.graph_builder)
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.graph_retriever)
        self.reranker = CausalPathReranker(self.graph_retriever)
        self.llm = LLMInterface(model=model_name, provider=provider, api_key=api_key)

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path):
        return None

    def index(self, documents):
        """Build causal graph and vector index from documents."""
        graph_result = self.graph_builder.index_documents(documents)
        vector_result = self.vector_retriever.index_corpus(documents)
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
