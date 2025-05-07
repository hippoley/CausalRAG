# pipeline.py
# Top-level orchestration of CausalRAG pipeline

from causal_graph.builder import CausalGraphBuilder
from causal_graph.retriever import CausalPathRetriever
from reranker.causal_path import CausalPathReranker
from retriever.vector_store import VectorStoreRetriever
from retriever.hybrid import HybridRetriever
from generator.prompt_builder import build_prompt
from generator.llm_interface import LLMInterface

class CausalRAGPipeline:
    def __init__(self):
        # Core components
        self.graph_builder = CausalGraphBuilder()
        self.vector_retriever = VectorStoreRetriever()
        self.graph_retriever = CausalPathRetriever(self.graph_builder)
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.graph_retriever)
        self.reranker = CausalPathReranker(self.graph_retriever)
        self.llm = LLMInterface()

    def index(self, documents):
        """Build graph + vector index from documents"""
        self.graph_builder.index_documents(documents)
        self.vector_retriever.index_corpus(documents)

    def run(self, query: str, top_k: int = 5) -> str:
        """Query → Retrieval → Rerank → Prompt → Generate"""
        # Step 1: Hybrid retrieval
        candidates = self.hybrid_retriever.retrieve(query, top_k=top_k)

        # Step 2: Rerank via causal path
        reranked = self.reranker.rerank(query, candidates)

        # Step 3: Build prompt with causal context
        causal_nodes = self.graph_retriever.retrieve_path_nodes(query)
        prompt = build_prompt(query, reranked[:top_k], causal_path=causal_nodes)

        # Step 4: Generate answer
        return self.llm.generate(prompt)