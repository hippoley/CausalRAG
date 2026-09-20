"""
Document retrieval components for fetching relevant passages.

This package provides retrievers based on different strategies including
vector embeddings, BM25, and hybrid approaches combining semantic and causal signals.
"""

from .vector_store import VectorStoreRetriever
from .hybrid import HybridRetriever
from .bm25_retriever import BM25Retriever

__all__ = [
    'VectorStoreRetriever',
    'HybridRetriever',
    'BM25Retriever'
] 