# __init__.py
# Expose top-level imports for external use

"""
CausalRAG: Causal Graph Enhanced Retrieval-Augmented Generation

This package integrates causal reasoning with retrieval-augmented generation
to improve the quality and accuracy of generated answers by considering
causal relationships between concepts.
"""

__version__ = "0.1.0"
__author__ = "CausalRAG Team"

# Core components
from .pipeline import CausalRAGPipeline
from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever

# Set default logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Convenience function to create and configure a pipeline
def create_pipeline(
    model_name="gpt-4", 
    embedding_model="all-MiniLM-L6-v2",
    graph_path=None, 
    index_path=None,
    config_path=None
):
    """
    Create and configure a CausalRAG pipeline
    
    Args:
        model_name: Name of LLM model to use
        embedding_model: Name of embedding model for vector store
        graph_path: Optional path to pre-built causal graph
        index_path: Optional path to pre-built vector index
        config_path: Optional path to pipeline configuration
        
    Returns:
        Configured CausalRAGPipeline instance
    """
    return CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        config_path=config_path
    )

# Define what's available via import *
__all__ = [
    'CausalRAGPipeline',
    'CausalGraphBuilder',
    'CausalPathRetriever',
    'create_pipeline',
]