"""CausalRAG: causal world models for goal-directed agents.

The legacy RAG pipeline remains available, while the v0.2 core introduces an
explicit causal belief state and an agent loop that can observe, retrieve, ask,
intervene, wait, stop, and learn from resulting transitions.
"""

__version__ = "0.2.0"
__author__ = "CausalRAG Team"

from .pipeline import CausalRAGPipeline
from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever
from .evaluation.evaluator import CausalEvaluator, EvaluationResult
from .agent import (
    ActionKind,
    AgentState,
    CandidateAction,
    CausalAgentLoop,
    DecisionRecord,
    Observation,
)
from .tools import ToolRegistry, ToolSpec
from .world_model import CausalBelief, CausalWorldModel, Evidence, Transition

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


def create_pipeline(
    model_name="gpt-4",
    embedding_model="all-MiniLM-L6-v2",
    graph_path=None,
    index_path=None,
    config_path=None,
):
    """Create the legacy one-shot CausalRAG pipeline."""
    return CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        config_path=config_path,
    )


__all__ = [
    "CausalRAGPipeline",
    "CausalGraphBuilder",
    "CausalPathRetriever",
    "CausalEvaluator",
    "EvaluationResult",
    "create_pipeline",
    "ActionKind",
    "AgentState",
    "CandidateAction",
    "CausalAgentLoop",
    "DecisionRecord",
    "Observation",
    "ToolRegistry",
    "ToolSpec",
    "CausalBelief",
    "CausalWorldModel",
    "Evidence",
    "Transition",
]
