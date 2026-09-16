"""CausalRAG: causal world models for goal-directed agents.

Use ``create_agent`` for the v0.2 goal-directed runtime. The legacy one-shot
``CausalRAGPipeline`` remains available for compatibility.
"""

import logging

__version__ = "0.2.0"
__author__ = "CausalRAG Team"

from .pipeline import CausalRAGPipeline
from .causal_graph.builder import CausalGraphBuilder
from .causal_graph.retriever import CausalPathRetriever
from .agent import (
    ActionKind,
    AgentState,
    CandidateAction,
    CausalAgentLoop,
    DecisionRecord,
    Observation,
)
from .agent.runtime import AgentRunResult, CausalAgent, create_agent
from .reasoning.llm import LLMCausalReasoner
from .tools import ToolRegistry, ToolSpec
from .world_model import CausalBelief, CausalWorldModel, Evidence, Transition

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    """Lazy-load optional legacy evaluation classes.

    This keeps ``import causalrag`` focused on the runtime path instead of
    importing pandas/ragas for users who do not need evaluation.
    """
    if name in {"CausalEvaluator", "EvaluationResult"}:
        from .evaluation.evaluator import CausalEvaluator, EvaluationResult

        return {
            "CausalEvaluator": CausalEvaluator,
            "EvaluationResult": EvaluationResult,
        }[name]
    raise AttributeError("module 'causalrag' has no attribute %r" % name)


def create_pipeline(
    model_name="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",
    graph_path=None,
    index_path=None,
    config_path=None,
    provider="openai",
    api_key=None,
    extractor_method="rule",
):
    """Create the legacy one-shot CausalRAG pipeline."""
    return CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        config_path=config_path,
        provider=provider,
        api_key=api_key,
        extractor_method=extractor_method,
    )


__all__ = [
    "CausalAgent",
    "AgentRunResult",
    "create_agent",
    "LLMCausalReasoner",
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
