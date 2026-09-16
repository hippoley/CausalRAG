"""CausalRAG: causal world models for goal-directed agents.

The default import exposes the lightweight causal-agent runtime. Legacy RAG,
graph and evaluation surfaces are loaded only when explicitly requested.
"""

import logging

__version__ = "0.2.0"
__author__ = "CausalRAG Team"

from .agent import (
    ActionKind,
    ActionScore,
    AgentState,
    CandidateAction,
    CausalAgentLoop,
    DecisionRecord,
    Observation,
)
from .agent.runtime import AgentRunResult, CausalAgent, create_agent
from .experiments import (
    DecisionPreferences,
    ExperimentContract,
    ExperimentUpdate,
    OutcomeLikelihood,
    expected_information_gain,
    posterior_for_outcome,
)
from .reasoning.hypothesis import HypothesisProposal, LLMHypothesisUpdater
from .reasoning.llm import LLMCausalReasoner
from .tools import ToolRegistry, ToolSpec
from .world_model import CausalBelief, CausalWorldModel, Evidence, Hypothesis, Transition

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    if name in {"CausalEvaluator", "EvaluationResult"}:
        try:
            from .evaluation.evaluator import CausalEvaluator, EvaluationResult
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Evaluation requires optional dependencies. Install them with: pip install 'causalrag[evaluation]'") from exc
        return {"CausalEvaluator": CausalEvaluator, "EvaluationResult": EvaluationResult}[name]
    if name == "CausalRAGPipeline":
        try:
            from .pipeline import CausalRAGPipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CausalRAGPipeline requires optional RAG dependencies. Install them with: pip install 'causalrag[rag]'") from exc
        return CausalRAGPipeline
    if name == "CausalGraphBuilder":
        try:
            from .causal_graph.builder import CausalGraphBuilder
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CausalGraphBuilder requires optional RAG dependencies. Install them with: pip install 'causalrag[rag]'") from exc
        return CausalGraphBuilder
    if name == "CausalPathRetriever":
        try:
            from .causal_graph.retriever import CausalPathRetriever
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CausalPathRetriever requires optional RAG dependencies. Install them with: pip install 'causalrag[rag]'") from exc
        return CausalPathRetriever
    raise AttributeError("module 'causalrag' has no attribute %r" % name)


def create_pipeline(model_name="gpt-5.6-terra", embedding_model="text-embedding-3-small", graph_path=None, index_path=None, config_path=None, provider="openai", api_key=None, extractor_method="rule", embedding_provider_name=None, embedding_api_key=None, embedding_provider=None, vector_backend="memory"):
    """Create the legacy one-shot RAG pipeline on demand."""
    try:
        from .pipeline import CausalRAGPipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("The legacy RAG pipeline requires optional dependencies. Install them with: pip install 'causalrag[rag]'") from exc
    return CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        config_path=config_path,
        provider=provider,
        api_key=api_key,
        extractor_method=extractor_method,
        embedding_provider_name=embedding_provider_name,
        embedding_api_key=embedding_api_key,
        embedding_provider=embedding_provider,
        vector_backend=vector_backend,
    )


__all__ = [
    "CausalAgent", "AgentRunResult", "create_agent", "LLMCausalReasoner", "LLMHypothesisUpdater", "HypothesisProposal", "create_pipeline",
    "ActionKind", "ActionScore", "AgentState", "CandidateAction", "CausalAgentLoop", "DecisionRecord", "Observation",
    "ToolRegistry", "ToolSpec", "CausalBelief", "CausalWorldModel", "Evidence", "Hypothesis", "Transition",
    "DecisionPreferences", "OutcomeLikelihood", "ExperimentContract", "ExperimentUpdate", "expected_information_gain", "posterior_for_outcome",
]
