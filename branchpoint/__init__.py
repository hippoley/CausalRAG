"""Branchpoint: causal world models for goal-directed agents.

The default import exposes the lightweight causal-agent runtime. Legacy retrieval,
graph and evaluation surfaces are loaded only when explicitly requested.
"""

import logging

__version__ = "0.3.0"
__author__ = "Branchpoint Team"

from .agent import (
    ActionKind,
    ActionScore,
    AgentState,
    CandidateAction,
    CausalAgentLoop,
    DecisionRecord,
    Observation,
    PendingEffect,
    TemporalEffectContract,
    TimeDriver,
    VirtualTimeDriver,
)
from .agent.runtime import AgentRunResult, CausalAgent, create_agent
from .experiments import (
    DecisionPreferences,
    ExperimentContract,
    ExperimentUpdate,
    ModelMismatchAssessment,
    ModelMismatchPolicy,
    OutcomeLikelihood,
    assess_model_mismatch,
    expected_information_gain,
    expanded_experiment_contract,
    outcome_surprisal,
    posterior_for_outcome,
    predictive_probability,
)
from .observability import (
    CAUSAL_TRACE_SCHEMA_VERSION,
    CausalTelemetry,
    configure_otlp_telemetry,
    replay_trace,
)
from .reasoning.hypothesis import HypothesisProposal, LLMHypothesisUpdater
from .reasoning.llm import LLMCausalReasoner
from .tools import ToolRegistry, ToolSpec
from .world_model import CausalBelief, CausalWorldModel, Evidence, Hypothesis, ModelMismatch, Transition

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    if name in {"CausalEvaluator", "EvaluationResult"}:
        try:
            from .evaluation.evaluator import CausalEvaluator, EvaluationResult
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Evaluation requires optional dependencies. Install them with: pip install 'branchpoint[evaluation]'") from exc
        return {"CausalEvaluator": CausalEvaluator, "EvaluationResult": EvaluationResult}[name]
    if name == "BranchpointPipeline":
        try:
            from .pipeline import BranchpointPipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("BranchpointPipeline requires optional retrieval dependencies. Install them with: pip install 'branchpoint[retrieval]'") from exc
        return BranchpointPipeline
    if name == "CausalGraphBuilder":
        try:
            from .causal_graph.builder import CausalGraphBuilder
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CausalGraphBuilder requires optional retrieval dependencies. Install them with: pip install 'branchpoint[retrieval]'") from exc
        return CausalGraphBuilder
    if name == "CausalPathRetriever":
        try:
            from .causal_graph.retriever import CausalPathRetriever
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("CausalPathRetriever requires optional retrieval dependencies. Install them with: pip install 'branchpoint[retrieval]'") from exc
        return CausalPathRetriever
    raise AttributeError("module 'branchpoint' has no attribute %r" % name)


def create_pipeline(model_name="gpt-5.6-terra", embedding_model="text-embedding-3-small", graph_path=None, index_path=None, config_path=None, provider="openai", api_key=None, extractor_method="rule", embedding_provider_name=None, embedding_api_key=None, embedding_provider=None, vector_backend="memory"):
    """Create the legacy one-shot retrieval pipeline on demand."""
    try:
        from .pipeline import BranchpointPipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("The legacy retrieval pipeline requires optional dependencies. Install them with: pip install 'branchpoint[retrieval]'" ) from exc
    return BranchpointPipeline(
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
    "TemporalEffectContract", "PendingEffect", "TimeDriver", "VirtualTimeDriver",
    "ToolRegistry", "ToolSpec", "CausalBelief", "CausalWorldModel", "Evidence", "Hypothesis", "ModelMismatch", "Transition",
    "DecisionPreferences", "OutcomeLikelihood", "ExperimentContract", "ExperimentUpdate", "expected_information_gain", "posterior_for_outcome",
    "ModelMismatchPolicy", "ModelMismatchAssessment", "assess_model_mismatch", "predictive_probability", "outcome_surprisal", "expanded_experiment_contract",
    "CAUSAL_TRACE_SCHEMA_VERSION", "CausalTelemetry", "configure_otlp_telemetry", "replay_trace",
]
