from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from causalrag.experiments import DecisionPreferences, ModelMismatchPolicy
from causalrag.generator.llm_interface import LLMInterface
from causalrag.reasoning.belief import LLMBeliefUpdater
from causalrag.reasoning.hypothesis import LLMHypothesisUpdater
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel

from .capabilities import RuntimeCapabilities
from .loop import CausalAgentLoop
from .state import AgentState
from .temporal import TimeDriver


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _sync_graph_beliefs(pipeline: Any, world_model: CausalWorldModel) -> None:
    graph = pipeline.graph_builder.get_graph()
    node_text = pipeline.graph_builder.node_text
    for cause_id, effect_id, data in graph.edges(data=True):
        cause = node_text.get(cause_id, str(cause_id))
        effect = node_text.get(effect_id, str(effect_id))
        try:
            probability = float(data.get("weight", 0.5))
        except (TypeError, ValueError):
            probability = 0.5
        world_model.upsert_belief(cause, effect, probability=probability)


@dataclass
class AgentRunResult:
    answer: str
    state: AgentState
    world_model: CausalWorldModel

    def to_dict(self) -> Dict[str, Any]:
        snapshot = self.world_model.snapshot()
        return {
            "answer": self.answer,
            "goal": self.state.goal,
            "steps": len(self.state.decisions),
            "executed_actions": self.state.step,
            "stop_reason": self.state.stop_reason,
            "runtime_capabilities": _jsonable(self.state.scratch.get("runtime_capabilities", {})),
            "decisions": _jsonable(self.state.decisions),
            "observations": _jsonable(self.state.observations),
            "beliefs": _jsonable(snapshot),
            "hypotheses": _jsonable(snapshot.get("hypotheses", [])),
            "open_world": _jsonable(snapshot.get("open_world", {})),
            "transitions": _jsonable(self.world_model.transitions),
        }


class CausalAgent:
    """User-facing causal agent with optional retrieval capabilities."""

    def __init__(self, loop: CausalAgentLoop, pipeline: Optional[Any] = None) -> None:
        self.loop = loop
        self.pipeline = pipeline

    def index(self, documents: Iterable[str]) -> "CausalAgent":
        if not self.loop.capabilities.retrieval:
            raise RuntimeError("Retrieval is disabled by the active RuntimeCapabilities ablation.")
        if self.pipeline is None:
            raise RuntimeError(
                "Indexing is not enabled for this agent. Install the RAG extra "
                "(`pip install 'causalrag[rag]'`) and create the agent with "
                "enable_retrieval=True or documents/index_path."
            )
        self.pipeline.index(list(documents))
        _sync_graph_beliefs(self.pipeline, self.loop.world_model)
        return self

    def run(self, goal: str, max_steps: int = 8) -> AgentRunResult:
        state = self.loop.run(goal=goal, max_steps=max_steps)
        answer = str(state.scratch.get("answer") or "")
        if not answer and state.observations:
            answer = str(state.observations[-1].result)
        return AgentRunResult(answer=answer, state=state, world_model=self.loop.world_model)

    @property
    def world_model(self) -> CausalWorldModel:
        return self.loop.world_model

    @property
    def tools(self) -> ToolRegistry:
        return self.loop.tools

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self.loop.capabilities


def _load_rag_pipeline():
    try:
        from causalrag.pipeline import CausalRAGPipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "CausalRAG retrieval requires the optional RAG dependencies. "
            "Install them with: pip install 'causalrag[rag]'"
        ) from exc
    return CausalRAGPipeline


def create_agent(
    model_name: str = "gpt-5.6-terra",
    embedding_model: str = "text-embedding-3-small",
    graph_path: Optional[str] = None,
    index_path: Optional[str] = None,
    documents: Optional[Iterable[str]] = None,
    tools: Optional[Iterable[ToolSpec]] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    world_model: Optional[CausalWorldModel] = None,
    extractor_method: str = "hybrid",
    enable_retrieval: Optional[bool] = None,
    llm: Optional[Any] = None,
    reasoner: Optional[Any] = None,
    belief_updater: Optional[Any] = None,
    hypothesis_updater: Optional[Any] = None,
    embedding_provider_name: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_provider: Optional[Any] = None,
    vector_backend: str = "memory",
    decision_preferences: Optional[DecisionPreferences] = None,
    time_driver: Optional[TimeDriver] = None,
    mismatch_policy: Optional[ModelMismatchPolicy] = None,
    capabilities: Optional[RuntimeCapabilities] = None,
) -> CausalAgent:
    """Create a ready-to-run causal agent.

    ``capabilities`` is the execution-level ablation surface. Disabling a
    capability changes runtime behavior rather than merely adding an experiment
    label. This is the supported way to run EIG/EVSI/temporal/open-world and
    generic-tool-loop ablations under the same outer agent interface.

    ``decision_preferences`` is deployment-owned consequence utility. It can
    override intervention utilities without changing the reasoner or capability
    implementation. Capability cost/risk/reversibility remain on ToolSpec.

    ``time_driver`` lets a simulator or deployment scheduler share the same
    causal clock used by runtime WAIT semantics. The default is deterministic
    virtual time and never blocks wall-clock execution.

    ``mismatch_policy`` controls runtime-owned open-world escalation. The model
    may propose new explanations only after observed outcomes are sufficiently
    improbable under the current modeled hypothesis set.

    The core runtime remains retrieval-free. When retrieval is enabled, hosted
    OpenAI embeddings are the default for OpenAI-backed agents and local
    sentence-transformers are opt-in through ``embedding_provider_name='local'``.
    Custom reasoners, belief updaters, and hypothesis updaters remain injectable.
    """
    runtime_capabilities = capabilities or RuntimeCapabilities.full()
    registry = ToolRegistry(tools, decision_preferences=decision_preferences)
    model_state = world_model or CausalWorldModel()
    pipeline = None

    retrieval_requested = (
        bool(documents or graph_path or index_path)
        if enable_retrieval is None
        else bool(enable_retrieval)
    )
    wants_retrieval = runtime_capabilities.retrieval and retrieval_requested

    if wants_retrieval:
        CausalRAGPipeline = _load_rag_pipeline()
        pipeline = CausalRAGPipeline(
            model_name=model_name,
            embedding_model=embedding_model,
            graph_path=graph_path,
            index_path=index_path,
            provider=provider,
            api_key=api_key,
            extractor_method=extractor_method,
            embedding_provider_name=embedding_provider_name,
            embedding_api_key=embedding_api_key,
            embedding_provider=embedding_provider,
            vector_backend=vector_backend,
        )
        if documents:
            pipeline.index(list(documents))
        _sync_graph_beliefs(pipeline, model_state)

        def retrieve_evidence(query: str, top_k: int = 5) -> Dict[str, Any]:
            candidates = pipeline.hybrid_retriever.retrieve(query, top_k=top_k)
            reranked = pipeline.reranker.rerank(query, candidates)
            paths = pipeline.graph_retriever.retrieve_paths(query, max_paths=5)
            return {
                "query": query,
                "evidence": reranked[:top_k],
                "causal_paths": paths,
            }

        if "retrieve_evidence" not in registry.specs():
            registry.register(
                ToolSpec(
                    name="retrieve_evidence",
                    description="Retrieve semantically and causally relevant evidence from the indexed corpus.",
                    handler=retrieve_evidence,
                    cost=0.05,
                    risk=0.0,
                    reversible=True,
                    metadata={
                        "kind": "retrieve",
                        "arguments": {"query": "str", "top_k": "int"},
                    },
                )
            )
        if llm is None:
            llm = pipeline.llm

    if reasoner is None:
        if llm is None:
            llm = LLMInterface(model=model_name, provider=provider, api_key=api_key)
        reasoner = LLMCausalReasoner(llm=llm, tools=registry)

    if belief_updater is None and llm is not None:
        belief_updater = LLMBeliefUpdater(llm=llm)
    if hypothesis_updater is None and llm is not None:
        hypothesis_updater = LLMHypothesisUpdater(llm=llm)

    loop = CausalAgentLoop(
        reasoner=reasoner,
        tools=registry,
        world_model=model_state,
        belief_updater=belief_updater,
        hypothesis_updater=hypothesis_updater,
        time_driver=time_driver,
        mismatch_policy=mismatch_policy,
        capabilities=runtime_capabilities,
    )
    return CausalAgent(loop=loop, pipeline=pipeline)
