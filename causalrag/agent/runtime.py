from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from causalrag.pipeline import CausalRAGPipeline
from causalrag.reasoning.belief import LLMBeliefUpdater
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel

from .loop import CausalAgentLoop
from .state import AgentState


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


def _sync_graph_beliefs(pipeline: CausalRAGPipeline, world_model: CausalWorldModel) -> None:
    """Seed explicit beliefs from the current extracted causal graph."""
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
        return {
            "answer": self.answer,
            "goal": self.state.goal,
            "steps": self.state.step,
            "stop_reason": self.state.stop_reason,
            "decisions": _jsonable(self.state.decisions),
            "observations": _jsonable(self.state.observations),
            "beliefs": _jsonable(self.world_model.snapshot()),
            "transitions": _jsonable(self.world_model.transitions),
        }


class CausalAgent:
    """User-facing causal agent with retrieval available out of the box."""

    def __init__(
        self,
        loop: CausalAgentLoop,
        pipeline: Optional[CausalRAGPipeline] = None,
    ) -> None:
        self.loop = loop
        self.pipeline = pipeline

    def index(self, documents: Iterable[str]) -> "CausalAgent":
        if self.pipeline is None:
            raise RuntimeError("This agent was created without a retrieval pipeline")
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


def create_agent(
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "all-MiniLM-L6-v2",
    graph_path: Optional[str] = None,
    index_path: Optional[str] = None,
    documents: Optional[Iterable[str]] = None,
    tools: Optional[Iterable[ToolSpec]] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    world_model: Optional[CausalWorldModel] = None,
    extractor_method: str = "hybrid",
) -> CausalAgent:
    """Create a ready-to-run causal agent.

    By default, indexing combines rule-based and model-based causal extraction.
    Set ``extractor_method='rule'`` for offline/low-cost indexing.
    """
    pipeline = CausalRAGPipeline(
        model_name=model_name,
        embedding_model=embedding_model,
        graph_path=graph_path,
        index_path=index_path,
        provider=provider,
        api_key=api_key,
        extractor_method=extractor_method,
    )
    if documents:
        pipeline.index(list(documents))

    model_state = world_model or CausalWorldModel()
    _sync_graph_beliefs(pipeline, model_state)

    registry = ToolRegistry(tools)

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
                metadata={"kind": "retrieve", "arguments": {"query": "str", "top_k": "int"}},
            )
        )

    llm = pipeline.llm
    reasoner = LLMCausalReasoner(llm=llm, tools=registry)
    belief_updater = LLMBeliefUpdater(llm=llm)
    loop = CausalAgentLoop(
        reasoner=reasoner,
        tools=registry,
        world_model=model_state,
        belief_updater=belief_updater,
    )
    return CausalAgent(loop=loop, pipeline=pipeline)
