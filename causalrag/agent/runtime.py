from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from causalrag.experiments import DecisionPreferences, ModelMismatchPolicy
from causalrag.generator.llm_interface import LLMInterface
from causalrag.observability import CausalTelemetry
from causalrag.reasoning.belief import LLMBeliefUpdater
from causalrag.reasoning.hypothesis import LLMHypothesisUpdater
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel

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


def _selected_score(decision) -> Optional[Any]:
    for score in decision.action_scores:
        if score.action_name == decision.selected.name and score.action_kind == decision.selected.kind:
            return score
    return None


def _emit_causal_run_events(
    telemetry: CausalTelemetry,
    state: AgentState,
    world_model: CausalWorldModel,
    *,
    transition_start: int,
) -> None:
    """Project runtime state into stable point-in-time causal telemetry events."""
    for decision in state.decisions:
        score = _selected_score(decision)
        attributes: Dict[str, Any] = {
            "causalrag.step": int(decision.step),
            "causalrag.action.name": decision.selected.name,
            "causalrag.action.kind": decision.selected.kind.value,
            "causalrag.candidate.count": len(decision.candidates),
            "causalrag.hypothesis.count": len(decision.beliefs_before.get("hypotheses", [])),
            "causalrag.open_world.model_mismatch": bool(
                decision.beliefs_before.get("open_world", {}).get("model_mismatch", False)
            ),
            "causalrag.action.tests_hypotheses": list(decision.selected.tests_hypotheses),
        }
        if decision.uncertainty:
            attributes["causalrag.uncertainty_present"] = True
            if telemetry.capture_content:
                attributes["causalrag.uncertainty"] = decision.uncertainty
        if score is not None:
            attributes.update(
                {
                    "causalrag.decision.total_utility": score.total_utility,
                    "causalrag.decision.goal_gain": score.goal_gain,
                    "causalrag.decision.information_gain": score.information_gain,
                    "causalrag.decision.information_source": score.information_source,
                    "causalrag.decision.cost": score.cost,
                    "causalrag.decision.risk": score.risk,
                    "causalrag.decision.irreversibility": score.irreversibility,
                }
            )
            if score.decision_value is not None:
                attributes["causalrag.decision.value"] = score.decision_value
            if score.expected_value_of_sample_information is not None:
                attributes["causalrag.decision.evsi"] = score.expected_value_of_sample_information
            if score.net_value_of_sampling is not None:
                attributes["causalrag.decision.net_value_of_sampling"] = score.net_value_of_sampling
        telemetry.event("causalrag.decision", attributes)

    for offset, observation in enumerate(state.observations):
        attributes = {
            "causalrag.observation.index": offset,
            "causalrag.action.name": observation.action_name,
            "causalrag.observation.has_temporal_effects": bool(observation.metadata.get("temporal_effects")),
            "causalrag.observation.has_model_mismatch": bool(observation.metadata.get("model_mismatch")),
        }
        if telemetry.capture_content:
            attributes["causalrag.observation.result"] = observation.result
        telemetry.event("causalrag.observation", attributes)

    transitions = world_model.transitions[transition_start:]
    for offset, transition in enumerate(transitions):
        effects = transition.expected_effects or {}
        step = effects.get("step", offset)
        posterior = effects.get("posterior")
        if posterior:
            telemetry.event(
                "causalrag.posterior.updated",
                {
                    "causalrag.transition.index": offset,
                    "causalrag.step": step,
                    "causalrag.experiment.id": effects.get("experiment_id"),
                    "causalrag.experiment.outcome": effects.get("observed_outcome"),
                    "causalrag.predictive_probability": effects.get("predictive_probability"),
                    "causalrag.surprisal": effects.get("surprisal"),
                    "causalrag.posterior": posterior,
                },
            )

        mismatch = effects.get("model_mismatch")
        if mismatch:
            telemetry.event(
                "causalrag.model_mismatch",
                {
                    "causalrag.transition.index": offset,
                    "causalrag.step": step,
                    "causalrag.predictive_probability": mismatch.get("predictive_probability"),
                    "causalrag.surprisal": mismatch.get("surprisal"),
                    "causalrag.model_mismatch.suspicious": bool(mismatch.get("suspicious")),
                    "causalrag.model_mismatch.hard": bool(mismatch.get("hard_mismatch")),
                    "causalrag.model_mismatch.escalated": bool(mismatch.get("escalate")),
                    "causalrag.model_mismatch.posterior_suppressed": bool(mismatch.get("posterior_suppressed")),
                    "causalrag.discovery.hypothesis_ids": mismatch.get("discovered_hypotheses") or [],
                },
            )

        for evaluation in effects.get("temporal_effects") or []:
            telemetry.event(
                "causalrag.temporal_attribution",
                {
                    "causalrag.transition.index": offset,
                    "causalrag.temporal.effect_id": evaluation.get("effect_id"),
                    "causalrag.action.name": evaluation.get("intervention"),
                    "causalrag.temporal.prediction_hypothesis": evaluation.get("prediction_hypothesis"),
                    "causalrag.temporal.matched_prediction": evaluation.get("matched_prediction"),
                    "causalrag.temporal.lag_seconds": evaluation.get("lag_seconds"),
                    "causalrag.temporal.within_window": evaluation.get("within_window"),
                    **(
                        {
                            "causalrag.temporal.expected": evaluation.get("expected"),
                            "causalrag.temporal.observed": evaluation.get("observed"),
                        }
                        if telemetry.capture_content
                        else {}
                    ),
                },
            )

    for discovery in state.scratch.get("hypothesis_discovery_events", []):
        telemetry.event(
            "causalrag.hypothesis_discovery",
            {
                "causalrag.step": discovery.get("step"),
                "causalrag.discovery.hypothesis_ids": discovery.get("hypotheses") or [],
                "causalrag.discovery.trigger_experiment": discovery.get("trigger", {}).get("experiment_id"),
                "causalrag.discovery.trigger_surprisal": discovery.get("trigger", {}).get("surprisal"),
            },
        )


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
            "trace_id": self.state.scratch.get("trace_id"),
            "causal_trace": _jsonable(self.state.scratch.get("causal_trace", [])),
            "decisions": _jsonable(self.state.decisions),
            "observations": _jsonable(self.state.observations),
            "beliefs": _jsonable(snapshot),
            "hypotheses": _jsonable(snapshot.get("hypotheses", [])),
            "open_world": _jsonable(snapshot.get("open_world", {})),
            "transitions": _jsonable(self.world_model.transitions),
        }


class CausalAgent:
    """User-facing causal agent with optional retrieval and causal telemetry."""

    def __init__(
        self,
        loop: CausalAgentLoop,
        pipeline: Optional[Any] = None,
        telemetry: Optional[CausalTelemetry] = None,
    ) -> None:
        self.loop = loop
        self.pipeline = pipeline
        self.telemetry = telemetry or getattr(loop.tools, "telemetry", None) or CausalTelemetry.from_environment()
        self.loop.tools.telemetry = self.telemetry
        self._last_trace_start = 0

    def index(self, documents: Iterable[str]) -> "CausalAgent":
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
        trace_start = self.telemetry.count()
        transition_start = len(self.loop.world_model.transitions)
        self._last_trace_start = trace_start
        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "causalrag",
            "gen_ai.agent.description": "Explicit causal decision runtime with experiments, interventions, temporal attribution, and open-world mismatch detection.",
            "causalrag.max_steps": int(max_steps),
            "causalrag.goal_characters": len(goal),
        }
        if self.telemetry.capture_content:
            attributes["causalrag.goal"] = goal

        trace_id = ""
        with self.telemetry.span("invoke_agent causalrag", attributes) as run_span:
            trace_id = run_span.trace_id
            state = self.loop.run(goal=goal, max_steps=max_steps)
            _emit_causal_run_events(
                self.telemetry,
                state,
                self.loop.world_model,
                transition_start=transition_start,
            )
            run_span.set_attribute("causalrag.stop_reason", state.stop_reason or "")
            run_span.set_attribute("causalrag.steps", len(state.decisions))
            run_span.set_attribute("causalrag.executed_actions", state.step)
            run_span.set_attribute(
                "causalrag.open_world.model_mismatch",
                bool(self.loop.world_model.snapshot().get("open_world", {}).get("model_mismatch", False)),
            )
            self.telemetry.event(
                "causalrag.run.completed",
                {
                    "causalrag.stop_reason": state.stop_reason,
                    "causalrag.steps": len(state.decisions),
                    "causalrag.executed_actions": state.step,
                },
            )

        state.scratch["trace_id"] = trace_id
        state.scratch["causal_trace"] = self.telemetry.records(since=trace_start)
        answer = str(state.scratch.get("answer") or "")
        if not answer and state.observations:
            answer = str(state.observations[-1].result)
        return AgentRunResult(answer=answer, state=state, world_model=self.loop.world_model)

    def export_last_trace(self, path: str | Path) -> Path:
        return self.telemetry.export_jsonl(path, since=self._last_trace_start)

    @property
    def world_model(self) -> CausalWorldModel:
        return self.loop.world_model

    @property
    def tools(self) -> ToolRegistry:
        return self.loop.tools


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
    telemetry: Optional[CausalTelemetry] = None,
    decision_hook: Optional[Any] = None,
) -> CausalAgent:
    """Create a ready-to-run causal agent.

    ``telemetry`` always supports a bounded local causal trace and can also be
    connected to OpenTelemetry. Content capture is opt-in; prompts, tool
    arguments, observations, and final content are not exported by default.
    """
    telemetry = telemetry or CausalTelemetry.from_environment()
    registry = ToolRegistry(
        tools,
        decision_preferences=decision_preferences,
        telemetry=telemetry,
    )
    model_state = world_model or CausalWorldModel()
    pipeline = None

    wants_retrieval = (
        bool(documents or graph_path or index_path)
        if enable_retrieval is None
        else enable_retrieval
    )

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
        if hasattr(llm, "telemetry"):
            llm.telemetry = telemetry

    if reasoner is None:
        if llm is None:
            llm = LLMInterface(
                model=model_name,
                provider=provider,
                api_key=api_key,
                telemetry=telemetry,
            )
        elif hasattr(llm, "telemetry"):
            llm.telemetry = telemetry
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
        decision_hook=decision_hook,
    )
    return CausalAgent(loop=loop, pipeline=pipeline, telemetry=telemetry)
