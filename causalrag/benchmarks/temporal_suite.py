from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from causalrag import (
    ActionKind,
    CandidateAction,
    CausalWorldModel,
    TemporalEffectContract,
    ToolSpec,
    VirtualTimeDriver,
    create_agent,
)

from .temporal_hidden_world import run_temporal_hidden_world


@dataclass
class TemporalEpisodeMetrics:
    scenario_id: str
    temporal_guard: bool
    success: bool
    conclusion: str
    wait_seconds: float
    premature_reads: int = 0
    missed_windows: int = 0
    contaminated_reads: int = 0
    interventions: int = 0

    @property
    def execution_regret(self) -> float:
        """Transparent benchmark-local cost, not a universal causal metric."""
        return (
            (0.0 if self.success else 1.0)
            + 0.25 * self.premature_reads
            + 0.50 * self.missed_windows
            + 0.50 * self.contaminated_reads
            + 0.01 * self.wait_seconds
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "temporal_guard": self.temporal_guard,
            "success": self.success,
            "conclusion": self.conclusion,
            "wait_seconds": self.wait_seconds,
            "premature_reads": self.premature_reads,
            "missed_windows": self.missed_windows,
            "contaminated_reads": self.contaminated_reads,
            "interventions": self.interventions,
            "execution_regret": self.execution_regret,
        }


@dataclass
class TemporalSuiteReport:
    temporal_guard: bool
    episodes: List[TemporalEpisodeMetrics]

    @property
    def success_rate(self) -> float:
        return sum(1 for episode in self.episodes if episode.success) / len(self.episodes)

    @property
    def mean_execution_regret(self) -> float:
        return sum(episode.execution_regret for episode in self.episodes) / len(self.episodes)

    def to_dict(self) -> Dict[str, object]:
        count = float(len(self.episodes))
        return {
            "temporal_guard": self.temporal_guard,
            "episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_execution_regret": self.mean_execution_regret,
            "mean_wait_seconds": sum(e.wait_seconds for e in self.episodes) / count,
            "premature_read_rate": sum(e.premature_reads for e in self.episodes) / count,
            "missed_window_rate": sum(e.missed_windows for e in self.episodes) / count,
            "contaminated_read_rate": sum(e.contaminated_reads for e in self.episodes) / count,
            "results": [episode.to_dict() for episode in self.episodes],
        }


def _base_world() -> CausalWorldModel:
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Intervention A controls the response.", probability=0.7)
    world.upsert_hypothesis("H2", "Another mechanism controls the response.", probability=0.3)
    return world


class LateWindowEnvironment:
    def __init__(self) -> None:
        self.clock = VirtualTimeDriver()
        self.intervention_time = None
        self.missed_reads = 0

    def intervene(self):
        self.intervention_time = self.clock.now_seconds
        return {"applied": True}

    def observe(self):
        elapsed = self.clock.now_seconds - float(self.intervention_time or 0.0)
        if elapsed < 5.0:
            return {"status": "baseline", "elapsed": elapsed}
        if elapsed <= 10.0:
            return {"status": "improved", "elapsed": elapsed}
        self.missed_reads += 1
        return {"status": "expired", "elapsed": elapsed}

    def tools(self, temporal_guard: bool) -> Sequence[ToolSpec]:
        contract = None
        if temporal_guard:
            contract = TemporalEffectContract(
                effect_id="transient_response",
                observe_with="observe_response",
                observation_key="status",
                earliest_seconds=5.0,
                latest_seconds=10.0,
                expected_outcomes={"H1": "improved", "H2": "unchanged"},
                protect_attribution=True,
            )
        return [
            ToolSpec(
                name="intervene",
                description="Apply intervention.",
                handler=self.intervene,
                metadata={"kind": "intervene"},
                temporal_effect_contract=contract,
            ),
            ToolSpec(
                name="observe_response",
                description="Observe the transient response.",
                handler=self.observe,
                metadata={"kind": "observe"},
            ),
        ]


class OverwaitThenObserveReasoner:
    def propose(self, state, world_model):
        if not state.observations:
            return [CandidateAction(ActionKind.INTERVENE, "intervene")]
        if not any(o.action_name == "observe_response" for o in state.observations):
            if state.virtual_time_seconds == 0.0:
                return [
                    CandidateAction(
                        ActionKind.WAIT,
                        "wait_too_long",
                        arguments={"seconds": 20.0},
                    )
                ]
            return [CandidateAction(ActionKind.OBSERVE, "observe_response")]
        status = next(
            o.result["status"]
            for o in reversed(state.observations)
            if o.action_name == "observe_response"
        )
        conclusion = "H1" if status == "improved" else "unknown"
        return [
            CandidateAction(
                ActionKind.STOP,
                "stop",
                arguments={"answer": conclusion},
                rationale="Use the transient response.",
            )
        ]

    def uncertainty(self, state, world_model):
        return "whether the transient response is still observable"


class ContaminationEnvironment:
    def __init__(self) -> None:
        self.clock = VirtualTimeDriver()
        self.a_time = None
        self.b_time = None
        self.contaminated_reads = 0

    def intervene_a(self):
        self.a_time = self.clock.now_seconds
        return {"applied": "A"}

    def intervene_b(self):
        self.b_time = self.clock.now_seconds
        return {"applied": "B"}

    def observe(self):
        elapsed = self.clock.now_seconds - float(self.a_time or 0.0)
        if elapsed < 5.0:
            return {"status": "baseline", "elapsed": elapsed}
        if self.b_time is not None:
            self.contaminated_reads += 1
            return {"status": "contaminated", "elapsed": elapsed}
        return {"status": "a_effect", "elapsed": elapsed}

    def tools(self, temporal_guard: bool) -> Sequence[ToolSpec]:
        contract = None
        if temporal_guard:
            contract = TemporalEffectContract(
                effect_id="a_marker",
                observe_with="observe_marker",
                observation_key="status",
                earliest_seconds=5.0,
                latest_seconds=10.0,
                expected_outcomes={"H1": "a_effect", "H2": "unchanged"},
                protect_attribution=True,
            )
        return [
            ToolSpec(
                name="intervene_a",
                description="Apply intervention A.",
                handler=self.intervene_a,
                metadata={"kind": "intervene"},
                temporal_effect_contract=contract,
            ),
            ToolSpec(
                name="intervene_b",
                description="Apply intervention B.",
                handler=self.intervene_b,
                metadata={"kind": "intervene"},
            ),
            ToolSpec(
                name="observe_marker",
                description="Observe A's marker.",
                handler=self.observe,
                metadata={"kind": "observe"},
            ),
        ]


class StackThenObserveReasoner:
    def propose(self, state, world_model):
        if not any(o.action_name == "intervene_a" for o in state.observations):
            return [CandidateAction(ActionKind.INTERVENE, "intervene_a")]
        if any(o.action_name == "observe_marker" for o in state.observations):
            status = next(
                o.result["status"]
                for o in reversed(state.observations)
                if o.action_name == "observe_marker"
            )
            conclusion = "H1" if status == "a_effect" else "unknown"
            return [
                CandidateAction(
                    ActionKind.STOP,
                    "stop",
                    arguments={"answer": conclusion},
                    rationale="Attribute the marker.",
                )
            ]
        if not any(o.action_name == "intervene_b" for o in state.observations):
            return [CandidateAction(ActionKind.INTERVENE, "intervene_b")]
        if state.virtual_time_seconds < 5.0:
            return [CandidateAction(ActionKind.WAIT, "wait", arguments={"seconds": 5.0})]
        return [CandidateAction(ActionKind.OBSERVE, "observe_marker")]

    def uncertainty(self, state, world_model):
        return "whether A caused the marker before B contaminated attribution"


def _run_late_window(temporal_guard: bool) -> TemporalEpisodeMetrics:
    environment = LateWindowEnvironment()
    agent = create_agent(
        reasoner=OverwaitThenObserveReasoner(),
        world_model=_base_world(),
        tools=environment.tools(temporal_guard),
        time_driver=environment.clock,
    )
    result = agent.run("Identify the transient causal effect.", max_steps=6)
    missed_events = sum(
        1
        for event in result.state.scratch.get("temporal_events", [])
        if event.get("kind") == "missed_observation_window"
    )
    return TemporalEpisodeMetrics(
        scenario_id="late_window",
        temporal_guard=temporal_guard,
        success=result.answer == "H1",
        conclusion=result.answer or "unknown",
        wait_seconds=result.state.virtual_time_seconds,
        missed_windows=max(environment.missed_reads, missed_events),
        interventions=sum(o.action_name == "intervene" for o in result.state.observations),
    )


def _run_stacking(temporal_guard: bool) -> TemporalEpisodeMetrics:
    environment = ContaminationEnvironment()
    agent = create_agent(
        reasoner=StackThenObserveReasoner(),
        world_model=_base_world(),
        tools=environment.tools(temporal_guard),
        time_driver=environment.clock,
    )
    result = agent.run("Identify A's effect without contaminating attribution.", max_steps=7)
    return TemporalEpisodeMetrics(
        scenario_id="stacked_intervention",
        temporal_guard=temporal_guard,
        success=result.answer == "H1",
        conclusion=result.answer or "unknown",
        wait_seconds=result.state.virtual_time_seconds,
        contaminated_reads=environment.contaminated_reads,
        interventions=sum(
            o.action_name in {"intervene_a", "intervene_b"}
            for o in result.state.observations
        ),
    )


def run_temporal_suite(temporal_guard: bool) -> TemporalSuiteReport:
    episodes: List[TemporalEpisodeMetrics] = []
    for hidden in ("H1", "H2"):
        metrics, _ = run_temporal_hidden_world(hidden, temporal_guard=temporal_guard)
        episodes.append(
            TemporalEpisodeMetrics(
                scenario_id=f"early_{hidden.lower()}",
                temporal_guard=temporal_guard,
                success=metrics.success,
                conclusion=metrics.conclusion,
                wait_seconds=metrics.virtual_time_seconds,
                premature_reads=metrics.premature_reads,
                interventions=1,
            )
        )
    episodes.append(_run_late_window(temporal_guard))
    episodes.append(_run_stacking(temporal_guard))
    return TemporalSuiteReport(temporal_guard=temporal_guard, episodes=episodes)


def compare_temporal_suite() -> Dict[str, Dict[str, object]]:
    naive = run_temporal_suite(False)
    guarded = run_temporal_suite(True)
    return {"naive": naive.to_dict(), "temporal_guard": guarded.to_dict()}
