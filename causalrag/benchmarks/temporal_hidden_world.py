from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from causalrag import (
    ActionKind,
    CandidateAction,
    CausalWorldModel,
    TemporalEffectContract,
    ToolSpec,
    VirtualTimeDriver,
    create_agent,
)


@dataclass
class TemporalHiddenWorldMetrics:
    hidden_hypothesis: str
    temporal_guard: bool
    observed_status: Optional[str]
    conclusion: str
    success: bool
    virtual_time_seconds: float
    wait_actions: int
    premature_reads: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_hypothesis": self.hidden_hypothesis,
            "temporal_guard": self.temporal_guard,
            "observed_status": self.observed_status,
            "conclusion": self.conclusion,
            "success": self.success,
            "virtual_time_seconds": self.virtual_time_seconds,
            "wait_actions": self.wait_actions,
            "premature_reads": self.premature_reads,
        }


class DelayedEffectEnvironment:
    """A tiny world where reading too early returns a stale baseline."""

    def __init__(self, hidden_hypothesis: str, lag_seconds: float = 5.0) -> None:
        if hidden_hypothesis not in {"H1", "H2"}:
            raise ValueError("hidden_hypothesis must be H1 or H2")
        self.hidden_hypothesis = hidden_hypothesis
        self.lag_seconds = float(lag_seconds)
        self.clock = VirtualTimeDriver()
        self.intervention_time: Optional[float] = None
        self.premature_reads = 0

    def open_valve(self):
        self.intervention_time = self.clock.now_seconds
        return {"applied": True, "at": self.intervention_time}

    def read_flow(self):
        if self.intervention_time is None:
            return {"status": "baseline", "at": self.clock.now_seconds}
        elapsed = self.clock.now_seconds - self.intervention_time
        if elapsed < self.lag_seconds:
            self.premature_reads += 1
            return {
                "status": "baseline",
                "at": self.clock.now_seconds,
                "elapsed": elapsed,
                "stale": True,
            }
        status = "improved" if self.hidden_hypothesis == "H1" else "unchanged"
        return {
            "status": status,
            "at": self.clock.now_seconds,
            "elapsed": elapsed,
            "stale": False,
        }

    def world_model(self) -> CausalWorldModel:
        world = CausalWorldModel()
        world.upsert_hypothesis("H1", "The valve is the flow bottleneck.", probability=0.5)
        world.upsert_hypothesis("H2", "A downstream restriction is the flow bottleneck.", probability=0.5)
        return world

    def tools(self, temporal_guard: bool) -> Sequence[ToolSpec]:
        contract = None
        if temporal_guard:
            contract = TemporalEffectContract(
                effect_id="valve_flow_response",
                observe_with="read_flow",
                observation_key="status",
                earliest_seconds=self.lag_seconds,
                latest_seconds=self.lag_seconds * 3.0,
                expected_outcomes={"H1": "improved", "H2": "unchanged"},
                falsification_weight=0.5,
                description="Flow response should only be judged after transport delay.",
            )
        return [
            ToolSpec(
                name="open_valve",
                description="Open the upstream valve.",
                handler=self.open_valve,
                metadata={"kind": "intervene", "benchmark": "temporal_hidden_world"},
                temporal_effect_contract=contract,
            ),
            ToolSpec(
                name="read_flow",
                description="Read the downstream flow response.",
                handler=self.read_flow,
                metadata={"kind": "observe", "benchmark": "temporal_hidden_world"},
            ),
        ]


class ImmediateReadReasoner:
    """Intentionally naive proposer: intervene, then read immediately."""

    def propose(self, state, world_model) -> Sequence[CandidateAction]:
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.INTERVENE,
                    name="open_valve",
                    rationale="Change the suspected cause.",
                )
            ]

        flow_observations = [
            observation
            for observation in state.observations
            if observation.action_name == "read_flow"
        ]
        if not flow_observations:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_flow",
                    rationale="Read the effect immediately after intervention.",
                )
            ]

        status = flow_observations[-1].result.get("status")
        if status == "improved":
            conclusion = "H1"
        elif status == "unchanged":
            conclusion = "H2"
        else:
            conclusion = "unknown"
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": conclusion},
                rationale="Use the observed response as the causal conclusion.",
            )
        ]

    def uncertainty(self, state, world_model):
        return "valve bottleneck versus downstream restriction"


def run_temporal_hidden_world(
    hidden_hypothesis: str = "H1",
    temporal_guard: bool = True,
    lag_seconds: float = 5.0,
) -> Tuple[TemporalHiddenWorldMetrics, object]:
    environment = DelayedEffectEnvironment(hidden_hypothesis, lag_seconds=lag_seconds)
    agent = create_agent(
        reasoner=ImmediateReadReasoner(),
        world_model=environment.world_model(),
        tools=environment.tools(temporal_guard=temporal_guard),
        time_driver=environment.clock,
    )
    result = agent.run(
        "Identify whether the valve or downstream restriction controls flow.",
        max_steps=5,
    )
    observed = next(
        (
            observation.result.get("status")
            for observation in result.state.observations
            if observation.action_name == "read_flow"
        ),
        None,
    )
    conclusion = result.answer or "unknown"
    metrics = TemporalHiddenWorldMetrics(
        hidden_hypothesis=hidden_hypothesis,
        temporal_guard=temporal_guard,
        observed_status=observed,
        conclusion=conclusion,
        success=conclusion == hidden_hypothesis,
        virtual_time_seconds=result.state.virtual_time_seconds,
        wait_actions=sum(
            1
            for decision in result.state.decisions
            if decision.selected.kind == ActionKind.WAIT
        ),
        premature_reads=environment.premature_reads,
    )
    return metrics, result


def compare_temporal_guard() -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for hidden in ("H1", "H2"):
        naive, _ = run_temporal_hidden_world(hidden, temporal_guard=False)
        guarded, _ = run_temporal_hidden_world(hidden, temporal_guard=True)
        report[hidden] = {
            "naive": naive.to_dict(),
            "temporal_guard": guarded.to_dict(),
        }
    return report
