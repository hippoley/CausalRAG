from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from causalrag.agent import ActionKind, CandidateAction
from causalrag.benchmarks.hidden_world import (
    HiddenWorldEnvironment,
    HiddenWorldReasoner,
    build_hvac_hidden_world,
)
from causalrag.benchmarks.temporal_hidden_world import (
    DelayedEffectEnvironment,
    ImmediateReadReasoner,
)
from causalrag.experiments import ExperimentContract, ModelMismatchPolicy, OutcomeLikelihood
from causalrag.reasoning.hypothesis import HypothesisProposal
from causalrag.tools import ToolSpec
from causalrag.world_model import CausalWorldModel


@dataclass
class ProbeScenarioRuntime:
    scenario_id: str
    environment: Any
    world_model: CausalWorldModel
    tools: Sequence[ToolSpec]
    default_reasoner: Any
    goal: str
    time_driver: Optional[Any] = None
    mismatch_policy: Optional[ModelMismatchPolicy] = None


def _metrics_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError(f"scenario metrics must be dict-like, got {type(value).__name__}")


class PlayableTemporalEnvironment(DelayedEffectEnvironment):
    """Delayed-effect benchmark adapted to the generic Playable Probe surface."""

    def tools(self) -> Sequence[ToolSpec]:
        # The same temporal contract is present regardless of runtime ablation.
        # The capability switch controls whether the runtime uses it.
        return super().tools(temporal_guard=True)

    def metrics(self, result) -> Dict[str, Any]:
        observed = next(
            (
                observation.result.get("status")
                for observation in result.state.observations
                if observation.action_name == "read_flow"
                and isinstance(observation.result, dict)
            ),
            None,
        )
        conclusion = result.answer or "unknown"
        active = {
            row.hypothesis_id: row.probability
            for row in result.world_model.hypotheses()
        }
        return {
            "hidden_hypothesis": self.hidden_hypothesis,
            "success": conclusion == self.hidden_hypothesis,
            "observed_status": observed,
            "conclusion": conclusion,
            "virtual_time_seconds": result.state.virtual_time_seconds,
            "wait_actions": sum(
                1
                for decision in result.state.decisions
                if decision.selected.kind == ActionKind.WAIT
            ),
            "premature_reads": self.premature_reads,
            "true_hypothesis_posterior": active.get(self.hidden_hypothesis, 0.0),
            "decision_rounds": len(result.state.decisions),
        }


def _open_world_contract(experiment_id: str, h1_rare: float, h2_rare: float):
    return ExperimentContract(
        experiment_id=experiment_id,
        outcomes=[
            OutcomeLikelihood(
                "ordinary",
                {"H1": 1.0 - h1_rare, "H2": 1.0 - h2_rare},
            ),
            OutcomeLikelihood(
                "novel_signature",
                {"H1": h1_rare, "H2": h2_rare},
            ),
        ],
    )


class PlayableOpenWorldEnvironment:
    """Closed initial hypothesis class with a repeatable out-of-model signature."""

    hidden_hypothesis = "H4"

    def __init__(self) -> None:
        self.reads = 0

    def world_model(self) -> CausalWorldModel:
        world = CausalWorldModel()
        world.upsert_hypothesis("H1", "Known fault one", probability=0.5)
        world.upsert_hypothesis("H2", "Known fault two", probability=0.5)
        return world

    def _novel(self):
        self.reads += 1
        return {"outcome": "novel_signature"}

    def tools(self) -> Sequence[ToolSpec]:
        return [
            ToolSpec(
                name="sensor_a",
                description="Independent residual sensor A.",
                handler=self._novel,
                experiment_contract=_open_world_contract("sensor_a_test", 0.04, 0.06),
                metadata={"kind": "observe", "benchmark": "open_world_probe"},
            ),
            ToolSpec(
                name="sensor_b",
                description="Independent residual sensor B.",
                handler=self._novel,
                experiment_contract=_open_world_contract("sensor_b_test", 0.05, 0.04),
                metadata={"kind": "observe", "benchmark": "open_world_probe"},
            ),
        ]

    def metrics(self, result) -> Dict[str, Any]:
        h4 = result.world_model.get_hypothesis("H4")
        success = bool(h4 is not None and h4.validated)
        snapshot = result.world_model.snapshot()
        return {
            "hidden_hypothesis": "H4",
            "success": success,
            "selected_hypothesis": (
                max(
                    result.world_model.hypotheses(),
                    key=lambda row: row.probability,
                ).hypothesis_id
                if result.world_model.hypotheses()
                else None
            ),
            "true_hypothesis_posterior": 0.0 if h4 is None else h4.probability,
            "discovered_hypothesis": h4 is not None,
            "discovered_validated": bool(h4 is not None and h4.validated),
            "model_mismatch_active": bool(snapshot["open_world"]["model_mismatch"]),
            "mismatch_warning_count": int(snapshot["open_world"]["mismatch_warning_count"]),
            "sensor_reads": self.reads,
            "decision_rounds": len(result.state.decisions),
        }


class OpenWorldProbeReasoner:
    """No-key reference policy that forces a falsifiable model-mismatch test."""

    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="sensor_a",
                    tests_hypotheses=["H1", "H2"],
                    rationale="First independent residual check.",
                )
            ]
        if state.step == 1:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="sensor_b",
                    tests_hypotheses=["H1", "H2"],
                    rationale="Second independent residual check.",
                )
            ]
        if state.step == 2:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="sensor_a",
                    tests_hypotheses=["H1", "H2", "H4"],
                    falsification_target="H4",
                    rationale="Re-test the novel signature against the discovered mechanism.",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "H4"},
                rationale="Stop after the out-of-model mechanism has been tested.",
            )
        ]

    def hypothesis_proposals(self, state, world_model):
        return []

    def discover_hypotheses(self, state, world_model, mismatch_context):
        return [
            HypothesisProposal(
                hypothesis_id="H4",
                statement="A previously unmodeled sensor drift mechanism causes the residual signature.",
                probability=0.99,
                rationale="Two independent residual measurements are unlikely under H1/H2.",
                falsifiers=["sensor A returns ordinary on repeat"],
                experiment_predictions={
                    "sensor_a_test": {
                        "ordinary": 0.05,
                        "novel_signature": 0.95,
                    },
                    "sensor_b_test": {
                        "ordinary": 0.10,
                        "novel_signature": 0.90,
                    },
                },
            )
        ]

    def uncertainty(self, state, world_model):
        return "whether the known model class is incomplete"


_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "hvac_hidden_world": {
        "id": "hvac_hidden_world",
        "label": "HVAC hidden mechanism",
        "description": "Diagnose filter, fan, or duct faults under noisy observations.",
        "hidden_hypotheses": ["H1", "H2", "H3"],
        "outcome_modes": ["deterministic", "stochastic"],
        "recommended_test": "Bayesian learning / EIG / EVSI",
        "default_hidden_hypothesis": "H2",
        "default_outcome_mode": "stochastic",
        "default_goal": "Identify the hidden HVAC causal mechanism using diagnostic experiments, then apply the intervention most likely to fix it.",
    },
    "temporal_delayed_effect": {
        "id": "temporal_delayed_effect",
        "label": "Delayed causal effect",
        "description": "Intervene on a valve; an immediate read is stale until transport delay elapses.",
        "hidden_hypotheses": ["H1", "H2"],
        "outcome_modes": ["deterministic"],
        "recommended_test": "Temporal attribution",
        "default_hidden_hypothesis": "H1",
        "default_outcome_mode": "deterministic",
        "default_goal": "Identify whether the valve or a downstream restriction controls flow without mistaking a stale immediate read for the intervention effect.",
    },
    "open_world_mismatch": {
        "id": "open_world_mismatch",
        "label": "Unknown mechanism / model mismatch",
        "description": "Observed signatures are improbable under every modeled H1/H2 explanation.",
        "hidden_hypotheses": ["H4"],
        "outcome_modes": ["deterministic"],
        "recommended_test": "Open-world mismatch + hypothesis discovery",
        "default_hidden_hypothesis": "H4",
        "default_outcome_mode": "deterministic",
        "default_goal": "Diagnose a failure that may lie outside the current modeled H1/H2 fault class and validate any newly discovered mechanism.",
    },
}


def scenario_summaries():
    return [dict(value) for value in _SCENARIOS.values()]


def validate_scenario_config(
    scenario_id: str,
    hidden_hypothesis: str,
    outcome_mode: str,
) -> None:
    spec = _SCENARIOS.get(str(scenario_id))
    if spec is None:
        raise ValueError(
            f"unknown scenario {scenario_id!r}; choose one of {sorted(_SCENARIOS)}"
        )
    if hidden_hypothesis not in spec["hidden_hypotheses"]:
        raise ValueError(
            f"{scenario_id} hidden_hypothesis must be one of {spec['hidden_hypotheses']}"
        )
    if outcome_mode not in spec["outcome_modes"]:
        raise ValueError(
            f"{scenario_id} outcome_mode must be one of {spec['outcome_modes']}"
        )


def build_scenario_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario_id = str(config.scenario)
    validate_scenario_config(
        scenario_id,
        str(config.hidden_hypothesis),
        str(config.outcome_mode),
    )

    if scenario_id == "hvac_hidden_world":
        scenario = build_hvac_hidden_world(config.hidden_hypothesis)
        environment = HiddenWorldEnvironment(
            scenario,
            outcome_mode=config.outcome_mode,
            seed=int(config.seed),
            outcome_coupling=config.stochastic_coupling,
        )
        return ProbeScenarioRuntime(
            scenario_id=scenario_id,
            environment=environment,
            world_model=environment.world_model(),
            tools=environment.tools(),
            default_reasoner=HiddenWorldReasoner(
                scenario,
                confidence_threshold=float(config.confidence_threshold),
                max_probes=int(config.max_probes),
            ),
            goal=str(_SCENARIOS[scenario_id]["default_goal"]),
        )

    if scenario_id == "temporal_delayed_effect":
        environment = PlayableTemporalEnvironment(config.hidden_hypothesis)
        return ProbeScenarioRuntime(
            scenario_id=scenario_id,
            environment=environment,
            world_model=environment.world_model(),
            tools=environment.tools(),
            default_reasoner=ImmediateReadReasoner(),
            goal=str(_SCENARIOS[scenario_id]["default_goal"]),
            time_driver=environment.clock,
        )

    environment = PlayableOpenWorldEnvironment()
    return ProbeScenarioRuntime(
        scenario_id=scenario_id,
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=OpenWorldProbeReasoner(),
        goal=str(_SCENARIOS[scenario_id]["default_goal"]),
        mismatch_policy=ModelMismatchPolicy(
            soft_predictive_threshold=0.06,
            hard_predictive_threshold=0.005,
            min_distinct_experiments=2,
            discovered_initial_probability=0.2,
        ),
    )


def scenario_metrics(environment: Any, result: Any) -> Dict[str, Any]:
    return _metrics_dict(environment.metrics(result))
