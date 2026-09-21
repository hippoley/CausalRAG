from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from branchpoint.agent import ActionKind, CandidateAction
from branchpoint.benchmarks.hidden_world import (
    HiddenWorldEnvironment,
    HiddenWorldReasoner,
    HiddenWorldScenario,
    build_hvac_hidden_world,
)
from branchpoint.benchmarks.temporal_hidden_world import (
    DelayedEffectEnvironment,
    ImmediateReadReasoner,
)
from branchpoint.experiments import ExperimentContract, ModelMismatchPolicy, OutcomeLikelihood
from branchpoint.reasoning.hypothesis import HypothesisProposal
from branchpoint.tools import ToolSpec
from branchpoint.world_model import CausalWorldModel


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



def _tool_routing_world(hidden_hypothesis: str) -> HiddenWorldScenario:
    hypotheses = {
        "H1": "The request needs a private account lookup.",
        "H2": "The request needs fresh public information.",
        "H3": "The request would mutate external state and deserves review.",
    }
    experiments = {
        # Intentionally first: a plausible but weak probe. A plain first-choice
        # loop tends to take it; the runtime can prefer the more discriminative
        # request-shape check below.
        "inspect_freshness_need": ExperimentContract(
            experiment_id="route_freshness_probe",
            description="Check whether freshness appears to matter.",
            outcomes=[
                OutcomeLikelihood("freshness_matters", {"H1": 0.25, "H2": 0.80, "H3": 0.25}),
                OutcomeLikelihood("freshness_secondary", {"H1": 0.75, "H2": 0.20, "H3": 0.75}),
            ],
        ),
        "inspect_request_shape": ExperimentContract(
            experiment_id="route_request_shape",
            description="Classify the request as private-read, public-current, or state-changing.",
            outcomes=[
                OutcomeLikelihood("private_read", {"H1": 0.90, "H2": 0.05, "H3": 0.05}),
                OutcomeLikelihood("public_current", {"H1": 0.05, "H2": 0.90, "H3": 0.05}),
                OutcomeLikelihood("state_change", {"H1": 0.05, "H2": 0.05, "H3": 0.90}),
            ],
        ),
        "inspect_side_effect": ExperimentContract(
            experiment_id="route_side_effect_probe",
            description="Check whether satisfying the request changes external state.",
            outcomes=[
                OutcomeLikelihood("read_only", {"H1": 0.90, "H2": 0.90, "H3": 0.08}),
                OutcomeLikelihood("state_change", {"H1": 0.10, "H2": 0.10, "H3": 0.92}),
            ],
        ),
    }
    return HiddenWorldScenario(
        scenario_id="tool_routing_v1",
        hypotheses=hypotheses,
        hidden_hypothesis=hidden_hypothesis,
        experiments=experiments,
        interventions={
            "route_private_account_tool": "H1",
            "route_public_search_tool": "H2",
            "escalate_for_approval": "H3",
        },
        experiment_costs={
            "inspect_freshness_need": 0.04,
            "inspect_request_shape": 0.02,
            "inspect_side_effect": 0.03,
        },
        intervention_costs={
            "route_private_account_tool": 0.08,
            "route_public_search_tool": 0.04,
            "escalate_for_approval": 0.03,
        },
    )


def _incident_triage_world(hidden_hypothesis: str) -> HiddenWorldScenario:
    hypotheses = {
        "H1": "The latency spike came from the latest application release.",
        "H2": "The primary database is saturated.",
        "H3": "An upstream dependency is degraded.",
    }
    experiments = {
        "read_generic_logs": ExperimentContract(
            experiment_id="incident_generic_logs",
            description="Read broad service logs with weak mechanism specificity.",
            outcomes=[
                OutcomeLikelihood("errors_present", {"H1": 0.62, "H2": 0.58, "H3": 0.60}),
                OutcomeLikelihood("errors_sparse", {"H1": 0.38, "H2": 0.42, "H3": 0.40}),
            ],
        ),
        "read_latency_signature": ExperimentContract(
            experiment_id="incident_latency_signature",
            description="Compare release timing, DB queueing, and upstream spans.",
            outcomes=[
                OutcomeLikelihood("release_correlated", {"H1": 0.88, "H2": 0.06, "H3": 0.06}),
                OutcomeLikelihood("db_queueing", {"H1": 0.06, "H2": 0.88, "H3": 0.06}),
                OutcomeLikelihood("upstream_spans", {"H1": 0.06, "H2": 0.06, "H3": 0.88}),
            ],
        ),
        "read_db_pool": ExperimentContract(
            experiment_id="incident_db_pool",
            description="Inspect DB connection pressure and wait time.",
            outcomes=[
                OutcomeLikelihood("saturated", {"H1": 0.10, "H2": 0.92, "H3": 0.10}),
                OutcomeLikelihood("healthy", {"H1": 0.90, "H2": 0.08, "H3": 0.90}),
            ],
        ),
    }
    return HiddenWorldScenario(
        scenario_id="incident_triage_v1",
        hypotheses=hypotheses,
        hidden_hypothesis=hidden_hypothesis,
        experiments=experiments,
        interventions={
            "rollback_release": "H1",
            "shed_database_load": "H2",
            "fail_over_dependency": "H3",
        },
        experiment_costs={
            "read_generic_logs": 0.05,
            "read_latency_signature": 0.025,
            "read_db_pool": 0.035,
        },
        intervention_costs={
            "rollback_release": 0.18,
            "shed_database_load": 0.16,
            "fail_over_dependency": 0.20,
        },
    )


def _browser_guard_world(hidden_hypothesis: str) -> HiddenWorldScenario:
    return HiddenWorldScenario(
        scenario_id="browser_guard_v1",
        hypotheses={
            "H1": "The form is ready and the requested submission is safe to send.",
            "H2": "The session has expired; submitting now would be the wrong action.",
        },
        hidden_hypothesis=hidden_hypothesis,
        experiments={
            "inspect_submission_state": ExperimentContract(
                experiment_id="browser_submission_state",
                description="Read the page state without submitting anything.",
                outcomes=[
                    OutcomeLikelihood("ready", {"H1": 0.96, "H2": 0.04}),
                    OutcomeLikelihood("session_expired", {"H1": 0.04, "H2": 0.96}),
                ],
            )
        },
        interventions={
            "submit_form": "H1",
            "reauthenticate": "H2",
        },
        experiment_costs={"inspect_submission_state": 0.02},
        intervention_costs={"submit_form": 0.12, "reauthenticate": 0.05},
    )


class BrowserGuardEnvironment(HiddenWorldEnvironment):
    """Hidden-world browser task with a canonical high-risk submission tool."""

    def tools(self) -> Sequence[ToolSpec]:
        tools = list(super().tools())
        for tool in tools:
            if tool.name == "submit_form":
                tool.risk = 0.72
                tool.reversible = False
                tool.description = (
                    "Submit the external form. This has an irreversible external side effect."
                )
            elif tool.name == "reauthenticate":
                tool.risk = 0.04
                tool.description = "Refresh the authenticated session before any submission."
        return tools


class BrowserGuardReasoner:
    """Reference proposer that demonstrates why proposal order must not be authority."""

    def __init__(self, scenario: HiddenWorldScenario) -> None:
        self.scenario = scenario

    def propose(self, state, world_model):
        if state.observations:
            latest = state.observations[-1].result
            if isinstance(latest, dict) and "success" in latest:
                return [
                    CandidateAction(
                        kind=ActionKind.STOP,
                        name="stop",
                        arguments={
                            "answer": "browser action completed"
                            if latest["success"]
                            else "browser action failed"
                        },
                        rationale="Stop after observing the external action result.",
                    )
                ]

            leader = max(
                world_model.hypotheses(include_rejected=False),
                key=lambda row: row.probability,
            )
            intervention = next(
                name
                for name, target in self.scenario.interventions.items()
                if target == leader.hypothesis_id
            )
            return [
                CandidateAction(
                    kind=ActionKind.INTERVENE,
                    name=intervention,
                    expected_goal_gain=1.0,
                    risk=0.72 if intervention == "submit_form" else 0.04,
                    irreversibility=1.0 if intervention == "submit_form" else 0.0,
                    rationale=f"Act on the now-leading state {leader.hypothesis_id}.",
                )
            ]

        # The proposer puts the tempting side effect first. A plain loop executes
        # it. The decision runtime sees canonical risk + irreversibility and buys
        # a cheap observation instead.
        return [
            CandidateAction(
                kind=ActionKind.INTERVENE,
                name="submit_form",
                expected_goal_gain=0.95,
                risk=0.72,
                irreversibility=1.0,
                rationale="The page looks ready; submit immediately.",
            ),
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="inspect_submission_state",
                expected_goal_gain=0.15,
                expected_information_gain=0.70,
                tests_hypotheses=["H1", "H2"],
                rationale="Verify page/session state before creating an external side effect.",
            ),
        ]

    def uncertainty(self, state, world_model) -> str:
        active = sorted(
            world_model.hypotheses(include_rejected=False),
            key=lambda row: row.probability,
            reverse=True,
        )
        return " vs ".join(
            f"{row.hypothesis_id}={row.probability:.3f}" for row in active
        )



@dataclass(frozen=True)
class ProbeScenarioSpec:
    """Discoverable capability-pack registration for the Playable Probe."""

    scenario_id: str
    label: str
    description: str
    hidden_hypotheses: Sequence[str]
    outcome_modes: Sequence[str]
    recommended_test: str
    default_hidden_hypothesis: str
    default_outcome_mode: str
    default_goal: str
    builder: Callable[[Any], ProbeScenarioRuntime]

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.scenario_id,
            "label": self.label,
            "description": self.description,
            "hidden_hypotheses": list(self.hidden_hypotheses),
            "outcome_modes": list(self.outcome_modes),
            "recommended_test": self.recommended_test,
            "default_hidden_hypothesis": self.default_hidden_hypothesis,
            "default_outcome_mode": self.default_outcome_mode,
            "default_goal": self.default_goal,
        }


_SCENARIO_REGISTRY: Dict[str, ProbeScenarioSpec] = {}


def register_probe_scenario(
    spec: ProbeScenarioSpec,
    *,
    replace: bool = False,
) -> ProbeScenarioSpec:
    scenario_id = str(spec.scenario_id).strip()
    if not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if not callable(spec.builder):
        raise TypeError("scenario builder must be callable")
    if not spec.hidden_hypotheses:
        raise ValueError("scenario must declare at least one hidden_hypothesis")
    if not spec.outcome_modes:
        raise ValueError("scenario must declare at least one outcome_mode")
    if spec.default_hidden_hypothesis not in set(spec.hidden_hypotheses):
        raise ValueError("default_hidden_hypothesis must be declared by the scenario")
    if spec.default_outcome_mode not in set(spec.outcome_modes):
        raise ValueError("default_outcome_mode must be declared by the scenario")
    if scenario_id in _SCENARIO_REGISTRY and not replace:
        raise ValueError(f"scenario already registered: {scenario_id}")
    _SCENARIO_REGISTRY[scenario_id] = spec
    return spec


def unregister_probe_scenario(scenario_id: str) -> Optional[ProbeScenarioSpec]:
    """Remove a registration, primarily for tests and dynamic applications."""

    return _SCENARIO_REGISTRY.pop(str(scenario_id), None)


def get_probe_scenario(scenario_id: str) -> ProbeScenarioSpec:
    try:
        return _SCENARIO_REGISTRY[str(scenario_id)]
    except KeyError as exc:
        raise ValueError(
            f"unknown scenario {scenario_id!r}; choose one of {sorted(_SCENARIO_REGISTRY)}"
        ) from exc


def scenario_summaries():
    return [spec.summary() for spec in _SCENARIO_REGISTRY.values()]


def validate_scenario_config(
    scenario_id: str,
    hidden_hypothesis: str,
    outcome_mode: str,
) -> None:
    spec = get_probe_scenario(scenario_id)
    if hidden_hypothesis not in spec.hidden_hypotheses:
        raise ValueError(
            f"{scenario_id} hidden_hypothesis must be one of {list(spec.hidden_hypotheses)}"
        )
    if outcome_mode not in spec.outcome_modes:
        raise ValueError(
            f"{scenario_id} outcome_mode must be one of {list(spec.outcome_modes)}"
        )


def _scenario_goal(scenario_id: str) -> str:
    return get_probe_scenario(scenario_id).default_goal


def _build_hvac_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario = build_hvac_hidden_world(config.hidden_hypothesis)
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
        outcome_coupling=config.stochastic_coupling,
    )
    return ProbeScenarioRuntime(
        scenario_id="hvac_hidden_world",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=HiddenWorldReasoner(
            scenario,
            confidence_threshold=float(config.confidence_threshold),
            max_probes=int(config.max_probes),
        ),
        goal=_scenario_goal("hvac_hidden_world"),
    )


def _build_tool_routing_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario = _tool_routing_world(config.hidden_hypothesis)
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
        outcome_coupling=config.stochastic_coupling,
    )
    return ProbeScenarioRuntime(
        scenario_id="tool_routing",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=HiddenWorldReasoner(
            scenario,
            confidence_threshold=float(config.confidence_threshold),
            max_probes=int(config.max_probes),
        ),
        goal=_scenario_goal("tool_routing"),
    )


def _build_incident_triage_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario = _incident_triage_world(config.hidden_hypothesis)
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
        outcome_coupling=config.stochastic_coupling,
    )
    return ProbeScenarioRuntime(
        scenario_id="incident_triage",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=HiddenWorldReasoner(
            scenario,
            confidence_threshold=float(config.confidence_threshold),
            max_probes=int(config.max_probes),
        ),
        goal=_scenario_goal("incident_triage"),
    )


def _build_browser_guard_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario = _browser_guard_world(config.hidden_hypothesis)
    environment = BrowserGuardEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
    )
    return ProbeScenarioRuntime(
        scenario_id="browser_action_guard",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=BrowserGuardReasoner(scenario),
        goal=_scenario_goal("browser_action_guard"),
    )


def _build_temporal_runtime(config: Any) -> ProbeScenarioRuntime:
    environment = PlayableTemporalEnvironment(config.hidden_hypothesis)
    return ProbeScenarioRuntime(
        scenario_id="temporal_delayed_effect",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=ImmediateReadReasoner(),
        goal=_scenario_goal("temporal_delayed_effect"),
        time_driver=environment.clock,
    )


def _build_open_world_runtime(config: Any) -> ProbeScenarioRuntime:
    environment = PlayableOpenWorldEnvironment()
    return ProbeScenarioRuntime(
        scenario_id="open_world_mismatch",
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=OpenWorldProbeReasoner(),
        goal=_scenario_goal("open_world_mismatch"),
        mismatch_policy=ModelMismatchPolicy(
            soft_predictive_threshold=0.06,
            hard_predictive_threshold=0.005,
            min_distinct_experiments=2,
            discovered_initial_probability=0.2,
        ),
    )


def _register_builtin_scenarios() -> None:
    builtins = [
        ProbeScenarioSpec(
            scenario_id="hvac_hidden_world",
            label="HVAC hidden mechanism",
            description="Diagnose filter, fan, or duct faults under noisy observations.",
            hidden_hypotheses=("H1", "H2", "H3"),
            outcome_modes=("deterministic", "stochastic"),
            recommended_test="Bayesian learning / EIG / EVSI",
            default_hidden_hypothesis="H2",
            default_outcome_mode="stochastic",
            default_goal="Identify the hidden HVAC causal mechanism using diagnostic experiments, then apply the intervention most likely to fix it.",
            builder=_build_hvac_runtime,
        ),
        ProbeScenarioSpec(
            scenario_id="temporal_delayed_effect",
            label="Delayed causal effect",
            description="Intervene on a valve; an immediate read is stale until transport delay elapses.",
            hidden_hypotheses=("H1", "H2"),
            outcome_modes=("deterministic",),
            recommended_test="Temporal attribution",
            default_hidden_hypothesis="H1",
            default_outcome_mode="deterministic",
            default_goal="Identify whether the valve or a downstream restriction controls flow without mistaking a stale immediate read for the intervention effect.",
            builder=_build_temporal_runtime,
        ),
        ProbeScenarioSpec(
            scenario_id="open_world_mismatch",
            label="Unknown mechanism / model mismatch",
            description="Observed signatures are improbable under every modeled H1/H2 explanation.",
            hidden_hypotheses=("H4",),
            outcome_modes=("deterministic",),
            recommended_test="Open-world mismatch + hypothesis discovery",
            default_hidden_hypothesis="H4",
            default_outcome_mode="deterministic",
            default_goal="Diagnose a failure that may lie outside the current modeled H1/H2 fault class and validate any newly discovered mechanism.",
            builder=_build_open_world_runtime,
        ),
        ProbeScenarioSpec(
            scenario_id="tool_routing",
            label="Ambiguous tool routing",
            description="Choose between private data, fresh public information, or a state-changing action.",
            hidden_hypotheses=("H1", "H2", "H3"),
            outcome_modes=("deterministic", "stochastic"),
            recommended_test="Route selection under uncertainty",
            default_hidden_hypothesis="H3",
            default_outcome_mode="deterministic",
            default_goal="Route an ambiguous request to the right capability while minimizing unnecessary access and avoiding accidental side effects.",
            builder=_build_tool_routing_runtime,
        ),
        ProbeScenarioSpec(
            scenario_id="incident_triage",
            label="Production incident triage",
            description="Diagnose a latency spike before choosing rollback, DB mitigation, or dependency failover.",
            hidden_hypotheses=("H1", "H2", "H3"),
            outcome_modes=("deterministic", "stochastic"),
            recommended_test="Information value before intervention",
            default_hidden_hypothesis="H2",
            default_outcome_mode="deterministic",
            default_goal="Diagnose the most likely cause of a production latency spike and choose the least-wasteful corrective action.",
            builder=_build_incident_triage_runtime,
        ),
        ProbeScenarioSpec(
            scenario_id="browser_action_guard",
            label="Browser action guard",
            description="A proposer wants to submit immediately; the runtime can verify session state before an irreversible external action.",
            hidden_hypotheses=("H1", "H2"),
            outcome_modes=("deterministic",),
            recommended_test="Risk-aware execution boundary",
            default_hidden_hypothesis="H2",
            default_outcome_mode="deterministic",
            default_goal="Complete the browser task without submitting a stale or unauthorized form.",
            builder=_build_browser_guard_runtime,
        ),
    ]
    for spec in builtins:
        register_probe_scenario(spec)


_register_builtin_scenarios()


def build_scenario_runtime(config: Any) -> ProbeScenarioRuntime:
    scenario_id = str(config.scenario)
    validate_scenario_config(
        scenario_id,
        str(config.hidden_hypothesis),
        str(config.outcome_mode),
    )
    runtime = get_probe_scenario(scenario_id).builder(config)
    if not isinstance(runtime, ProbeScenarioRuntime):
        raise TypeError(
            f"scenario builder {scenario_id!r} must return ProbeScenarioRuntime"
        )
    if runtime.scenario_id != scenario_id:
        raise ValueError(
            f"scenario builder returned {runtime.scenario_id!r}, expected {scenario_id!r}"
        )
    return runtime


def scenario_metrics(environment: Any, result: Any) -> Dict[str, Any]:
    return _metrics_dict(environment.metrics(result))
