from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from causalrag import ActionKind, CandidateAction, ToolSpec, create_agent
from causalrag.experiments import ExperimentContract, OutcomeLikelihood
from causalrag.world_model import CausalWorldModel


@dataclass(frozen=True)
class HiddenWorldScenario:
    """A reproducible hidden-mechanism decision problem."""

    scenario_id: str
    hypotheses: Mapping[str, str]
    hidden_hypothesis: str
    experiments: Mapping[str, ExperimentContract]
    interventions: Mapping[str, str]
    experiment_costs: Mapping[str, float] = field(default_factory=dict)
    intervention_costs: Mapping[str, float] = field(default_factory=dict)
    failure_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_hypothesis not in self.hypotheses:
            raise ValueError("hidden_hypothesis must be declared in hypotheses")
        for name, contract in self.experiments.items():
            if set(contract.hypothesis_ids()) != set(self.hypotheses):
                raise ValueError(
                    f"experiment {name} must model exactly the scenario hypotheses"
                )
        for intervention, target in self.interventions.items():
            if target not in self.hypotheses:
                raise ValueError(
                    f"intervention {intervention} targets unknown hypothesis {target}"
                )


@dataclass
class HiddenWorldMetrics:
    scenario_id: str
    hidden_hypothesis: str
    selected_hypothesis: str
    selected_intervention: Optional[str]
    success: bool
    probes: int
    interventions: int
    decision_rounds: int
    total_cost: float
    true_hypothesis_posterior: float
    identification_correct: bool
    causal_regret: float
    brier_score: float
    seed: Optional[int] = None
    outcome_mode: str = "deterministic"

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hidden_hypothesis": self.hidden_hypothesis,
            "selected_hypothesis": self.selected_hypothesis,
            "selected_intervention": self.selected_intervention,
            "success": self.success,
            "probes": self.probes,
            "interventions": self.interventions,
            "decision_rounds": self.decision_rounds,
            "total_cost": self.total_cost,
            "true_hypothesis_posterior": self.true_hypothesis_posterior,
            "identification_correct": self.identification_correct,
            "causal_regret": self.causal_regret,
            "brier_score": self.brier_score,
            "seed": self.seed,
            "outcome_mode": self.outcome_mode,
        }


@dataclass
class HiddenWorldSuiteReport:
    episodes: int
    success_rate: float
    identification_accuracy: float
    mean_true_hypothesis_posterior: float
    mean_brier_score: float
    mean_probes: float
    mean_total_cost: float
    mean_causal_regret: float
    per_hidden: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "identification_accuracy": self.identification_accuracy,
            "mean_true_hypothesis_posterior": self.mean_true_hypothesis_posterior,
            "mean_brier_score": self.mean_brier_score,
            "mean_probes": self.mean_probes,
            "mean_total_cost": self.mean_total_cost,
            "mean_causal_regret": self.mean_causal_regret,
            "per_hidden": self.per_hidden,
        }


class HiddenWorldEnvironment:
    def __init__(
        self,
        scenario: HiddenWorldScenario,
        outcome_mode: str = "deterministic",
        seed: Optional[int] = None,
    ) -> None:
        if outcome_mode not in {"deterministic", "stochastic"}:
            raise ValueError("outcome_mode must be 'deterministic' or 'stochastic'")
        self.scenario = scenario
        self.outcome_mode = outcome_mode
        self.seed = seed
        self._rng = random.Random(seed)
        self.probes = 0
        self.interventions = 0
        self.total_cost = 0.0
        self.selected_intervention: Optional[str] = None
        self.last_intervention_success = False

    def _deterministic_outcome(self, contract: ExperimentContract) -> str:
        hidden = self.scenario.hidden_hypothesis
        return max(
            contract.outcome_labels(),
            key=lambda label: contract.likelihood(label, hidden),
        )

    def _stochastic_outcome(self, contract: ExperimentContract) -> str:
        hidden = self.scenario.hidden_hypothesis
        draw = self._rng.random()
        cumulative = 0.0
        labels = contract.outcome_labels()
        for label in labels:
            cumulative += contract.likelihood(label, hidden)
            if draw <= cumulative:
                return label
        return labels[-1]

    def _outcome(self, contract: ExperimentContract) -> str:
        if self.outcome_mode == "stochastic":
            return self._stochastic_outcome(contract)
        return self._deterministic_outcome(contract)

    def experiment_tool(self, name: str, contract: ExperimentContract) -> ToolSpec:
        cost = float(self.scenario.experiment_costs.get(name, 0.05))

        def handler() -> Dict[str, object]:
            self.probes += 1
            self.total_cost += cost
            outcome = self._outcome(contract)
            return {
                contract.outcome_key: outcome,
                "experiment_id": contract.experiment_id,
            }

        return ToolSpec(
            name=name,
            description=contract.description or f"Run experiment {contract.experiment_id}",
            handler=handler,
            cost=cost,
            risk=0.0,
            reversible=True,
            metadata={"kind": "observe", "benchmark": "hidden_world"},
            experiment_contract=contract,
        )

    def intervention_tool(self, name: str, target: str) -> ToolSpec:
        cost = float(self.scenario.intervention_costs.get(name, 0.2))

        def handler() -> Dict[str, object]:
            self.interventions += 1
            self.total_cost += cost
            self.selected_intervention = name
            self.last_intervention_success = target == self.scenario.hidden_hypothesis
            return {
                "success": self.last_intervention_success,
                "intervention": name,
                "target_hypothesis": target,
            }

        return ToolSpec(
            name=name,
            description=f"Apply intervention for {target}: {self.scenario.hypotheses[target]}",
            handler=handler,
            cost=cost,
            risk=0.0,
            reversible=True,
            metadata={"kind": "intervene", "benchmark": "hidden_world"},
        )

    def tools(self) -> List[ToolSpec]:
        tools = [
            self.experiment_tool(name, contract)
            for name, contract in self.scenario.experiments.items()
        ]
        tools.extend(
            self.intervention_tool(name, target)
            for name, target in self.scenario.interventions.items()
        )
        return tools

    def world_model(self) -> CausalWorldModel:
        world = CausalWorldModel()
        prior = 1.0 / len(self.scenario.hypotheses)
        for hypothesis_id, statement in self.scenario.hypotheses.items():
            world.upsert_hypothesis(
                hypothesis_id,
                statement,
                probability=prior,
                rationale="Uniform HiddenWorld prior.",
            )
        return world

    def metrics(self, result) -> HiddenWorldMetrics:
        world = result.world_model
        hypotheses = world.hypotheses()
        selected = max(hypotheses, key=lambda item: item.probability)
        true_hypothesis = world.get_hypothesis(self.scenario.hidden_hypothesis)
        min_intervention_cost = min(
            (
                float(self.scenario.intervention_costs.get(name, 0.2))
                for name, target in self.scenario.interventions.items()
                if target == self.scenario.hidden_hypothesis
            ),
            default=0.2,
        )
        oracle_utility = 1.0 - min_intervention_cost
        actual_reward = 1.0 if self.last_intervention_success else 0.0
        actual_utility = actual_reward - self.total_cost
        regret = max(0.0, oracle_utility - actual_utility)
        brier = sum(
            (
                hypothesis.probability
                - (1.0 if hypothesis.hypothesis_id == self.scenario.hidden_hypothesis else 0.0)
            )
            ** 2
            for hypothesis in hypotheses
        )
        return HiddenWorldMetrics(
            scenario_id=self.scenario.scenario_id,
            hidden_hypothesis=self.scenario.hidden_hypothesis,
            selected_hypothesis=selected.hypothesis_id,
            selected_intervention=self.selected_intervention,
            success=self.last_intervention_success,
            probes=self.probes,
            interventions=self.interventions,
            decision_rounds=len(result.state.decisions),
            total_cost=self.total_cost,
            true_hypothesis_posterior=true_hypothesis.probability,
            identification_correct=selected.hypothesis_id == self.scenario.hidden_hypothesis,
            causal_regret=regret,
            brier_score=brier,
            seed=self.seed,
            outcome_mode=self.outcome_mode,
        )


class HiddenWorldReasoner:
    """Reference no-key policy for the benchmark."""

    def __init__(
        self,
        scenario: HiddenWorldScenario,
        confidence_threshold: float = 0.8,
        max_probes: int = 3,
    ) -> None:
        self.scenario = scenario
        self.confidence_threshold = confidence_threshold
        self.max_probes = max_probes

    def propose(self, state, world_model) -> Sequence[CandidateAction]:
        if state.observations:
            latest = state.observations[-1].result
            if isinstance(latest, dict) and "success" in latest:
                return [
                    CandidateAction(
                        kind=ActionKind.STOP,
                        name="stop",
                        arguments={
                            "answer": "HiddenWorld intervention succeeded."
                            if latest["success"]
                            else "HiddenWorld intervention failed."
                        },
                        rationale="Intervention outcome observed.",
                    )
                ]

        active = world_model.hypotheses(include_rejected=False)
        leader = max(active, key=lambda item: item.probability)
        probe_count = sum(
            1
            for observation in state.observations
            if observation.action_name in self.scenario.experiments
        )

        if leader.probability >= self.confidence_threshold or probe_count >= self.max_probes:
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
                    rationale=f"Act on leading hypothesis {leader.hypothesis_id}.",
                )
            ]

        hypothesis_ids = list(self.scenario.hypotheses)
        return [
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name=name,
                expected_information_gain=0.0,
                tests_hypotheses=hypothesis_ids,
                rationale="Let runtime Bayesian EIG choose among diagnostic experiments.",
            )
            for name in self.scenario.experiments
        ]

    def uncertainty(self, state, world_model) -> str:
        active = sorted(
            world_model.hypotheses(include_rejected=False),
            key=lambda item: item.probability,
            reverse=True,
        )
        return " vs ".join(
            f"{item.hypothesis_id}={item.probability:.3f}" for item in active
        )


def build_hvac_hidden_world(hidden_hypothesis: str = "H2") -> HiddenWorldScenario:
    hypotheses = {
        "H1": "The HVAC filter is clogged.",
        "H2": "The supply fan is underperforming.",
        "H3": "The supply duct is obstructed.",
    }
    experiments = {
        "measure_filter_pressure": ExperimentContract(
            experiment_id="filter_pressure_test",
            description="Measure filter pressure drop.",
            outcomes=[
                OutcomeLikelihood("high", {"H1": 0.90, "H2": 0.15, "H3": 0.15}),
                OutcomeLikelihood("normal", {"H1": 0.10, "H2": 0.85, "H3": 0.85}),
            ],
        ),
        "measure_fan_rpm": ExperimentContract(
            experiment_id="fan_rpm_test",
            description="Measure fan RPM under load.",
            outcomes=[
                OutcomeLikelihood("low", {"H1": 0.15, "H2": 0.90, "H3": 0.20}),
                OutcomeLikelihood("normal", {"H1": 0.85, "H2": 0.10, "H3": 0.80}),
            ],
        ),
        "measure_duct_pressure": ExperimentContract(
            experiment_id="duct_pressure_test",
            description="Measure downstream duct static pressure.",
            outcomes=[
                OutcomeLikelihood("high", {"H1": 0.20, "H2": 0.20, "H3": 0.90}),
                OutcomeLikelihood("normal", {"H1": 0.80, "H2": 0.80, "H3": 0.10}),
            ],
        ),
    }
    return HiddenWorldScenario(
        scenario_id="hvac_three_faults_v1",
        hypotheses=hypotheses,
        hidden_hypothesis=hidden_hypothesis,
        experiments=experiments,
        interventions={
            "replace_filter": "H1",
            "repair_fan": "H2",
            "clear_duct": "H3",
        },
        experiment_costs={
            "measure_filter_pressure": 0.04,
            "measure_fan_rpm": 0.06,
            "measure_duct_pressure": 0.05,
        },
        intervention_costs={
            "replace_filter": 0.20,
            "repair_fan": 0.25,
            "clear_duct": 0.30,
        },
    )


def run_hidden_world(
    scenario: Optional[HiddenWorldScenario] = None,
    confidence_threshold: float = 0.8,
    max_probes: int = 3,
    max_steps: int = 6,
    outcome_mode: str = "deterministic",
    seed: Optional[int] = None,
) -> Tuple[HiddenWorldMetrics, object]:
    scenario = scenario or build_hvac_hidden_world()
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=outcome_mode,
        seed=seed,
    )
    world = environment.world_model()
    reasoner = HiddenWorldReasoner(
        scenario,
        confidence_threshold=confidence_threshold,
        max_probes=max_probes,
    )
    agent = create_agent(
        world_model=world,
        reasoner=reasoner,
        tools=environment.tools(),
    )
    result = agent.run(
        "Identify the hidden causal mechanism and apply the successful intervention.",
        max_steps=max_steps,
    )
    return environment.metrics(result), result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_hidden_world_suite(
    seeds: Iterable[int] = range(20),
    hidden_hypotheses: Sequence[str] = ("H1", "H2", "H3"),
    confidence_threshold: float = 0.8,
    max_probes: int = 3,
    max_steps: int = 6,
) -> Tuple[HiddenWorldSuiteReport, List[HiddenWorldMetrics]]:
    episode_metrics: List[HiddenWorldMetrics] = []
    for hidden in hidden_hypotheses:
        for seed in seeds:
            metrics, _result = run_hidden_world(
                build_hvac_hidden_world(hidden),
                confidence_threshold=confidence_threshold,
                max_probes=max_probes,
                max_steps=max_steps,
                outcome_mode="stochastic",
                seed=int(seed),
            )
            episode_metrics.append(metrics)

    per_hidden: Dict[str, Dict[str, float]] = {}
    for hidden in hidden_hypotheses:
        rows = [row for row in episode_metrics if row.hidden_hypothesis == hidden]
        per_hidden[hidden] = {
            "episodes": float(len(rows)),
            "success_rate": _mean(1.0 if row.success else 0.0 for row in rows),
            "identification_accuracy": _mean(
                1.0 if row.identification_correct else 0.0 for row in rows
            ),
            "mean_true_hypothesis_posterior": _mean(
                row.true_hypothesis_posterior for row in rows
            ),
            "mean_brier_score": _mean(row.brier_score for row in rows),
            "mean_probes": _mean(float(row.probes) for row in rows),
            "mean_causal_regret": _mean(row.causal_regret for row in rows),
        }

    report = HiddenWorldSuiteReport(
        episodes=len(episode_metrics),
        success_rate=_mean(1.0 if row.success else 0.0 for row in episode_metrics),
        identification_accuracy=_mean(
            1.0 if row.identification_correct else 0.0 for row in episode_metrics
        ),
        mean_true_hypothesis_posterior=_mean(
            row.true_hypothesis_posterior for row in episode_metrics
        ),
        mean_brier_score=_mean(row.brier_score for row in episode_metrics),
        mean_probes=_mean(float(row.probes) for row in episode_metrics),
        mean_total_cost=_mean(row.total_cost for row in episode_metrics),
        mean_causal_regret=_mean(row.causal_regret for row in episode_metrics),
        per_hidden=per_hidden,
    )
    return report, episode_metrics
