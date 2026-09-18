from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from causalrag.benchmarks.hidden_world import HiddenWorldScenario
from causalrag.experiments import ExperimentContract, OutcomeLikelihood


SCENARIO_CONTRACT_VERSION = "causalrag.probe.scenario.v1"
_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")


def _safe_id(value: Any, field_name: str) -> str:
    value = str(value or "").strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_SAFE_ID.pattern!r}; got {value!r}"
        )
    return value


def _bounded_number(
    value: Any,
    field_name: str,
    *,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    number = float(value)
    if number < minimum or number > maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}; got {number}"
        )
    return number


@dataclass(frozen=True)
class DeclarativeExperiment:
    name: str
    description: str
    contract: ExperimentContract
    cost: float = 0.05
    risk: float = 0.0


@dataclass(frozen=True)
class DeclarativeIntervention:
    name: str
    target_hypothesis: str
    description: str
    cost: float = 0.2
    risk: float = 0.0
    reversible: bool = True


@dataclass(frozen=True)
class ProbeScenarioContract:
    """Safe declarative hidden-mechanism world for the Playable Probe.

    Contracts describe a finite hypothesis frame, discrete experiments, and
    interventions. They cannot contain Python handlers, imports, URLs, shell
    commands, or arbitrary executable code. The runtime generates tools and the
    reference policy from this validated data.
    """

    scenario_id: str
    label: str
    description: str
    goal: str
    hypotheses: Mapping[str, str]
    experiments: Sequence[DeclarativeExperiment]
    interventions: Sequence[DeclarativeIntervention]
    failure_penalty: float = 1.0
    schema_version: str = SCENARIO_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCENARIO_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported scenario contract version {self.schema_version!r}; "
                f"expected {SCENARIO_CONTRACT_VERSION!r}"
            )
        _safe_id(self.scenario_id, "scenario_id")
        if not str(self.label).strip():
            raise ValueError("label must be non-empty")
        if not str(self.goal).strip():
            raise ValueError("goal must be non-empty")

        hypothesis_ids = list(self.hypotheses)
        if not 2 <= len(hypothesis_ids) <= 12:
            raise ValueError("scenario contracts require 2-12 hypotheses")
        for hypothesis_id, statement in self.hypotheses.items():
            _safe_id(hypothesis_id, "hypothesis id")
            if not str(statement).strip():
                raise ValueError(f"hypothesis {hypothesis_id} statement must be non-empty")

        if not 1 <= len(self.experiments) <= 32:
            raise ValueError("scenario contracts require 1-32 experiments")
        experiment_names = [item.name for item in self.experiments]
        if len(experiment_names) != len(set(experiment_names)):
            raise ValueError("experiment tool names must be unique")
        for experiment in self.experiments:
            _safe_id(experiment.name, "experiment name")
            if set(experiment.contract.hypothesis_ids()) != set(hypothesis_ids):
                raise ValueError(
                    f"experiment {experiment.name} must model exactly the declared hypotheses"
                )

        if not 1 <= len(self.interventions) <= 32:
            raise ValueError("scenario contracts require 1-32 interventions")
        intervention_names = [item.name for item in self.interventions]
        if len(intervention_names) != len(set(intervention_names)):
            raise ValueError("intervention tool names must be unique")
        if set(experiment_names).intersection(intervention_names):
            raise ValueError("experiment and intervention tool names must not overlap")
        targets = set()
        for intervention in self.interventions:
            _safe_id(intervention.name, "intervention name")
            if intervention.target_hypothesis not in self.hypotheses:
                raise ValueError(
                    f"intervention {intervention.name} targets unknown hypothesis "
                    f"{intervention.target_hypothesis}"
                )
            targets.add(intervention.target_hypothesis)
        missing_targets = set(hypothesis_ids) - targets
        if missing_targets:
            raise ValueError(
                "every hypothesis requires at least one intervention; missing "
                + ", ".join(sorted(missing_targets))
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProbeScenarioContract":
        if not isinstance(raw, Mapping):
            raise TypeError("scenario_contract must be a JSON object")
        schema_version = str(raw.get("schema_version") or SCENARIO_CONTRACT_VERSION)
        scenario_id = _safe_id(raw.get("id") or raw.get("scenario_id"), "scenario_id")
        label = str(raw.get("label") or scenario_id).strip()
        description = str(raw.get("description") or "").strip()
        goal = str(raw.get("goal") or "").strip()

        hypothesis_rows = raw.get("hypotheses")
        if not isinstance(hypothesis_rows, list):
            raise ValueError("hypotheses must be a JSON array")
        hypotheses: Dict[str, str] = {}
        for row in hypothesis_rows:
            if not isinstance(row, Mapping):
                raise ValueError("every hypothesis must be a JSON object")
            hypothesis_id = _safe_id(row.get("id"), "hypothesis id")
            if hypothesis_id in hypotheses:
                raise ValueError(f"duplicate hypothesis id: {hypothesis_id}")
            hypotheses[hypothesis_id] = str(row.get("statement") or "").strip()

        experiment_rows = raw.get("experiments")
        if not isinstance(experiment_rows, list):
            raise ValueError("experiments must be a JSON array")
        experiments = []
        for row in experiment_rows:
            if not isinstance(row, Mapping):
                raise ValueError("every experiment must be a JSON object")
            name = _safe_id(row.get("name"), "experiment name")
            outcome_rows = row.get("outcomes")
            if not isinstance(outcome_rows, list):
                raise ValueError(f"experiment {name} outcomes must be a JSON array")
            if len(outcome_rows) > 20:
                raise ValueError(f"experiment {name} may have at most 20 outcomes")
            outcomes = []
            for outcome_row in outcome_rows:
                if not isinstance(outcome_row, Mapping):
                    raise ValueError(
                        f"experiment {name} outcome entries must be JSON objects"
                    )
                label_value = str(
                    outcome_row.get("label") or outcome_row.get("outcome") or ""
                ).strip()
                likelihoods = outcome_row.get("likelihoods")
                if not isinstance(likelihoods, Mapping):
                    raise ValueError(
                        f"experiment {name} outcome {label_value!r} needs likelihoods"
                    )
                outcomes.append(
                    OutcomeLikelihood(
                        outcome=label_value,
                        likelihoods={
                            str(key): float(value)
                            for key, value in likelihoods.items()
                        },
                    )
                )
            contract = ExperimentContract(
                experiment_id=str(row.get("experiment_id") or name),
                description=str(row.get("description") or "").strip(),
                outcome_key=str(row.get("outcome_key") or "outcome").strip(),
                outcomes=outcomes,
            )
            experiments.append(
                DeclarativeExperiment(
                    name=name,
                    description=str(row.get("description") or "").strip(),
                    contract=contract,
                    cost=_bounded_number(row.get("cost", 0.05), f"{name}.cost"),
                    risk=_bounded_number(row.get("risk", 0.0), f"{name}.risk"),
                )
            )

        intervention_rows = raw.get("interventions")
        if not isinstance(intervention_rows, list):
            raise ValueError("interventions must be a JSON array")
        interventions = []
        for row in intervention_rows:
            if not isinstance(row, Mapping):
                raise ValueError("every intervention must be a JSON object")
            name = _safe_id(row.get("name"), "intervention name")
            interventions.append(
                DeclarativeIntervention(
                    name=name,
                    target_hypothesis=_safe_id(
                        row.get("target") or row.get("target_hypothesis"),
                        f"{name}.target",
                    ),
                    description=str(row.get("description") or "").strip(),
                    cost=_bounded_number(row.get("cost", 0.2), f"{name}.cost"),
                    risk=_bounded_number(row.get("risk", 0.0), f"{name}.risk"),
                    reversible=bool(row.get("reversible", True)),
                )
            )

        return cls(
            scenario_id=scenario_id,
            label=label,
            description=description,
            goal=goal,
            hypotheses=hypotheses,
            experiments=tuple(experiments),
            interventions=tuple(interventions),
            failure_penalty=_bounded_number(
                raw.get("failure_penalty", 1.0),
                "failure_penalty",
                maximum=1000.0,
            ),
            schema_version=schema_version,
        )

    def validate_run(self, hidden_hypothesis: str, outcome_mode: str) -> None:
        if hidden_hypothesis not in self.hypotheses:
            raise ValueError(
                f"hidden_hypothesis must be one of {list(self.hypotheses)}"
            )
        if outcome_mode not in {"deterministic", "stochastic"}:
            raise ValueError("declarative scenarios support deterministic or stochastic outcomes")

    def to_hidden_world(self, hidden_hypothesis: str) -> HiddenWorldScenario:
        self.validate_run(hidden_hypothesis, "deterministic")
        return HiddenWorldScenario(
            scenario_id=self.scenario_id,
            hypotheses=dict(self.hypotheses),
            hidden_hypothesis=hidden_hypothesis,
            experiments={item.name: item.contract for item in self.experiments},
            interventions={
                item.name: item.target_hypothesis
                for item in self.interventions
            },
            experiment_costs={item.name: item.cost for item in self.experiments},
            experiment_risks={item.name: item.risk for item in self.experiments},
            intervention_costs={item.name: item.cost for item in self.interventions},
            intervention_risks={item.name: item.risk for item in self.interventions},
            intervention_reversible={
                item.name: item.reversible for item in self.interventions
            },
            failure_penalty=self.failure_penalty,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.scenario_id,
            "label": self.label,
            "description": self.description,
            "goal": self.goal,
            "hidden_hypotheses": list(self.hypotheses),
            "outcome_modes": ["deterministic", "stochastic"],
            "recommended_test": "Declarative Bayesian experiment / intervention loop",
            "default_hidden_hypothesis": next(iter(self.hypotheses)),
            "default_outcome_mode": "deterministic",
            "declarative": True,
            "experiment_count": len(self.experiments),
            "intervention_count": len(self.interventions),
        }

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.scenario_id,
            "label": self.label,
            "description": self.description,
            "goal": self.goal,
            "hypotheses": [
                {"id": hypothesis_id, "statement": statement}
                for hypothesis_id, statement in self.hypotheses.items()
            ],
            "experiments": [
                {
                    "name": item.name,
                    "experiment_id": item.contract.experiment_id,
                    "description": item.description,
                    "outcome_key": item.contract.outcome_key,
                    "cost": item.cost,
                    "risk": item.risk,
                    "outcomes": [
                        {
                            "label": outcome.outcome,
                            "likelihoods": dict(outcome.likelihoods),
                        }
                        for outcome in item.contract.outcomes
                    ],
                }
                for item in self.experiments
            ],
            "interventions": [
                {
                    "name": item.name,
                    "target": item.target_hypothesis,
                    "description": item.description,
                    "cost": item.cost,
                    "risk": item.risk,
                    "reversible": item.reversible,
                }
                for item in self.interventions
            ],
            "failure_penalty": self.failure_penalty,
        }


def example_scenario_contract() -> Dict[str, Any]:
    """Small non-HVAC example used by API/UI docs and tests."""

    return {
        "schema_version": SCENARIO_CONTRACT_VERSION,
        "id": "pump_diagnosis_demo",
        "label": "Pump diagnosis demo",
        "description": "Distinguish inlet blockage from motor degradation.",
        "goal": "Identify the pump fault with as few measurements as possible, then apply the matching repair.",
        "hypotheses": [
            {"id": "H1", "statement": "The pump inlet is blocked."},
            {"id": "H2", "statement": "The pump motor is degraded."},
        ],
        "experiments": [
            {
                "name": "measure_inlet_pressure",
                "description": "Measure inlet pressure under load.",
                "cost": 0.04,
                "risk": 0.0,
                "outcomes": [
                    {"label": "low", "likelihoods": {"H1": 0.9, "H2": 0.2}},
                    {"label": "normal", "likelihoods": {"H1": 0.1, "H2": 0.8}},
                ],
            },
            {
                "name": "measure_motor_current",
                "description": "Measure motor current under load.",
                "cost": 0.05,
                "risk": 0.0,
                "outcomes": [
                    {"label": "high", "likelihoods": {"H1": 0.2, "H2": 0.85}},
                    {"label": "normal", "likelihoods": {"H1": 0.8, "H2": 0.15}},
                ],
            },
        ],
        "interventions": [
            {
                "name": "clear_inlet",
                "target": "H1",
                "description": "Clear the inlet obstruction.",
                "cost": 0.2,
                "risk": 0.05,
                "reversible": True,
            },
            {
                "name": "repair_motor",
                "target": "H2",
                "description": "Repair the degraded pump motor.",
                "cost": 0.3,
                "risk": 0.1,
                "reversible": True,
            },
        ],
    }
