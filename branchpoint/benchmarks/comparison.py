from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from branchpoint import create_agent
from branchpoint.experiments import InterventionContract

from .hidden_world import (
    HiddenWorldEnvironment,
    HiddenWorldMetrics,
    HiddenWorldScenario,
    build_hvac_hidden_world,
)
from .policies import (
    CheapestProbePolicy,
    ConservativeEIGPolicy,
    DecisionValuePolicy,
    GreedyEIGPolicy,
    RandomProbePolicy,
    RiskSensitiveDecisionValuePolicy,
)


@dataclass
class PolicyReport:
    policy_id: str
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
            "policy_id": self.policy_id,
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


@dataclass
class PolicyComparisonReport:
    seeds: List[int]
    hidden_hypotheses: List[str]
    reports: Dict[str, PolicyReport]

    def to_dict(self) -> Dict[str, object]:
        return {
            "seeds": self.seeds,
            "hidden_hypotheses": self.hidden_hypotheses,
            "policies": {
                policy_id: report.to_dict()
                for policy_id, report in self.reports.items()
            },
        }


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _attach_intervention_contracts(
    tools,
    scenario: HiddenWorldScenario,
    wrong_intervention_utility: float = 0.0,
) -> None:
    """Attach explicit intervention outcome utilities to runtime capabilities."""
    hypothesis_ids = list(scenario.hypotheses)
    by_name = {tool.name: tool for tool in tools}
    wrong_utility = float(wrong_intervention_utility)
    for action_name, target in scenario.interventions.items():
        tool = by_name[action_name]
        tool.intervention_contract = InterventionContract(
            intervention_id=action_name,
            description=(
                f"Successful when {target} is the true hidden mechanism; "
                f"wrong-intervention utility={wrong_utility:.3f}."
            ),
            utilities={
                hypothesis_id: 1.0 if hypothesis_id == target else wrong_utility
                for hypothesis_id in hypothesis_ids
            },
        )


def run_policy_episode(
    policy_type,
    scenario: HiddenWorldScenario,
    seed: int,
    max_steps: int = 7,
) -> Tuple[HiddenWorldMetrics, object]:
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode="stochastic",
        seed=seed,
    )
    world = environment.world_model()
    policy = policy_type(scenario, seed=100_000 + int(seed))
    tools = environment.tools()
    if getattr(policy_type, "requires_intervention_contracts", False):
        _attach_intervention_contracts(
            tools,
            scenario,
            wrong_intervention_utility=float(
                getattr(policy_type, "wrong_intervention_utility", 0.0)
            ),
        )
    agent = create_agent(
        world_model=world,
        reasoner=policy,
        tools=tools,
    )
    result = agent.run(
        "Identify the hidden causal mechanism and apply the successful intervention.",
        max_steps=max_steps,
    )
    return environment.metrics(result), result


def _summarize(
    policy_id: str,
    metrics: Sequence[HiddenWorldMetrics],
    hidden_hypotheses: Sequence[str],
) -> PolicyReport:
    per_hidden: Dict[str, Dict[str, float]] = {}
    for hidden in hidden_hypotheses:
        rows = [row for row in metrics if row.hidden_hypothesis == hidden]
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
            "mean_total_cost": _mean(row.total_cost for row in rows),
            "mean_causal_regret": _mean(row.causal_regret for row in rows),
        }
    return PolicyReport(
        policy_id=policy_id,
        episodes=len(metrics),
        success_rate=_mean(1.0 if row.success else 0.0 for row in metrics),
        identification_accuracy=_mean(
            1.0 if row.identification_correct else 0.0 for row in metrics
        ),
        mean_true_hypothesis_posterior=_mean(
            row.true_hypothesis_posterior for row in metrics
        ),
        mean_brier_score=_mean(row.brier_score for row in metrics),
        mean_probes=_mean(float(row.probes) for row in metrics),
        mean_total_cost=_mean(row.total_cost for row in metrics),
        mean_causal_regret=_mean(row.causal_regret for row in metrics),
        per_hidden=per_hidden,
    )


def compare_hidden_world_policies(
    seeds: Iterable[int] = range(20),
    hidden_hypotheses: Sequence[str] = ("H1", "H2", "H3"),
    policy_types: Sequence[type] = (
        GreedyEIGPolicy,
        ConservativeEIGPolicy,
        DecisionValuePolicy,
        CheapestProbePolicy,
        RandomProbePolicy,
    ),
    max_steps: int = 7,
) -> Tuple[PolicyComparisonReport, Dict[str, List[HiddenWorldMetrics]]]:
    seeds = [int(seed) for seed in seeds]
    hidden_hypotheses = list(hidden_hypotheses)
    all_metrics: Dict[str, List[HiddenWorldMetrics]] = {}
    reports: Dict[str, PolicyReport] = {}

    for policy_type in policy_types:
        policy_id = str(policy_type.policy_id)
        rows: List[HiddenWorldMetrics] = []
        for hidden in hidden_hypotheses:
            for seed in seeds:
                metrics, _result = run_policy_episode(
                    policy_type,
                    build_hvac_hidden_world(hidden),
                    seed=seed,
                    max_steps=max_steps,
                )
                rows.append(metrics)
        all_metrics[policy_id] = rows
        reports[policy_id] = _summarize(policy_id, rows, hidden_hypotheses)

    return (
        PolicyComparisonReport(
            seeds=seeds,
            hidden_hypotheses=hidden_hypotheses,
            reports=reports,
        ),
        all_metrics,
    )


def _risk_policy_type(wrong_action_loss: float):
    loss = max(0.0, float(wrong_action_loss))
    label = str(loss).replace(".", "p")
    return type(
        f"RiskSensitiveDecisionValueLoss{label}",
        (RiskSensitiveDecisionValuePolicy,),
        {
            "policy_id": f"decision_value_wrong_loss_{label}",
            "wrong_intervention_utility": -loss,
        },
    )


def compare_risk_sensitivity(
    wrong_action_losses: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
    seeds: Iterable[int] = range(20),
    hidden_hypotheses: Sequence[str] = ("H1", "H2", "H3"),
    max_steps: int = 7,
) -> Tuple[PolicyComparisonReport, Dict[str, List[HiddenWorldMetrics]]]:
    """Measure the policy frontier as wrong-action loss increases.

    The environment, experiment likelihoods, capability prices, and seeds stay
    fixed. Only the intervention utility assigned to a wrong action changes.
    This makes phase changes in observe-vs-intervene behavior directly visible
    instead of hiding a magic risk constant in the policy implementation.
    """
    policies = tuple(_risk_policy_type(loss) for loss in wrong_action_losses)
    return compare_hidden_world_policies(
        seeds=seeds,
        hidden_hypotheses=hidden_hypotheses,
        policy_types=policies,
        max_steps=max_steps,
    )
