from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from causalrag.world_model.models import CausalWorldModel

from .models import ExperimentContract, posterior_for_outcome


@dataclass(frozen=True)
class InterventionContract:
    """Runtime-owned utility of an intervention under each causal hypothesis.

    Utilities are domain values before capability acquisition cost. ToolSpec.cost
    is subtracted separately so the same intervention semantics can be priced
    differently by deployment.
    """

    intervention_id: str
    utilities: Mapping[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.intervention_id).strip():
            raise ValueError("intervention_id must be non-empty")
        if len(self.utilities) < 2:
            raise ValueError("an intervention contract requires at least two hypotheses")
        if any(not str(hypothesis_id).strip() for hypothesis_id in self.utilities):
            raise ValueError("hypothesis ids must be non-empty")

    def hypothesis_ids(self):
        return [str(value) for value in self.utilities]

    def utility(self, hypothesis_id: str) -> float:
        return float(self.utilities[hypothesis_id])

    def summary(self) -> Dict[str, object]:
        # Proposer sees the domain of applicability, not hidden world truth.
        return {
            "intervention_id": self.intervention_id,
            "description": self.description,
            "hypotheses": self.hypothesis_ids(),
        }


@dataclass(frozen=True)
class InterventionValue:
    action_name: str
    expected_outcome_utility: float
    capability_cost: float
    net_value: float


@dataclass(frozen=True)
class ExperimentDecisionValue:
    action_name: str
    best_action_now: str
    best_value_now: float
    expected_best_value_after: float
    experiment_cost: float
    evsi: float
    net_value_of_sampling: float


def _normalized_distribution(
    world_model: CausalWorldModel,
    hypothesis_ids: Iterable[str],
) -> Dict[str, float]:
    ids = [str(value) for value in hypothesis_ids]
    if not ids:
        return {}
    raw: Dict[str, float] = {}
    for hypothesis_id in ids:
        hypothesis = world_model.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            return {}
        raw[hypothesis_id] = (
            0.0
            if hypothesis.status == "rejected"
            else max(0.0, float(hypothesis.probability))
        )
    total = sum(raw.values())
    if total <= 0.0:
        uniform = 1.0 / len(ids)
        return {hypothesis_id: uniform for hypothesis_id in ids}
    return {hypothesis_id: value / total for hypothesis_id, value in raw.items()}


def intervention_value(
    action_name: str,
    contract: InterventionContract,
    world_model: CausalWorldModel,
    capability_cost: float = 0.0,
) -> Optional[InterventionValue]:
    prior = _normalized_distribution(world_model, contract.hypothesis_ids())
    if not prior:
        return None
    expected = sum(
        prior[hypothesis_id] * contract.utility(hypothesis_id)
        for hypothesis_id in prior
    )
    cost = float(capability_cost)
    return InterventionValue(
        action_name=action_name,
        expected_outcome_utility=expected,
        capability_cost=cost,
        net_value=expected - cost,
    )


def best_intervention_value(
    world_model: CausalWorldModel,
    tools,
) -> Optional[InterventionValue]:
    values = []
    for name, spec in tools.specs().items():
        contract = getattr(spec, "intervention_contract", None)
        if contract is None:
            continue
        value = intervention_value(
            name,
            contract,
            world_model,
            capability_cost=float(spec.cost),
        )
        if value is not None:
            values.append(value)
    return max(values, key=lambda item: item.net_value) if values else None


def _posterior_world(
    world_model: CausalWorldModel,
    posterior: Mapping[str, float],
) -> CausalWorldModel:
    projected = CausalWorldModel()
    for hypothesis in world_model.hypotheses():
        projected.upsert_hypothesis(
            hypothesis.hypothesis_id,
            hypothesis.statement,
            probability=float(posterior.get(hypothesis.hypothesis_id, hypothesis.probability)),
            rationale=hypothesis.rationale,
            falsifiers=hypothesis.falsifiers,
        )
    return projected


def experiment_decision_value(
    action_name: str,
    experiment: ExperimentContract,
    world_model: CausalWorldModel,
    tools,
    experiment_cost: float = 0.0,
) -> Optional[ExperimentDecisionValue]:
    best_now = best_intervention_value(world_model, tools)
    if best_now is None:
        return None
    prior = _normalized_distribution(world_model, experiment.hypothesis_ids())
    if not prior:
        return None

    expected_after = 0.0
    total_outcome_probability = 0.0
    for outcome in experiment.outcomes:
        outcome_probability = sum(
            prior[hypothesis_id] * float(outcome.likelihoods[hypothesis_id])
            for hypothesis_id in prior
        )
        if outcome_probability <= 0.0:
            continue
        posterior = posterior_for_outcome(experiment, world_model, outcome.outcome)
        projected = _posterior_world(world_model, posterior)
        best_after = best_intervention_value(projected, tools)
        if best_after is None:
            return None
        expected_after += outcome_probability * best_after.net_value
        total_outcome_probability += outcome_probability

    if total_outcome_probability <= 0.0:
        return None
    expected_after /= total_outcome_probability
    evsi = expected_after - best_now.net_value
    cost = float(experiment_cost)
    return ExperimentDecisionValue(
        action_name=action_name,
        best_action_now=best_now.action_name,
        best_value_now=best_now.net_value,
        expected_best_value_after=expected_after,
        experiment_cost=cost,
        evsi=evsi,
        net_value_of_sampling=evsi - cost,
    )
