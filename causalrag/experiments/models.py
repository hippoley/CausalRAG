from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from causalrag.world_model.models import CausalWorldModel, Evidence


@dataclass(frozen=True)
class OutcomeLikelihood:
    outcome: str
    likelihoods: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.outcome).strip():
            raise ValueError("outcome must be non-empty")
        if not self.likelihoods:
            raise ValueError("likelihoods must be non-empty")
        for hypothesis_id, value in self.likelihoods.items():
            if not str(hypothesis_id).strip():
                raise ValueError("hypothesis ids must be non-empty")
            probability = float(value)
            if probability < 0.0 or probability > 1.0:
                raise ValueError("outcome likelihoods must be between 0 and 1")


@dataclass(frozen=True)
class ExperimentContract:
    """Runtime-owned P(outcome | hypothesis, action) for discrete experiments."""

    experiment_id: str
    outcomes: List[OutcomeLikelihood]
    outcome_key: str = "outcome"
    description: str = ""
    tolerance: float = field(default=1e-6, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.experiment_id).strip():
            raise ValueError("experiment_id must be non-empty")
        if len(self.outcomes) < 2:
            raise ValueError("an experiment contract requires at least two outcomes")
        labels = [outcome.outcome for outcome in self.outcomes]
        if len(labels) != len(set(labels)):
            raise ValueError("experiment outcomes must have unique labels")
        hypothesis_ids = self.hypothesis_ids()
        if len(hypothesis_ids) < 2:
            raise ValueError("an experiment contract requires at least two hypotheses")
        for outcome in self.outcomes:
            if set(outcome.likelihoods) != set(hypothesis_ids):
                raise ValueError("every outcome must provide likelihoods for the same hypotheses")
        for hypothesis_id in hypothesis_ids:
            total = sum(float(outcome.likelihoods[hypothesis_id]) for outcome in self.outcomes)
            if abs(total - 1.0) > self.tolerance:
                raise ValueError(f"likelihoods for {hypothesis_id} must sum to 1; got {total}")

    def hypothesis_ids(self) -> List[str]:
        return [] if not self.outcomes else [str(value) for value in self.outcomes[0].likelihoods.keys()]

    def outcome_labels(self) -> List[str]:
        return [outcome.outcome for outcome in self.outcomes]

    def likelihood(self, outcome: str, hypothesis_id: str) -> float:
        for item in self.outcomes:
            if item.outcome == outcome:
                return float(item.likelihoods[hypothesis_id])
        raise KeyError(f"unknown outcome: {outcome}")

    def resolve_outcome(self, observation: Any) -> Optional[str]:
        if isinstance(observation, str):
            value = observation
        elif isinstance(observation, Mapping):
            value = observation.get(self.outcome_key)
        else:
            value = None
        if value is None:
            return None
        value = str(value)
        return value if value in self.outcome_labels() else None

    def summary(self) -> Dict[str, Any]:
        """Compact proposer-facing semantics; likelihoods stay runtime-owned."""
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypotheses": self.hypothesis_ids(),
            "outcomes": self.outcome_labels(),
            "outcome_key": self.outcome_key,
        }


@dataclass
class ExperimentUpdate:
    experiment_id: str
    outcome: str
    prior: Dict[str, float]
    posterior: Dict[str, float]
    expected_information_gain: float


def _entropy(distribution: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in distribution if p > 0.0)


def contract_applicable(contract: ExperimentContract, world_model: CausalWorldModel) -> bool:
    """A Bayesian contract is usable only when all modeled hypotheses exist."""
    return all(world_model.get_hypothesis(hypothesis_id) is not None for hypothesis_id in contract.hypothesis_ids())


def _normalized_prior(contract: ExperimentContract, world_model: CausalWorldModel) -> Dict[str, float]:
    if not contract_applicable(contract, world_model):
        return {}
    raw: Dict[str, float] = {}
    for hypothesis_id in contract.hypothesis_ids():
        hypothesis = world_model.get_hypothesis(hypothesis_id)
        raw[hypothesis_id] = 0.0 if hypothesis.status == "rejected" else max(0.0, float(hypothesis.probability))
    total = sum(raw.values())
    if total <= 0.0:
        uniform = 1.0 / len(raw)
        return {hypothesis_id: uniform for hypothesis_id in raw}
    return {hypothesis_id: value / total for hypothesis_id, value in raw.items()}


def posterior_for_outcome(contract: ExperimentContract, world_model: CausalWorldModel, outcome: str) -> Dict[str, float]:
    prior = _normalized_prior(contract, world_model)
    if not prior:
        return {}
    numerators = {hypothesis_id: prior_probability * contract.likelihood(outcome, hypothesis_id) for hypothesis_id, prior_probability in prior.items()}
    evidence_probability = sum(numerators.values())
    if evidence_probability <= 0.0:
        return prior
    return {hypothesis_id: numerator / evidence_probability for hypothesis_id, numerator in numerators.items()}


def expected_information_gain(contract: ExperimentContract, world_model: CausalWorldModel) -> float:
    prior = _normalized_prior(contract, world_model)
    if not prior:
        return 0.0
    prior_entropy = _entropy(prior.values())
    if prior_entropy <= 0.0:
        return 0.0
    expected_posterior_entropy = 0.0
    for outcome in contract.outcomes:
        outcome_probability = sum(prior[hypothesis_id] * float(outcome.likelihoods[hypothesis_id]) for hypothesis_id in prior)
        if outcome_probability <= 0.0:
            continue
        posterior = posterior_for_outcome(contract, world_model, outcome.outcome)
        expected_posterior_entropy += outcome_probability * _entropy(posterior.values())
    information = max(0.0, prior_entropy - expected_posterior_entropy)
    return max(0.0, min(1.0, information / prior_entropy))


def apply_experiment_observation(contract: ExperimentContract, world_model: CausalWorldModel, observation: Any, source: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[ExperimentUpdate]:
    if not contract_applicable(contract, world_model):
        return None
    outcome = contract.resolve_outcome(observation)
    if outcome is None:
        return None
    prior = _normalized_prior(contract, world_model)
    posterior = posterior_for_outcome(contract, world_model, outcome)
    information_gain = expected_information_gain(contract, world_model)
    for hypothesis_id, probability in posterior.items():
        prior_probability = prior[hypothesis_id]
        evidence = Evidence(
            source=source,
            statement=f"Experiment {contract.experiment_id} observed outcome '{outcome}': normalized prior {prior_probability:.4f} -> posterior {probability:.4f}",
            weight=probability - prior_probability,
            kind="bayesian_experiment",
            metadata={"experiment_id": contract.experiment_id, "outcome": outcome, **(metadata or {})},
        )
        world_model.set_hypothesis_probability(hypothesis_id, probability=probability, evidence=evidence)
    return ExperimentUpdate(experiment_id=contract.experiment_id, outcome=outcome, prior=prior, posterior=posterior, expected_information_gain=information_gain)
