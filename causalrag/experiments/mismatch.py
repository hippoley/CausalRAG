from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from causalrag.world_model.models import CausalWorldModel, ModelMismatch

from .models import ExperimentContract, outcome_surprisal, predictive_probability


@dataclass(frozen=True)
class ModelMismatchPolicy:
    """Runtime policy for deciding when the modeled explanation set is inadequate.

    A single merely-unlikely observation is not enough to declare the world
    model broken. Soft surprises must appear across multiple distinct
    experiments, while a near-impossible outcome can trigger immediately.
    """

    soft_predictive_threshold: float = 0.05
    hard_predictive_threshold: float = 0.005
    min_distinct_experiments: int = 2
    recent_window: int = 4
    discovered_initial_probability: float = 0.2

    def __post_init__(self) -> None:
        soft = float(self.soft_predictive_threshold)
        hard = float(self.hard_predictive_threshold)
        if soft <= 0.0 or soft >= 1.0:
            raise ValueError("soft_predictive_threshold must be between 0 and 1")
        if hard <= 0.0 or hard > soft:
            raise ValueError("hard_predictive_threshold must be positive and <= soft threshold")
        if int(self.min_distinct_experiments) < 1:
            raise ValueError("min_distinct_experiments must be >= 1")
        if int(self.recent_window) < 1:
            raise ValueError("recent_window must be >= 1")
        if not 0.0 < float(self.discovered_initial_probability) < 0.5:
            raise ValueError("discovered_initial_probability must be between 0 and 0.5")


@dataclass(frozen=True)
class ModelMismatchAssessment:
    experiment_id: str
    outcome: str
    predictive_probability: float
    surprisal: float
    suspicious: bool
    hard_mismatch: bool
    escalate: bool
    suppress_closed_world_posterior: bool
    mismatch_id: Optional[str] = None


def assess_model_mismatch(
    contract: ExperimentContract,
    world_model: CausalWorldModel,
    observation: Any,
    policy: Optional[ModelMismatchPolicy] = None,
    metadata: Optional[dict] = None,
) -> Optional[ModelMismatchAssessment]:
    """Assess whether an outcome is evidence for ``none of the above``.

    Suspicious observations are retained as warning evidence. The first soft
    surprise does not activate ``none_of_the_above``; escalation requires either
    a near-impossible outcome or corroborating surprises across distinct
    experiments. This check happens before a closed-world posterior is
    committed so escalation can preserve uncertainty instead of forcing mass
    onto an inadequate model class.
    """
    policy = policy or ModelMismatchPolicy()
    outcome = contract.resolve_outcome(observation)
    if outcome is None:
        return None

    predictive = predictive_probability(contract, world_model, outcome)
    surprisal = outcome_surprisal(contract, world_model, outcome)
    suspicious = predictive <= float(policy.soft_predictive_threshold)
    hard = predictive <= float(policy.hard_predictive_threshold)

    mismatch_id = None
    if suspicious:
        mismatch_id = f"mismatch-{len(world_model.model_mismatches) + 1}"
        world_model.record_model_mismatch(
            ModelMismatch(
                mismatch_id=mismatch_id,
                experiment_id=contract.experiment_id,
                outcome=outcome,
                predictive_probability=predictive,
                surprisal=surprisal,
                severity="hard" if hard else "soft",
                metadata=dict(metadata or {}),
            )
        )

    recent = world_model.unresolved_model_mismatches()[-int(policy.recent_window) :]
    distinct_experiments = {item.experiment_id for item in recent}
    escalate = bool(
        hard
        or (
            suspicious
            and len(distinct_experiments) >= int(policy.min_distinct_experiments)
        )
    )
    if escalate:
        world_model.activate_model_mismatch()

    return ModelMismatchAssessment(
        experiment_id=contract.experiment_id,
        outcome=outcome,
        predictive_probability=predictive,
        surprisal=surprisal,
        suspicious=suspicious,
        hard_mismatch=hard,
        escalate=escalate,
        suppress_closed_world_posterior=escalate,
        mismatch_id=mismatch_id,
    )


def maybe_resolve_model_mismatch(world_model: CausalWorldModel) -> bool:
    """Resolve open-world mismatch once a discovered hypothesis is validated."""
    resolved = any(
        hypothesis.origin == "discovered"
        and hypothesis.validated
        and hypothesis.status != "rejected"
        for hypothesis in world_model.hypotheses()
    )
    if resolved and world_model.model_mismatch_active:
        world_model.resolve_model_mismatch()
    return resolved
