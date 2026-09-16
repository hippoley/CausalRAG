from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from causalrag.agent.temporal import PendingEffect
from causalrag.world_model.models import CausalWorldModel

from .decision import experiment_decision_value
from .models import ExperimentContract


@dataclass(frozen=True)
class TemporalDecisionPreferences:
    """Deployment-owned opportunity cost of delaying a causal observation.

    Consequence utility remains in DecisionPreferences. This object only prices
    elapsed time, keeping outcome preferences and temporal/SLA preferences
    independent in the runtime control plane.
    """

    wait_cost_per_second: float = 0.0

    def __post_init__(self) -> None:
        if float(self.wait_cost_per_second) < 0.0:
            raise ValueError("wait_cost_per_second must be >= 0")


@dataclass(frozen=True)
class TemporalObservationValue:
    effect_id: str
    observe_with: str
    offset_seconds: float
    observation_at: float
    wait_seconds: float
    experiment_id: str
    evsi: float
    measurement_cost: float
    wait_cost: float
    net_value: float
    experiment_contract: ExperimentContract = field(repr=False, compare=False)


def temporal_observation_values(
    effect: PendingEffect,
    world_model: CausalWorldModel,
    tools,
    now: float,
    preferences: Optional[TemporalDecisionPreferences] = None,
) -> List[TemporalObservationValue]:
    """Value every still-reachable observation point for one pending effect.

    Each observation point owns a real ExperimentContract. Decision value is
    therefore the existing one-step EVSI minus measurement cost and deployment
    wait opportunity cost; no separate temporal confidence heuristic is used.
    """

    prefs = preferences or TemporalDecisionPreferences()
    now = float(now)
    values: List[TemporalObservationValue] = []

    for point in effect.observation_points:
        observation_at = effect.started_at + float(point.offset_seconds)
        if observation_at < now or observation_at > effect.expires_at:
            continue
        decision = experiment_decision_value(
            action_name=effect.observe_with,
            experiment=point.experiment_contract,
            world_model=world_model,
            tools=tools,
            experiment_cost=float(point.measurement_cost),
        )
        if decision is None:
            continue
        wait_seconds = max(0.0, observation_at - now)
        wait_cost = wait_seconds * float(prefs.wait_cost_per_second)
        values.append(
            TemporalObservationValue(
                effect_id=effect.effect_id,
                observe_with=effect.observe_with,
                offset_seconds=float(point.offset_seconds),
                observation_at=observation_at,
                wait_seconds=wait_seconds,
                experiment_id=point.experiment_contract.experiment_id,
                evsi=decision.evsi,
                measurement_cost=float(point.measurement_cost),
                wait_cost=wait_cost,
                net_value=decision.net_value_of_sampling - wait_cost,
                experiment_contract=point.experiment_contract,
            )
        )
    return values


def best_temporal_observation_value(
    effect: PendingEffect,
    world_model: CausalWorldModel,
    tools,
    now: float,
    preferences: Optional[TemporalDecisionPreferences] = None,
) -> Optional[TemporalObservationValue]:
    values = temporal_observation_values(
        effect=effect,
        world_model=world_model,
        tools=tools,
        now=now,
        preferences=preferences,
    )
    if not values:
        return None
    # Prefer greater decision value; on an exact tie, observe earlier.
    return max(values, key=lambda value: (value.net_value, -value.observation_at))
