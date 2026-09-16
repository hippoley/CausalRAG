from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from causalrag.experiments import ExperimentContract


class TimeDriver(Protocol):
    """Adapter for the environment clock used by WAIT."""

    @property
    def now_seconds(self) -> float:
        ...

    def advance(self, seconds: float) -> float:
        ...


@dataclass
class VirtualTimeDriver:
    now_seconds: float = 0.0

    def advance(self, seconds: float) -> float:
        delta = max(0.0, float(seconds))
        self.now_seconds += delta
        return delta


@dataclass(frozen=True)
class TemporalObservationPoint:
    """A time-indexed observation model after an intervention.

    `experiment_contract` is a real P(outcome | hypothesis, observe_at=t)
    model, so temporal scheduling reuses Bayesian EIG/EVSI rather than a
    separate heuristic confidence score.
    """

    offset_seconds: float
    experiment_contract: "ExperimentContract"
    measurement_cost: float = 0.0

    def __post_init__(self) -> None:
        if float(self.offset_seconds) < 0.0:
            raise ValueError("offset_seconds must be >= 0")
        if float(self.measurement_cost) < 0.0:
            raise ValueError("measurement_cost must be >= 0")


@dataclass(frozen=True)
class TemporalEffectContract:
    """Expected delayed observation after an intervention."""

    effect_id: str
    observe_with: str
    observation_key: str
    earliest_seconds: float
    latest_seconds: float
    expected_outcomes: Mapping[str, Any]
    falsification_weight: float = 0.5
    description: str = ""
    observe_arguments: Mapping[str, Any] = field(default_factory=dict)
    protect_attribution: bool = True
    observation_points: Tuple[TemporalObservationPoint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.effect_id).strip():
            raise ValueError("effect_id must be non-empty")
        if not str(self.observe_with).strip():
            raise ValueError("observe_with must be non-empty")
        if not str(self.observation_key).strip():
            raise ValueError("observation_key must be non-empty")
        if float(self.earliest_seconds) < 0.0:
            raise ValueError("earliest_seconds must be >= 0")
        if float(self.latest_seconds) < float(self.earliest_seconds):
            raise ValueError("latest_seconds must be >= earliest_seconds")
        if not self.expected_outcomes:
            raise ValueError("expected_outcomes must not be empty")
        if not 0.0 < float(self.falsification_weight) <= 1.0:
            raise ValueError("falsification_weight must be in (0, 1]")
        offsets = [float(point.offset_seconds) for point in self.observation_points]
        if len(offsets) != len(set(offsets)):
            raise ValueError("temporal observation offsets must be unique")
        for offset in offsets:
            if offset < float(self.earliest_seconds) or offset > float(self.latest_seconds):
                raise ValueError("temporal observation points must fall inside the effect window")

    def expected_for(self, hypothesis_id: str) -> Optional[Any]:
        return self.expected_outcomes.get(str(hypothesis_id))


@dataclass
class PendingEffect:
    effect_id: str
    intervention: str
    observe_with: str
    observation_key: str
    started_at: float
    ready_at: float
    expires_at: float
    expected_outcomes: Dict[str, Any]
    falsification_weight: float = 0.5
    observe_arguments: Dict[str, Any] = field(default_factory=dict)
    protect_attribution: bool = True
    observation_points: Tuple[TemporalObservationPoint, ...] = field(default_factory=tuple)
    observed: bool = False
    expired: bool = False
    expiry_recorded: bool = False
    matched_prediction: Optional[bool] = None
    observed_value: Any = None
    planned_observation_at: Optional[float] = None
    planned_experiment_contract: Optional[Any] = field(default=None, repr=False)
    planned_temporal_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready(self, now: float) -> bool:
        return not self.observed and not self.expired and float(now) >= self.ready_at

    def is_premature(self, now: float) -> bool:
        return not self.observed and not self.expired and float(now) < self.ready_at

    def refresh(self, now: float) -> None:
        if not self.observed and float(now) > self.expires_at:
            self.expired = True

    def seconds_until_ready(self, now: float) -> float:
        return max(0.0, self.ready_at - float(now))

    def seconds_until_expiry(self, now: float) -> float:
        return max(0.0, self.expires_at - float(now))

    def expected_for(self, hypothesis_id: str) -> Optional[Any]:
        return self.expected_outcomes.get(str(hypothesis_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "intervention": self.intervention,
            "observe_with": self.observe_with,
            "observation_key": self.observation_key,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "expires_at": self.expires_at,
            "observe_arguments": dict(self.observe_arguments),
            "protect_attribution": self.protect_attribution,
            "observation_points": [point.offset_seconds for point in self.observation_points],
            "planned_observation_at": self.planned_observation_at,
            "planned_temporal_value": self.planned_temporal_value,
            "observed": self.observed,
            "expired": self.expired,
            "matched_prediction": self.matched_prediction,
            "observed_value": self.observed_value,
        }


def pending_effect_from_contract(
    intervention: str,
    contract: TemporalEffectContract,
    now: float,
) -> PendingEffect:
    now = float(now)
    return PendingEffect(
        effect_id=contract.effect_id,
        intervention=str(intervention),
        observe_with=contract.observe_with,
        observation_key=contract.observation_key,
        started_at=now,
        ready_at=now + float(contract.earliest_seconds),
        expires_at=now + float(contract.latest_seconds),
        expected_outcomes=dict(contract.expected_outcomes),
        falsification_weight=float(contract.falsification_weight),
        observe_arguments=dict(contract.observe_arguments),
        protect_attribution=bool(contract.protect_attribution),
        observation_points=tuple(contract.observation_points),
        metadata={"description": contract.description},
    )
