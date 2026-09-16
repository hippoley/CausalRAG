from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class TemporalEffectContract:
    """Expected delayed observation after an intervention.

    The contract belongs to runtime/capability semantics. It states when an
    intervention can first be evaluated, when that observation becomes stale,
    which capability measures the effect, and which outcome is expected under
    each causal hypothesis.
    """

    effect_id: str
    observe_with: str
    observation_key: str
    earliest_seconds: float
    latest_seconds: float
    expected_outcomes: Mapping[str, Any]
    falsification_weight: float = 0.5
    description: str = ""

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
    observed: bool = False
    expired: bool = False
    matched_prediction: Optional[bool] = None
    observed_value: Any = None
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
        metadata={"description": contract.description},
    )
