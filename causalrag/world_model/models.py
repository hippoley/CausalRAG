from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Evidence:
    """A single piece of evidence that can update a causal belief."""

    source: str
    statement: str
    weight: float = 0.0
    kind: str = "observation"
    metadata: Dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=_now)


@dataclass
class CausalBelief:
    """A defeasible belief about a directional causal mechanism."""

    cause: str
    effect: str
    probability: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    counterevidence: List[Evidence] = field(default_factory=list)
    mechanism: Optional[str] = None
    temporal_lag: Optional[str] = None
    version: int = 1
    updated_at: str = field(default_factory=_now)

    @property
    def key(self) -> Tuple[str, str]:
        return self.cause, self.effect

    def update(self, evidence: Evidence) -> None:
        """Apply a bounded additive belief update.

        This deliberately stays simple in v0.2. The API is stable so the update
        rule can later be replaced by Bayesian, learned, or domain-specific
        estimators without changing the agent loop.
        """
        weight = max(-1.0, min(1.0, evidence.weight))
        if weight >= 0:
            self.evidence.append(evidence)
        else:
            self.counterevidence.append(evidence)

        if weight >= 0:
            self.probability += (1.0 - self.probability) * weight
        else:
            self.probability += self.probability * weight

        self.probability = max(0.001, min(0.999, self.probability))
        self.version += 1
        self.updated_at = _now()


@dataclass
class Transition:
    """Experience memory: state + intervention -> observed outcome."""

    action: str
    arguments: Dict[str, Any]
    observation: Any
    context: Dict[str, Any] = field(default_factory=dict)
    expected_effects: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


class CausalWorldModel:
    """Explicit, continuously updated causal belief state.

    The model intentionally does not depend on NetworkX, an LLM vendor, or a
    particular causal inference package. Those are adapters around this core.
    """

    def __init__(self) -> None:
        self._beliefs: Dict[Tuple[str, str], CausalBelief] = {}
        self.transitions: List[Transition] = []

    def upsert_belief(
        self,
        cause: str,
        effect: str,
        probability: float = 0.5,
        **kwargs: Any,
    ) -> CausalBelief:
        key = (cause, effect)
        if key not in self._beliefs:
            self._beliefs[key] = CausalBelief(
                cause=cause,
                effect=effect,
                probability=max(0.001, min(0.999, probability)),
                **kwargs,
            )
        return self._beliefs[key]

    def beliefs(self) -> List[CausalBelief]:
        return list(self._beliefs.values())

    def get(self, cause: str, effect: str) -> Optional[CausalBelief]:
        return self._beliefs.get((cause, effect))

    def update_belief(
        self,
        cause: str,
        effect: str,
        evidence: Evidence,
        prior: float = 0.5,
    ) -> CausalBelief:
        belief = self.upsert_belief(cause, effect, probability=prior)
        belief.update(evidence)
        return belief

    def record_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def strongest_causes(self, effect: str, limit: int = 5) -> List[CausalBelief]:
        matches = [b for b in self._beliefs.values() if b.effect == effect]
        return sorted(matches, key=lambda b: b.probability, reverse=True)[:limit]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "beliefs": [
                {
                    "cause": b.cause,
                    "effect": b.effect,
                    "probability": b.probability,
                    "context": b.context,
                    "mechanism": b.mechanism,
                    "temporal_lag": b.temporal_lag,
                    "version": b.version,
                }
                for b in self.beliefs()
            ],
            "transition_count": len(self.transitions),
        }
