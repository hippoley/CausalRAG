from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Evidence:
    """A single piece of evidence that can update a causal belief or hypothesis."""

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
        weight = max(-1.0, min(1.0, evidence.weight))
        if weight >= 0:
            self.evidence.append(evidence)
            self.probability += (1.0 - self.probability) * weight
        else:
            self.counterevidence.append(evidence)
            self.probability += self.probability * weight

        self.probability = max(0.001, min(0.999, self.probability))
        self.version += 1
        self.updated_at = _now()


@dataclass
class Hypothesis:
    """A competing explanation that remains explicitly falsifiable."""

    hypothesis_id: str
    statement: str
    probability: float = 0.5
    rationale: str = ""
    falsifiers: List[str] = field(default_factory=list)
    supporting_evidence: List[Evidence] = field(default_factory=list)
    conflicting_evidence: List[Evidence] = field(default_factory=list)
    status: str = "active"
    version: int = 1
    updated_at: str = field(default_factory=_now)

    def _refresh_status(self) -> None:
        if self.probability >= 0.9:
            self.status = "supported"
        elif self.probability <= 0.1:
            self.status = "rejected"
        else:
            self.status = "active"

    def update(self, evidence: Evidence) -> None:
        weight = max(-1.0, min(1.0, evidence.weight))
        if weight >= 0:
            self.supporting_evidence.append(evidence)
            self.probability += (1.0 - self.probability) * weight
        else:
            self.conflicting_evidence.append(evidence)
            self.probability += self.probability * weight

        self.probability = max(0.001, min(0.999, self.probability))
        self._refresh_status()
        self.version += 1
        self.updated_at = _now()

    def set_probability(self, probability: float, evidence: Optional[Evidence] = None) -> None:
        """Set an externally computed probability while preserving evidence direction.

        For exact posterior updates, ``evidence.weight`` is authoritative about
        whether the observation supported or conflicted with the hypothesis.
        This matters when the external estimator normalizes a set of operational
        credences before computing a posterior: comparing the posterior to the
        raw pre-normalized credence can give the wrong evidence direction.
        """
        self.probability = max(0.001, min(0.999, float(probability)))
        if evidence is not None:
            if float(evidence.weight) >= 0.0:
                self.supporting_evidence.append(evidence)
            else:
                self.conflicting_evidence.append(evidence)
        self._refresh_status()
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
    """Explicit, continuously updated causal and epistemic state."""

    def __init__(self) -> None:
        self._beliefs: Dict[Tuple[str, str], CausalBelief] = {}
        self._hypotheses: Dict[str, Hypothesis] = {}
        self.transitions: List[Transition] = []

    def upsert_belief(self, cause: str, effect: str, probability: float = 0.5, **kwargs: Any) -> CausalBelief:
        key = (cause, effect)
        if key not in self._beliefs:
            self._beliefs[key] = CausalBelief(cause=cause, effect=effect, probability=max(0.001, min(0.999, probability)), **kwargs)
        return self._beliefs[key]

    def beliefs(self) -> List[CausalBelief]:
        return list(self._beliefs.values())

    def get(self, cause: str, effect: str) -> Optional[CausalBelief]:
        return self._beliefs.get((cause, effect))

    def update_belief(self, cause: str, effect: str, evidence: Evidence, prior: float = 0.5) -> CausalBelief:
        belief = self.upsert_belief(cause, effect, probability=prior)
        belief.update(evidence)
        return belief

    def upsert_hypothesis(self, hypothesis_id: str, statement: str, probability: float = 0.5, rationale: str = "", falsifiers: Optional[Iterable[str]] = None) -> Hypothesis:
        hypothesis_id = str(hypothesis_id).strip()
        if not hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        existing = self._hypotheses.get(hypothesis_id)
        if existing is None:
            existing = Hypothesis(hypothesis_id=hypothesis_id, statement=str(statement).strip(), probability=max(0.001, min(0.999, float(probability))), rationale=str(rationale or ""), falsifiers=[str(item) for item in (falsifiers or []) if str(item).strip()])
            self._hypotheses[hypothesis_id] = existing
        else:
            if statement:
                existing.statement = str(statement).strip()
            if rationale:
                existing.rationale = str(rationale)
            if falsifiers:
                existing.falsifiers = [str(item) for item in falsifiers if str(item).strip()]
            existing.updated_at = _now()
        return existing

    def hypotheses(self, include_rejected: bool = True) -> List[Hypothesis]:
        values = list(self._hypotheses.values())
        return values if include_rejected else [h for h in values if h.status != "rejected"]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def update_hypothesis(self, hypothesis_id: str, evidence: Evidence) -> Optional[Hypothesis]:
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return None
        hypothesis.update(evidence)
        return hypothesis

    def set_hypothesis_probability(self, hypothesis_id: str, probability: float, evidence: Optional[Evidence] = None) -> Optional[Hypothesis]:
        """Set an exact externally-computed posterior while preserving evidence history."""
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return None
        hypothesis.set_probability(probability, evidence=evidence)
        return hypothesis

    def sync_hypotheses(self, proposals: Iterable[Any]) -> None:
        for proposal in proposals:
            data = proposal if isinstance(proposal, dict) else {
                "id": getattr(proposal, "hypothesis_id", None),
                "statement": getattr(proposal, "statement", ""),
                "probability": getattr(proposal, "probability", 0.5),
                "rationale": getattr(proposal, "rationale", ""),
                "falsifiers": getattr(proposal, "falsifiers", []),
            }
            hypothesis_id = data.get("id") or data.get("hypothesis_id")
            statement = data.get("statement")
            if not hypothesis_id or not statement:
                continue
            try:
                probability = float(data.get("probability", 0.5))
            except (TypeError, ValueError):
                probability = 0.5
            self.upsert_hypothesis(str(hypothesis_id), str(statement), probability, str(data.get("rationale") or ""), data.get("falsifiers") or [])

    def record_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def strongest_causes(self, effect: str, limit: int = 5) -> List[CausalBelief]:
        matches = [b for b in self._beliefs.values() if b.effect == effect]
        return sorted(matches, key=lambda b: b.probability, reverse=True)[:limit]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "beliefs": [{"cause": b.cause, "effect": b.effect, "probability": b.probability, "context": b.context, "mechanism": b.mechanism, "temporal_lag": b.temporal_lag, "version": b.version} for b in self.beliefs()],
            "hypotheses": [{"id": h.hypothesis_id, "statement": h.statement, "probability": h.probability, "rationale": h.rationale, "falsifiers": h.falsifiers, "status": h.status, "support_count": len(h.supporting_evidence), "conflict_count": len(h.conflicting_evidence), "version": h.version} for h in self.hypotheses()],
            "transition_count": len(self.transitions),
        }
