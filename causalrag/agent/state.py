from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .actions import DecisionRecord
from .temporal import PendingEffect


@dataclass
class Observation:
    action_name: str
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    goal: str
    step: int = 0
    max_steps: int = 10
    done: bool = False
    stop_reason: Optional[str] = None
    observations: List[Observation] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    scratch: Dict[str, Any] = field(default_factory=dict)
    virtual_time_seconds: float = 0.0
    pending_effects: List[PendingEffect] = field(default_factory=list)

    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.step)

    def advance_time(self, seconds: float) -> float:
        delta = max(0.0, float(seconds))
        self.virtual_time_seconds += delta
        for effect in self.pending_effects:
            effect.refresh(self.virtual_time_seconds)
        return delta

    def schedule_effect(self, effect: PendingEffect) -> None:
        self.pending_effects.append(effect)

    def active_pending_effects(self) -> List[PendingEffect]:
        now = self.virtual_time_seconds
        for effect in self.pending_effects:
            effect.refresh(now)
        return [effect for effect in self.pending_effects if not effect.observed and not effect.expired]

    def next_effect_ready_in(self) -> Optional[float]:
        active = self.active_pending_effects()
        if not active:
            return None
        return min(effect.seconds_until_ready(self.virtual_time_seconds) for effect in active)
