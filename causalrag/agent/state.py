from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .actions import DecisionRecord


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

    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.step)
