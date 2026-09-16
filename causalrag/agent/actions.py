from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionKind(str, Enum):
    OBSERVE = "observe"
    RETRIEVE = "retrieve"
    ASK = "ask"
    INTERVENE = "intervene"
    WAIT = "wait"
    STOP = "stop"


@dataclass
class CandidateAction:
    kind: ActionKind
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    expected_goal_gain: float = 0.0
    expected_information_gain: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    irreversibility: float = 0.0
    rationale: str = ""

    @property
    def utility(self) -> float:
        return (
            self.expected_goal_gain
            + self.expected_information_gain
            - self.cost
            - self.risk
            - self.irreversibility
        )


@dataclass
class DecisionRecord:
    step: int
    uncertainty: Optional[str]
    candidates: List[CandidateAction]
    selected: CandidateAction
    beliefs_before: Dict[str, Any]
    rationale: str = ""
