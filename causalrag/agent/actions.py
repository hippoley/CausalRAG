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
    tests_hypotheses: List[str] = field(default_factory=list)
    falsification_target: Optional[str] = None

    @property
    def utility(self) -> float:
        """Legacy/model-proposed utility before runtime epistemic rescoring."""
        return (
            self.expected_goal_gain
            + self.expected_information_gain
            - self.cost
            - self.risk
            - self.irreversibility
        )


@dataclass
class ActionScore:
    """Runtime score used to select one candidate action."""

    candidate_index: int
    action_name: str
    action_kind: ActionKind
    total_utility: float
    goal_gain: float
    information_gain: float
    information_source: str
    model_information_gain: float
    discrimination_score: Optional[float]
    cost: float
    risk: float
    irreversibility: float


@dataclass
class DecisionRecord:
    step: int
    uncertainty: Optional[str]
    candidates: List[CandidateAction]
    selected: CandidateAction
    beliefs_before: Dict[str, Any]
    rationale: str = ""
    action_scores: List[ActionScore] = field(default_factory=list)
