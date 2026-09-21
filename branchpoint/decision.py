from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

from branchpoint.agent.actions import ActionKind, ActionScore, CandidateAction
from branchpoint.agent.capabilities import RuntimeCapabilities
from branchpoint.reasoning.policy import rank_actions
from branchpoint.tools.base import ToolRegistry, ToolSpec
from branchpoint.world_model.models import CausalWorldModel


@dataclass(frozen=True)
class DecisionResult:
    """Result of one bounded runtime arbitration."""

    selected: CandidateAction
    ranked_actions: Tuple[CandidateAction, ...]
    scores: Tuple[ActionScore, ...]

    @property
    def changed_proposer_order(self) -> bool:
        return bool(self.ranked_actions and self.ranked_actions[0] is not self.selected)


def decide(
    candidates: Sequence[CandidateAction],
    *,
    tools: Optional[Union[ToolRegistry, Iterable[ToolSpec]]] = None,
    world_model: Optional[CausalWorldModel] = None,
    capabilities: Optional[RuntimeCapabilities] = None,
) -> DecisionResult:
    """Arbitrate one bounded decision without running a full agent loop.

    The first candidate is treated as proposer order, not execution authority.
    When ToolSpec objects are supplied, their cost, risk, reversibility, and
    contracts are canonical inputs to runtime scoring.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        stop = CandidateAction(
            kind=ActionKind.STOP,
            name="stop",
            rationale="No candidate actions were supplied.",
        )
        return DecisionResult(selected=stop, ranked_actions=(stop,), scores=())

    if tools is None:
        registry = None
    elif isinstance(tools, ToolRegistry):
        registry = tools
    else:
        registry = ToolRegistry(tools)

    ranked = rank_actions(
        candidate_list,
        world_model=world_model,
        tools=registry,
        capabilities=capabilities,
    )
    ranked_actions = tuple(action for action, _score in ranked)
    scores = tuple(score for _action, score in ranked)
    selected = ranked_actions[0]

    return DecisionResult(
        selected=selected,
        ranked_actions=ranked_actions,
        scores=scores,
    )
