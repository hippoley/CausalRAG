from __future__ import annotations

from typing import Optional, Protocol, Sequence

from branchpoint.agent.actions import CandidateAction
from branchpoint.agent.state import AgentState
from branchpoint.world_model.models import CausalWorldModel


class Reasoner(Protocol):
    """Model-agnostic policy interface for causal agency."""

    def propose(self, state: AgentState, world_model: CausalWorldModel) -> Sequence[CandidateAction]:
        ...

    def uncertainty(self, state: AgentState, world_model: CausalWorldModel) -> Optional[str]:
        ...
