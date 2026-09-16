from __future__ import annotations

from typing import Optional, Protocol, Sequence

from causalrag.agent.actions import CandidateAction
from causalrag.agent.state import AgentState
from causalrag.world_model.models import CausalWorldModel


class Reasoner(Protocol):
    """Model-agnostic policy interface for causal agency."""

    def propose(self, state: AgentState, world_model: CausalWorldModel) -> Sequence[CandidateAction]:
        ...

    def uncertainty(self, state: AgentState, world_model: CausalWorldModel) -> Optional[str]:
        ...
