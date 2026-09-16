from __future__ import annotations

from typing import Sequence

from causalrag.agent.actions import ActionKind, CandidateAction


def select_action(candidates: Sequence[CandidateAction]) -> CandidateAction:
    """Select an action using causal utility rather than raw model preference."""
    if not candidates:
        return CandidateAction(
            kind=ActionKind.STOP,
            name="stop",
            rationale="No valid candidate actions were proposed.",
        )
    return max(candidates, key=lambda action: action.utility)
