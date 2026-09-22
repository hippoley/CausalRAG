from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


class ExecutionGuardError(RuntimeError):
    """Base error for consequence-time execution guard failures."""


class ExecutionGuardDenied(ExecutionGuardError):
    def __init__(self, decision: "ExecutionGuardDecision") -> None:
        self.decision = decision
        super().__init__(decision.reason)


@dataclass(frozen=True)
class ExecutionGuardDecision:
    allowed: bool
    reason_code: str = "allowed"
    reason: str = "Execution precondition satisfied"
    policy_id: str = "branchpoint.execution_guard.v1"
    evidence_version: Optional[str] = None


def normalize_guard_decision(value: Any) -> ExecutionGuardDecision:
    if isinstance(value, ExecutionGuardDecision):
        return value
    if value is True:
        return ExecutionGuardDecision(allowed=True)
    if value is False:
        return ExecutionGuardDecision(
            allowed=False,
            reason_code="precondition_failed",
            reason="Execution precondition failed.",
        )
    if isinstance(value, Mapping):
        return ExecutionGuardDecision(
            allowed=bool(value.get("allowed", False)),
            reason_code=str(value.get("reason_code") or (
                "allowed" if value.get("allowed") else "precondition_failed"
            )),
            reason=str(value.get("reason") or (
                "Execution precondition satisfied"
                if value.get("allowed")
                else "Execution precondition failed."
            )),
            policy_id=str(
                value.get("policy_id") or "branchpoint.execution_guard.v1"
            ),
            evidence_version=(
                None
                if value.get("evidence_version") is None
                else str(value.get("evidence_version"))
            ),
        )
    raise ExecutionGuardError(
        "Execution guards must return bool, mapping, or ExecutionGuardDecision."
    )
