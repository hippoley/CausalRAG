from __future__ import annotations

import math

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from branchpoint.authorization import (
    AuthorizationContext,
    CapabilityAuthorizationPolicy,
)
from branchpoint.execution import canonical_effect, effect_hash
from branchpoint.tools import ToolRegistry


class ExecutionGateOutcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"


@dataclass(frozen=True)
class ExecutionGateDecision:
    outcome: ExecutionGateOutcome
    tool_name: str
    proposal_hash: Optional[str]
    reason_code: str
    reason: str
    required_permissions: Tuple[str, ...] = ()
    missing_permissions: Tuple[str, ...] = ()
    risk: Optional[float] = None
    cost: Optional[float] = None
    reversible: Optional[bool] = None
    durable_receipt_required: Optional[bool] = None
    policy_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "tool_name": self.tool_name,
            "proposal_hash": self.proposal_hash,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "required_permissions": list(self.required_permissions),
            "missing_permissions": list(self.missing_permissions),
            "risk": self.risk,
            "cost": self.cost,
            "reversible": self.reversible,
            "durable_receipt_required": self.durable_receipt_required,
            "policy_id": self.policy_id,
        }


class ToolExecutionGate:
    """Framework-neutral pre-execution policy over canonical ToolSpecs.

    This class never executes tools. It evaluates current principal authority
    plus server-owned ToolSpec risk/reversibility and returns one of:
    allow, deny, or require_human.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        auto_execute_max_risk: float = 0.0,
        allow_irreversible_auto_execute: bool = False,
        authorization_policy: Optional[CapabilityAuthorizationPolicy] = None,
    ) -> None:
        threshold = float(auto_execute_max_risk)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "auto_execute_max_risk must be finite and non-negative"
            )
        self.tools = tools
        self.auto_execute_max_risk = threshold
        self.allow_irreversible_auto_execute = bool(
            allow_irreversible_auto_execute
        )
        self.authorization_policy = (
            authorization_policy
            or tools.authorization_policy
            or CapabilityAuthorizationPolicy()
        )
        if tools.authorization_policy is None:
            tools.authorization_policy = self.authorization_policy

    def preview(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        authorization_context: Optional[AuthorizationContext] = None,
    ) -> ExecutionGateDecision:
        normalized = str(tool_name).strip()
        if not normalized:
            return ExecutionGateDecision(
                ExecutionGateOutcome.DENY,
                tool_name="",
                proposal_hash=None,
                reason_code="missing_tool_name",
                reason="Tool name must be non-empty.",
            )

        try:
            tool = self.tools.get(normalized)
        except KeyError:
            return ExecutionGateDecision(
                ExecutionGateOutcome.DENY,
                tool_name=normalized,
                proposal_hash=None,
                reason_code="unknown_tool",
                reason=f"No canonical ToolSpec is registered for {normalized!r}.",
            )

        try:
            canonical_effect(normalized, arguments)
            proposal_digest = effect_hash(normalized, arguments)
        except Exception as exc:
            return ExecutionGateDecision(
                ExecutionGateOutcome.DENY,
                tool_name=normalized,
                proposal_hash=None,
                reason_code="non_canonical_arguments",
                reason=f"Arguments are not canonical JSON-compatible values: {exc}",
                risk=_safe_float(tool.risk),
                cost=_safe_float(tool.cost),
                reversible=bool(tool.reversible),
                durable_receipt_required=bool(tool.require_durable_receipt),
            )

        authorization = self.authorization_policy.authorize(
            authorization_context,
            tool_name=normalized,
            required_permissions=tool.required_permissions,
            arguments=arguments,
        )
        common = {
            "tool_name": normalized,
            "proposal_hash": proposal_digest,
            "required_permissions": authorization.required_permissions,
            "risk": _safe_float(tool.risk),
            "cost": _safe_float(tool.cost),
            "reversible": bool(tool.reversible),
            "durable_receipt_required": bool(tool.require_durable_receipt),
            "policy_id": authorization.policy_id,
        }

        if not authorization.allowed:
            return ExecutionGateDecision(
                ExecutionGateOutcome.DENY,
                reason_code=authorization.reason_code,
                reason=authorization.reason,
                missing_permissions=authorization.missing_permissions,
                **common,
            )

        risk = _safe_float(tool.risk)
        cost = _safe_float(tool.cost)
        if risk is None or risk < 0.0:
            return ExecutionGateDecision(
                ExecutionGateOutcome.REQUIRE_HUMAN,
                reason_code="invalid_tool_risk",
                reason=(
                    "Canonical ToolSpec risk is not a finite non-negative "
                    "number; automatic execution is disabled."
                ),
                **common,
            )
        if cost is None or cost < 0.0:
            return ExecutionGateDecision(
                ExecutionGateOutcome.REQUIRE_HUMAN,
                reason_code="invalid_tool_cost",
                reason=(
                    "Canonical ToolSpec cost is not a finite non-negative "
                    "number; automatic execution is disabled."
                ),
                **common,
            )

        if not tool.reversible and not self.allow_irreversible_auto_execute:
            return ExecutionGateDecision(
                ExecutionGateOutcome.REQUIRE_HUMAN,
                reason_code="irreversible_action",
                reason="Irreversible tool calls require trusted approval.",
                **common,
            )

        if risk > self.auto_execute_max_risk:
            return ExecutionGateDecision(
                ExecutionGateOutcome.REQUIRE_HUMAN,
                reason_code="risk_threshold",
                reason=(
                    f"Tool risk {risk:g} exceeds automatic execution "
                    f"threshold {self.auto_execute_max_risk:g}."
                ),
                **common,
            )

        return ExecutionGateDecision(
            ExecutionGateOutcome.ALLOW,
            reason_code="auto_execute",
            reason="Canonical Branchpoint policy permits automatic execution.",
            **common,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
