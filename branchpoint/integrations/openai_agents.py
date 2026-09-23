from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from branchpoint.authorization import (
    AuthorizationContext,
    CapabilityAuthorizationPolicy,
)
from branchpoint.tools import ToolRegistry, ToolSpec


class ApprovalOutcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"


@dataclass(frozen=True)
class OpenAIAgentsToolDecision:
    outcome: ApprovalOutcome
    tool_name: Optional[str]
    call_id: Optional[str]
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reason_code: str = ""
    reason: str = ""
    required_permissions: Tuple[str, ...] = ()
    missing_permissions: Tuple[str, ...] = ()
    risk: Optional[float] = None
    reversible: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "arguments": dict(self.arguments),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "required_permissions": list(self.required_permissions),
            "missing_permissions": list(self.missing_permissions),
            "risk": self.risk,
            "reversible": self.reversible,
        }


@dataclass(frozen=True)
class OpenAIAgentsResolution:
    state: Any
    decisions: Tuple[OpenAIAgentsToolDecision, ...]
    pending: Tuple[Any, ...]

    @property
    def can_resume(self) -> bool:
        return not self.pending


AuthorizationResolver = Callable[
    [Any, str, Mapping[str, Any], str],
    Optional[AuthorizationContext],
]


class OpenAIAgentsApprovalAdapter:
    """Map OpenAI Agents SDK approval interruptions onto Branchpoint policy.

    This adapter owns only the pre-execution approval decision. Tool execution
    remains owned by the OpenAI Agents SDK. Tools that require Branchpoint's
    durable execution receipt boundary therefore fail closed here instead of
    being auto-approved through a path that would bypass that boundary.
    """

    def __init__(
        self,
        tools: ToolRegistry | Iterable[ToolSpec],
        *,
        auto_approve_max_risk: float = 0.0,
        allow_irreversible_auto_approval: bool = False,
        authorization_policy: Optional[CapabilityAuthorizationPolicy] = None,
        authorization_context: Optional[AuthorizationContext] = None,
        authorization_resolver: Optional[AuthorizationResolver] = None,
    ) -> None:
        if authorization_context is not None and authorization_resolver is not None:
            raise ValueError(
                "Pass authorization_context or authorization_resolver, not both."
            )
        max_risk = float(auto_approve_max_risk)
        if not math.isfinite(max_risk) or max_risk < 0.0:
            raise ValueError(
                "auto_approve_max_risk must be finite and non-negative"
            )

        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.auto_approve_max_risk = max_risk
        self.allow_irreversible_auto_approval = bool(
            allow_irreversible_auto_approval
        )
        registry_policy = (
            self.tools.authorization_policy
            if isinstance(self.tools.authorization_policy, CapabilityAuthorizationPolicy)
            else None
        )
        self.authorization_policy = (
            authorization_policy
            or registry_policy
            or CapabilityAuthorizationPolicy()
        )
        self.authorization_context = authorization_context
        self.authorization_resolver = authorization_resolver

    @staticmethod
    def _tool_name(interruption: Any) -> Optional[str]:
        value = getattr(interruption, "tool_name", None)
        if value is None:
            raw = getattr(interruption, "raw_item", None)
            if isinstance(raw, Mapping):
                value = raw.get("name")
            elif raw is not None:
                value = getattr(raw, "name", None)
        text = str(value).strip() if value is not None else ""
        return text or None

    @staticmethod
    def _call_id(interruption: Any) -> Optional[str]:
        value = getattr(interruption, "call_id", None)
        if value is None:
            raw = getattr(interruption, "raw_item", None)
            if isinstance(raw, Mapping):
                value = raw.get("call_id") or raw.get("id")
            elif raw is not None:
                value = getattr(raw, "call_id", None) or getattr(raw, "id", None)
        return None if value is None else str(value)

    @staticmethod
    def _arguments(interruption: Any) -> tuple[Optional[dict[str, Any]], str]:
        value = getattr(interruption, "arguments", None)
        if value is None:
            raw = getattr(interruption, "raw_item", None)
            if isinstance(raw, Mapping):
                value = (
                    raw.get("arguments")
                    if raw.get("arguments") is not None
                    else raw.get("params") or raw.get("input")
                )
            elif raw is not None:
                value = getattr(raw, "arguments", None)
                if value is None:
                    value = getattr(raw, "params", None) or getattr(raw, "input", None)

        if isinstance(value, Mapping):
            return dict(value), ""
        if not isinstance(value, str) or not value.strip():
            return None, "missing_arguments"
        try:
            parsed = json.loads(
                value,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant: {constant}")
                ),
            )
        except (json.JSONDecodeError, ValueError):
            return None, "malformed_arguments"
        if not isinstance(parsed, dict):
            return None, "non_object_arguments"
        return parsed, ""

    def _context(
        self,
        *,
        run_context: Any,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_id: str,
        override: Optional[AuthorizationContext],
    ) -> Optional[AuthorizationContext]:
        if override is not None:
            return override
        if self.authorization_resolver is not None:
            return self.authorization_resolver(
                run_context,
                tool_name,
                arguments,
                call_id,
            )
        return self.authorization_context

    def decide(
        self,
        interruption: Any,
        *,
        authorization_context: Optional[AuthorizationContext] = None,
        run_context: Any = None,
    ) -> OpenAIAgentsToolDecision:
        tool_name = self._tool_name(interruption)
        call_id = self._call_id(interruption)
        arguments, argument_error = self._arguments(interruption)

        if tool_name is None:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=None,
                call_id=call_id,
                reason_code="missing_tool_name",
                reason="Tool identity is unavailable; approval remains paused.",
            )
        if arguments is None:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=tool_name,
                call_id=call_id,
                reason_code=argument_error,
                reason=(
                    "Tool arguments are not a validated JSON object; approval "
                    "remains paused."
                ),
            )

        try:
            tool = self.tools.get(tool_name)
        except KeyError:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code="unknown_tool",
                reason=(
                    "No canonical Branchpoint ToolSpec is registered for this "
                    "tool; approval remains paused."
                ),
            )

        context = self._context(
            run_context=run_context,
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id or "",
            override=authorization_context,
        )
        authorization = self.authorization_policy.authorize(
            context,
            tool_name=tool_name,
            required_permissions=tool.required_permissions,
            arguments=arguments,
        )
        if not authorization.allowed:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.DENY,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code=authorization.reason_code,
                reason=authorization.reason,
                required_permissions=authorization.required_permissions,
                missing_permissions=authorization.missing_permissions,
                risk=tool_risk,
                reversible=bool(tool.reversible),
            )

        tool_risk = float(tool.risk)
        if not math.isfinite(tool_risk) or tool_risk < 0.0:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code="invalid_tool_risk",
                reason=(
                    "Canonical ToolSpec risk is not a finite non-negative "
                    "number; approval remains paused."
                ),
                required_permissions=authorization.required_permissions,
                risk=tool_risk,
                reversible=bool(tool.reversible),
            )

        if tool.require_durable_receipt:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.DENY,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code="durable_execution_boundary_required",
                reason=(
                    "This tool requires a Branchpoint durable execution receipt "
                    "and cannot be executed directly by the approval-only adapter."
                ),
                required_permissions=authorization.required_permissions,
                risk=tool_risk,
                reversible=bool(tool.reversible),
            )

        if not tool.reversible and not self.allow_irreversible_auto_approval:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code="irreversible_action",
                reason="Irreversible tool calls require human approval.",
                required_permissions=authorization.required_permissions,
                risk=tool_risk,
                reversible=False,
            )

        if tool_risk > self.auto_approve_max_risk:
            return OpenAIAgentsToolDecision(
                ApprovalOutcome.REQUIRE_HUMAN,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason_code="risk_threshold",
                reason=(
                    f"Tool risk {tool_risk:g} exceeds auto-approval "
                    f"threshold {self.auto_approve_max_risk:g}."
                ),
                required_permissions=authorization.required_permissions,
                risk=tool_risk,
                reversible=bool(tool.reversible),
            )

        return OpenAIAgentsToolDecision(
            ApprovalOutcome.ALLOW,
            tool_name=tool_name,
            call_id=call_id,
            arguments=arguments,
            reason_code="auto_approved",
            reason="Canonical Branchpoint policy permits automatic approval.",
            required_permissions=authorization.required_permissions,
            risk=tool_risk,
            reversible=bool(tool.reversible),
        )

    def needs_approval(
        self,
        tool_name: str,
        *,
        authorization_context: Optional[AuthorizationContext] = None,
    ):
        """Return an OpenAI Agents function-tool needs_approval callback."""

        normalized = str(tool_name).strip()
        if not normalized:
            raise ValueError("tool_name must be non-empty")

        async def callback(
            run_context: Any,
            tool_parameters: dict[str, Any],
            call_id: str,
        ) -> bool:
            synthetic = _SyntheticApprovalItem(
                tool_name=normalized,
                call_id=str(call_id),
                arguments=dict(tool_parameters),
            )
            decision = self.decide(
                synthetic,
                authorization_context=authorization_context,
                run_context=run_context,
            )
            return decision.outcome is not ApprovalOutcome.ALLOW

        return callback

    def resolve(
        self,
        run_result: Any,
        *,
        authorization_context: Optional[AuthorizationContext] = None,
        run_context: Any = None,
    ) -> OpenAIAgentsResolution:
        """Resolve policy-decidable interruptions without resuming the run.

        ALLOW calls are approved on the returned RunState. DENY calls are
        rejected with the Branchpoint reason. REQUIRE_HUMAN calls are left
        untouched in pending for application UI or another human workflow.
        """

        interruptions = tuple(getattr(run_result, "interruptions", ()) or ())
        to_state = getattr(run_result, "to_state", None)
        if not callable(to_state):
            raise TypeError("run_result must expose a callable to_state()")
        state = to_state()

        decisions = []
        pending = []
        for interruption in interruptions:
            decision = self.decide(
                interruption,
                authorization_context=authorization_context,
                run_context=run_context,
            )
            decisions.append(decision)
            if decision.outcome is ApprovalOutcome.ALLOW:
                state.approve(interruption)
            elif decision.outcome is ApprovalOutcome.DENY:
                state.reject(
                    interruption,
                    rejection_message=decision.reason,
                )
            else:
                pending.append(interruption)

        return OpenAIAgentsResolution(
            state=state,
            decisions=tuple(decisions),
            pending=tuple(pending),
        )


@dataclass(frozen=True)
class _SyntheticApprovalItem:
    tool_name: str
    call_id: str
    arguments: Mapping[str, Any]
