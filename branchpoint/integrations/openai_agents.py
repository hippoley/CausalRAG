from __future__ import annotations

import asyncio
import inspect
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

    By default this adapter owns only the pre-execution approval decision.
    Existing SDK tools continue to execute through the SDK and durable
    Branchpoint tools fail closed if that path would bypass the receipt ledger.

    ``function_tool(...)`` is the explicit exception: it builds an SDK
    FunctionTool whose invocation crosses the Branchpoint ToolRegistry boundary
    immediately before the application handler, so authorization, durable
    receipts, replay, and downstream idempotency remain enforced.
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
        registry_policy = self.tools.authorization_policy
        self.authorization_policy = (
            authorization_policy
            or registry_policy
            or CapabilityAuthorizationPolicy()
        )
        # Approval and execution must consult one policy source. When callers
        # explicitly override the adapter policy, bind the same object to the
        # registry used by Branchpoint-bound FunctionTools.
        if authorization_policy is not None or registry_policy is None:
            self.tools.authorization_policy = self.authorization_policy
        self.authorization_context = authorization_context
        self.authorization_resolver = authorization_resolver
        self._branchpoint_execution_tools: set[str] = set()

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
                risk=None,
                reversible=bool(tool.reversible),
            )

        try:
            tool_risk = float(tool.risk)
        except (TypeError, ValueError):
            tool_risk = float("nan")
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

        if (
            tool.require_durable_receipt
            and tool_name not in self._branchpoint_execution_tools
        ):
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

    def function_tool(
        self,
        tool_name: str,
        *,
        params_json_schema: Mapping[str, Any],
        strict_json_schema: bool = True,
        authorization_context: Optional[AuthorizationContext] = None,
    ):
        """Create an OpenAI Agents FunctionTool executed through Branchpoint.

        The SDK still owns model/run orchestration. Invocation crosses the
        Branchpoint ToolRegistry boundary immediately before the application
        handler. Durable tools derive a stable effect id from the SDK tool call
        id so resumed/replayed calls converge on one execution receipt.
        """

        normalized = str(tool_name).strip()
        if not normalized:
            raise ValueError("tool_name must be non-empty")
        tool = self.tools.get(normalized)
        if tool.require_durable_receipt and self.tools.execution_ledger is None:
            from branchpoint.execution import ExecutionBoundaryError

            raise ExecutionBoundaryError(
                f"Tool {normalized!r} requires a durable execution receipt, "
                "but no execution_ledger is configured."
            )
        if inspect.iscoroutinefunction(tool.handler):
            raise TypeError(
                "OpenAI Agents Branchpoint-bound tools currently require a "
                "synchronous ToolSpec handler"
            )

        schema = dict(params_json_schema)
        if schema.get("type") != "object":
            raise ValueError("params_json_schema must define an object schema")

        try:
            from agents import FunctionTool
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Agents integration requires the optional SDK. "
                "Install this repository with: pip install -e \".[openai-agents]\""
            ) from exc

        async def on_invoke_tool(run_context: Any, raw_arguments: str) -> Any:
            synthetic = _SyntheticApprovalItem(
                tool_name=normalized,
                call_id=str(getattr(run_context, "tool_call_id", "") or ""),
                arguments=raw_arguments,
            )
            arguments, error = self._arguments(synthetic)
            if arguments is None:
                raise ValueError(
                    f"Invalid arguments for {normalized!r}: {error}"
                )

            call_id = str(
                getattr(run_context, "tool_call_id", "") or ""
            ).strip()
            effect_id = None
            if tool.require_durable_receipt:
                if not call_id:
                    raise ValueError(
                        "Durable OpenAI Agents tools require a non-empty "
                        "tool_call_id"
                    )
                qualified_name = str(
                    getattr(run_context, "qualified_tool_name", normalized)
                    or normalized
                )
                effect_id = (
                    f"openai-agents:{qualified_name}:{call_id}"
                )

            context = self._context(
                run_context=run_context,
                tool_name=normalized,
                arguments=arguments,
                call_id=call_id,
                override=authorization_context,
            )

            return await asyncio.to_thread(
                self.tools.execute,
                normalized,
                arguments,
                effect_id=effect_id,
                authorization_context=context,
            )

        function_tool = FunctionTool(
            name=normalized,
            description=tool.description,
            params_json_schema=schema,
            on_invoke_tool=on_invoke_tool,
            strict_json_schema=bool(strict_json_schema),
            needs_approval=self.needs_approval(
                normalized,
                authorization_context=authorization_context,
            ),
        )
        self._branchpoint_execution_tools.add(normalized)
        return function_tool

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
