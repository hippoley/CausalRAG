from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Callable, Iterable, Mapping, Optional

from branchpoint.authorization import (
    AuthorizationContext,
    CapabilityAuthorizationPolicy,
)
from branchpoint.tools import ToolRegistry, ToolSpec

try:
    from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
    from langchain_core.messages import ToolMessage
except ImportError as exc:  # pragma: no cover - exercised by import guidance
    raise RuntimeError(
        "LangChain integration requires the optional dependencies. "
        "Install this repository with: pip install -e \".[langchain]\""
    ) from exc


AuthorizationResolver = Callable[
    [Any, str, Mapping[str, Any], str],
    Optional[AuthorizationContext],
]


class LangChainBranchpointError(RuntimeError):
    """Invalid or ungoverned tool call at the LangChain execution boundary."""


def _tool_result_content(result: Any) -> str | list[str | dict[Any, Any]]:
    if isinstance(result, str):
        return result
    if isinstance(result, list) and all(isinstance(item, (str, dict)) for item in result):
        return result
    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return str(result)


class LangChainBranchpointMiddleware(AgentMiddleware):
    """Execute canonical LangChain tool calls through Branchpoint.

    LangChain continues to own model orchestration, graph state, checkpointing,
    and human-interrupt flow. Once a tool call reaches this middleware,
    Branchpoint owns execution authorization and durable receipt semantics for
    every registered ToolSpec.
    """

    def __init__(
        self,
        tools: ToolRegistry | Iterable[ToolSpec],
        *,
        authorization_policy: Optional[CapabilityAuthorizationPolicy] = None,
        authorization_context: Optional[AuthorizationContext] = None,
        authorization_resolver: Optional[AuthorizationResolver] = None,
        unknown_tool_policy: str = "deny",
    ) -> None:
        super().__init__()
        if authorization_context is not None and authorization_resolver is not None:
            raise ValueError(
                "Pass authorization_context or authorization_resolver, not both."
            )
        if unknown_tool_policy not in {"deny", "passthrough"}:
            raise ValueError("unknown_tool_policy must be 'deny' or 'passthrough'")

        self.tools = []
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        registry_policy = self.registry.authorization_policy
        self.authorization_policy = (
            authorization_policy
            or registry_policy
            or CapabilityAuthorizationPolicy()
        )
        if authorization_policy is not None or registry_policy is None:
            self.registry.authorization_policy = self.authorization_policy

        self.authorization_context = authorization_context
        self.authorization_resolver = authorization_resolver
        self.unknown_tool_policy = unknown_tool_policy

    @staticmethod
    def _parts(request: ToolCallRequest) -> tuple[str, dict[str, Any], str]:
        call = request.tool_call
        name = str(call.get("name") or "").strip()
        if not name:
            raise LangChainBranchpointError("tool call name must be non-empty")

        args = call.get("args")
        if not isinstance(args, Mapping):
            raise LangChainBranchpointError(
                f"tool call arguments for {name!r} must be an object"
            )

        call_id = str(call.get("id") or "").strip()
        return name, dict(args), call_id

    @staticmethod
    def _runtime_context(request: ToolCallRequest) -> Any:
        runtime = getattr(request, "runtime", None)
        return None if runtime is None else getattr(runtime, "context", None)

    def _context(
        self,
        request: ToolCallRequest,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> Optional[AuthorizationContext]:
        if self.authorization_resolver is not None:
            return self.authorization_resolver(
                self._runtime_context(request),
                tool_name,
                arguments,
                call_id,
            )
        return self.authorization_context

    def _registered(self, name: str) -> bool:
        return name in self.registry.specs()

    @staticmethod
    def _effect_id(tool: ToolSpec, name: str, call_id: str) -> Optional[str]:
        if not tool.require_durable_receipt:
            return None
        if not call_id:
            raise LangChainBranchpointError(
                f"Durable LangChain tool {name!r} requires a non-empty tool call id"
            )
        return f"langchain:{name}:{call_id}"

    @staticmethod
    def _evidence(
        *,
        tool: ToolSpec,
        tool_name: str,
        call_id: str,
        effect_id: Optional[str],
        replayed: bool,
        receipt: Any,
        policy: Any,
    ) -> dict[str, Any]:
        return {
            "schema_version": "branchpoint.langchain.execution.v1",
            "execution_boundary": "branchpoint",
            "tool_name": tool_name,
            "call_id": call_id or None,
            "durable": bool(tool.require_durable_receipt),
            "effect_id": effect_id,
            "replayed": bool(replayed),
            "receipt_status": (
                getattr(receipt, "status", None)
                if receipt is not None
                else "not_required"
            ),
            "effect_hash": (
                getattr(receipt, "effect_hash", None)
                if receipt is not None
                else None
            ),
            "authorization_rechecked": True,
            "authorization_policy_id": getattr(policy, "policy_id", None),
        }

    def _message(
        self,
        *,
        result: Any,
        tool_name: str,
        call_id: str,
        evidence: Mapping[str, Any],
    ) -> ToolMessage:
        if not call_id:
            raise LangChainBranchpointError(
                f"LangChain tool {tool_name!r} requires a non-empty call id "
                "to produce a ToolMessage"
            )
        return ToolMessage(
            content=_tool_result_content(result),
            tool_call_id=call_id,
            name=tool_name,
            status="success",
            artifact={"branchpoint": dict(evidence)},
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        name, arguments, call_id = self._parts(request)
        if not self._registered(name):
            if self.unknown_tool_policy == "passthrough":
                return handler(request)
            raise LangChainBranchpointError(
                f"No canonical Branchpoint ToolSpec is registered for {name!r}"
            )

        tool = self.registry.get(name)
        effect_id = self._effect_id(tool, name, call_id)
        ledger = self.registry.execution_ledger
        prior = ledger.get(effect_id) if effect_id is not None and ledger is not None else None
        replayed = bool(prior is not None and prior.status == "succeeded")
        context = self._context(
            request,
            tool_name=name,
            arguments=arguments,
            call_id=call_id,
        )

        result = self.registry.execute(
            name,
            arguments,
            effect_id=effect_id,
            authorization_context=context,
        )
        receipt = ledger.get(effect_id) if effect_id is not None and ledger is not None else None
        return self._message(
            result=result,
            tool_name=name,
            call_id=call_id,
            evidence=self._evidence(
                tool=tool,
                tool_name=name,
                call_id=call_id,
                effect_id=effect_id,
                replayed=replayed,
                receipt=receipt,
                policy=self.authorization_policy,
            ),
        )

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        name, arguments, call_id = self._parts(request)
        if not self._registered(name):
            if self.unknown_tool_policy == "passthrough":
                return await handler(request)
            raise LangChainBranchpointError(
                f"No canonical Branchpoint ToolSpec is registered for {name!r}"
            )

        tool = self.registry.get(name)
        effect_id = self._effect_id(tool, name, call_id)
        ledger = self.registry.execution_ledger
        prior = (
            await asyncio.to_thread(ledger.get, effect_id)
            if effect_id is not None and ledger is not None
            else None
        )
        replayed = bool(prior is not None and prior.status == "succeeded")
        context = self._context(
            request,
            tool_name=name,
            arguments=arguments,
            call_id=call_id,
        )

        result = await self.registry.execute_async(
            name,
            arguments,
            effect_id=effect_id,
            authorization_context=context,
        )
        receipt = (
            await asyncio.to_thread(ledger.get, effect_id)
            if effect_id is not None and ledger is not None
            else None
        )
        return self._message(
            result=result,
            tool_name=name,
            call_id=call_id,
            evidence=self._evidence(
                tool=tool,
                tool_name=name,
                call_id=call_id,
                effect_id=effect_id,
                replayed=replayed,
                receipt=receipt,
                policy=self.authorization_policy,
            ),
        )


def langchain_human_in_the_loop(
    tools: ToolRegistry | Iterable[ToolSpec],
    *,
    auto_approve_max_risk: float = 0.0,
    allow_irreversible_auto_approval: bool = False,
):
    """Build LangChain's native HITL middleware from canonical ToolSpec policy.

    High-risk or irreversible calls are interrupted using approve/reject only.
    Execution-time authorization remains Branchpoint-owned and is rechecked
    after resume.
    """

    threshold = float(auto_approve_max_risk)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "auto_approve_max_risk must be finite and non-negative"
        )

    registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
    interrupt_on: dict[str, Any] = {}
    for name, tool in registry.specs().items():
        try:
            risk = float(tool.risk)
        except (TypeError, ValueError):
            risk = float("nan")

        requires_review = (
            not math.isfinite(risk)
            or risk < 0.0
            or risk > threshold
            or (not tool.reversible and not allow_irreversible_auto_approval)
        )
        interrupt_on[name] = (
            {
                "allowed_decisions": ["approve", "reject"],
                "description": (
                    "Branchpoint policy requires human approval before this "
                    "tool crosses the execution boundary."
                ),
            }
            if requires_review
            else False
        )

    from langchain.agents.middleware import HumanInTheLoopMiddleware

    return HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        edit_notice=None,
    )


def langchain_branchpoint_stack(
    tools: ToolRegistry | Iterable[ToolSpec],
    *,
    auto_approve_max_risk: float = 0.0,
    allow_irreversible_auto_approval: bool = False,
    authorization_policy: Optional[CapabilityAuthorizationPolicy] = None,
    authorization_context: Optional[AuthorizationContext] = None,
    authorization_resolver: Optional[AuthorizationResolver] = None,
    unknown_tool_policy: str = "deny",
):
    """Return LangChain middleware in safe human-gate -> execution order."""

    registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
    hitl = langchain_human_in_the_loop(
        registry,
        auto_approve_max_risk=auto_approve_max_risk,
        allow_irreversible_auto_approval=allow_irreversible_auto_approval,
    )
    execution = LangChainBranchpointMiddleware(
        registry,
        authorization_policy=authorization_policy,
        authorization_context=authorization_context,
        authorization_resolver=authorization_resolver,
        unknown_tool_policy=unknown_tool_policy,
    )
    return (hitl, execution)
