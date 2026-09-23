from __future__ import annotations

import asyncio
import inspect

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from branchpoint.agent.temporal import TemporalEffectContract
    from branchpoint.experiments import (
        DecisionPreferences,
        ExperimentContract,
        InterventionContract,
    )
    from branchpoint.observability import CausalTelemetry
    from branchpoint.execution import ExecutionLedger


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    risk: float = 0.0
    cost: float = 0.0
    reversible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_contract: Optional["ExperimentContract"] = None
    intervention_contract: Optional["InterventionContract"] = None
    temporal_effect_contract: Optional["TemporalEffectContract"] = None
    require_durable_receipt: bool = False
    idempotency_key_argument: Optional[str] = None
    required_permissions: Tuple[str, ...] = ()


class ToolRegistry:
    def __init__(
        self,
        tools: Optional[Iterable[ToolSpec]] = None,
        decision_preferences: Optional["DecisionPreferences"] = None,
        telemetry: Optional["CausalTelemetry"] = None,
        execution_ledger: Optional["ExecutionLedger"] = None,
        authorization_policy: Optional[Any] = None,
    ) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.decision_preferences = decision_preferences
        self.telemetry = telemetry
        self.execution_ledger = execution_ledger
        self.authorization_policy = authorization_policy
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        if self.decision_preferences is not None:
            tool = self.decision_preferences.apply_to_tool(tool)
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def specs(self) -> Mapping[str, ToolSpec]:
        return dict(self._tools)

    def _invoke(self, tool: ToolSpec, arguments: Dict[str, Any], attributes: Dict[str, Any]) -> Any:
        if self.telemetry is None:
            return tool.handler(**arguments)

        if self.telemetry.capture_content:
            attributes["branchpoint.tool.arguments"] = dict(arguments)
        with self.telemetry.span(f"execute_tool {tool.name}", attributes) as span:
            result = tool.handler(**arguments)
            if self.telemetry.capture_content:
                span.set_attribute("branchpoint.tool.result", result)
            return result

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        effect_id: Optional[str] = None,
        authorization_context: Optional[Any] = None,
    ) -> Any:
        tool = self.get(name)
        if inspect.iscoroutinefunction(tool.handler):
            raise TypeError(
                f"Tool {tool.name!r} has an async handler; use execute_async()."
            )
        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "branchpoint.tool.kind": tool.metadata.get("kind", "tool"),
            "branchpoint.tool.cost": float(tool.cost),
            "branchpoint.tool.risk": float(tool.risk),
            "branchpoint.tool.reversible": bool(tool.reversible),
            "branchpoint.tool.argument_names": sorted(str(key) for key in arguments),
        }

        from branchpoint.authorization import AuthorizationDenied, CapabilityAuthorizationPolicy

        policy = self.authorization_policy or CapabilityAuthorizationPolicy()
        authorization = policy.authorize(
            authorization_context,
            tool_name=tool.name,
            required_permissions=tool.required_permissions,
            arguments=arguments,
        )
        if self.telemetry is not None:
            auth_attributes = {
                "branchpoint.authorization.allowed": bool(authorization.allowed),
                "branchpoint.authorization.reason_code": authorization.reason_code,
                "branchpoint.authorization.policy_id": authorization.policy_id,
                "branchpoint.authorization.required_permission_count": len(
                    authorization.required_permissions
                ),
                "branchpoint.authorization.missing_permission_count": len(
                    authorization.missing_permissions
                ),
                "gen_ai.tool.name": tool.name,
            }
            if self.telemetry.capture_content and authorization.principal_id:
                auth_attributes["branchpoint.authorization.principal_id"] = (
                    authorization.principal_id
                )
            self.telemetry.event("branchpoint.authorization", auth_attributes)
        if not authorization.allowed:
            raise AuthorizationDenied(authorization)

        ledger = self.execution_ledger
        if tool.require_durable_receipt and ledger is None:
            from branchpoint.execution import ExecutionBoundaryError
            raise ExecutionBoundaryError(
                f"Tool {tool.name!r} requires a durable execution receipt, "
                "but no execution_ledger is configured."
            )

        if ledger is None:
            return self._invoke(tool, dict(arguments), attributes)

        if effect_id is None:
            if tool.require_durable_receipt:
                from branchpoint.execution import ExecutionBoundaryError
                raise ExecutionBoundaryError(
                    f"Tool {tool.name!r} requires a non-empty effect_id."
                )
            return self._invoke(tool, dict(arguments), attributes)

        from branchpoint.execution import (
            EffectIdentityConflict,
            ExecutionInProgress,
            PreviousExecutionFailed,
        )

        if tool.idempotency_key_argument:
            existing = arguments.get(tool.idempotency_key_argument)
            if existing is not None and str(existing) != str(effect_id):
                raise EffectIdentityConflict(
                    f"Argument {tool.idempotency_key_argument!r} conflicts with effect_id."
                )

        receipt, claimed = ledger.claim(effect_id, tool.name, arguments)
        attributes["branchpoint.execution.effect_id"] = effect_id
        attributes["branchpoint.execution.effect_hash"] = receipt.effect_hash

        if not claimed:
            if receipt.status == "succeeded":
                if self.telemetry is not None:
                    self.telemetry.event(
                        "branchpoint.execution.replayed",
                        {
                            "branchpoint.execution.effect_id": effect_id,
                            "branchpoint.execution.effect_hash": receipt.effect_hash,
                            "gen_ai.tool.name": tool.name,
                        },
                    )
                return receipt.result
            if receipt.status == "failed":
                raise PreviousExecutionFailed(
                    f"Effect {effect_id!r} previously failed with "
                    f"{receipt.error_type or 'error'}: {receipt.error_message or ''}"
                )
            raise ExecutionInProgress(
                f"Effect {effect_id!r} is already in_flight. "
                "Reconcile the authoritative world outcome before retrying."
            )

        call_arguments = dict(arguments)
        if tool.idempotency_key_argument:
            call_arguments[tool.idempotency_key_argument] = effect_id
            attributes["branchpoint.execution.downstream_idempotency"] = True
        else:
            attributes["branchpoint.execution.downstream_idempotency"] = False

        try:
            result = self._invoke(tool, call_arguments, attributes)
            ledger.complete(effect_id, result)
            if self.telemetry is not None:
                self.telemetry.event(
                    "branchpoint.execution.completed",
                    {
                        "branchpoint.execution.effect_id": effect_id,
                        "branchpoint.execution.effect_hash": receipt.effect_hash,
                        "gen_ai.tool.name": tool.name,
                    },
                )
            return result
        except Exception as exc:
            ledger.fail(effect_id, exc)
            raise

    async def _invoke_async(
        self,
        tool: ToolSpec,
        arguments: Dict[str, Any],
        attributes: Dict[str, Any],
    ) -> Any:
        async def invoke_handler() -> Any:
            if inspect.iscoroutinefunction(tool.handler):
                return await tool.handler(**arguments)
            result = await asyncio.to_thread(tool.handler, **arguments)
            if inspect.isawaitable(result):
                return await result
            return result

        if self.telemetry is None:
            return await invoke_handler()

        if self.telemetry.capture_content:
            attributes["branchpoint.tool.arguments"] = dict(arguments)
        with self.telemetry.span(f"execute_tool {tool.name}", attributes) as span:
            result = await invoke_handler()
            if self.telemetry.capture_content:
                span.set_attribute("branchpoint.tool.result", result)
            return result

    async def execute_async(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        effect_id: Optional[str] = None,
        authorization_context: Optional[Any] = None,
    ) -> Any:
        """Async execution with the same authorization and receipt contract as execute()."""

        tool = self.get(name)
        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "branchpoint.tool.kind": tool.metadata.get("kind", "tool"),
            "branchpoint.tool.cost": float(tool.cost),
            "branchpoint.tool.risk": float(tool.risk),
            "branchpoint.tool.reversible": bool(tool.reversible),
            "branchpoint.tool.argument_names": sorted(str(key) for key in arguments),
            "branchpoint.execution.async": True,
        }

        from branchpoint.authorization import AuthorizationDenied, CapabilityAuthorizationPolicy

        policy = self.authorization_policy or CapabilityAuthorizationPolicy()
        authorization = policy.authorize(
            authorization_context,
            tool_name=tool.name,
            required_permissions=tool.required_permissions,
            arguments=arguments,
        )
        if self.telemetry is not None:
            auth_attributes = {
                "branchpoint.authorization.allowed": bool(authorization.allowed),
                "branchpoint.authorization.reason_code": authorization.reason_code,
                "branchpoint.authorization.policy_id": authorization.policy_id,
                "branchpoint.authorization.required_permission_count": len(
                    authorization.required_permissions
                ),
                "branchpoint.authorization.missing_permission_count": len(
                    authorization.missing_permissions
                ),
                "gen_ai.tool.name": tool.name,
            }
            if self.telemetry.capture_content and authorization.principal_id:
                auth_attributes["branchpoint.authorization.principal_id"] = (
                    authorization.principal_id
                )
            self.telemetry.event("branchpoint.authorization", auth_attributes)
        if not authorization.allowed:
            raise AuthorizationDenied(authorization)

        ledger = self.execution_ledger
        if tool.require_durable_receipt and ledger is None:
            from branchpoint.execution import ExecutionBoundaryError

            raise ExecutionBoundaryError(
                f"Tool {tool.name!r} requires a durable execution receipt, "
                "but no execution_ledger is configured."
            )

        if ledger is None:
            return await self._invoke_async(tool, dict(arguments), attributes)

        if effect_id is None:
            if tool.require_durable_receipt:
                from branchpoint.execution import ExecutionBoundaryError

                raise ExecutionBoundaryError(
                    f"Tool {tool.name!r} requires a non-empty effect_id."
                )
            return await self._invoke_async(tool, dict(arguments), attributes)

        from branchpoint.execution import (
            EffectIdentityConflict,
            ExecutionInProgress,
            PreviousExecutionFailed,
        )

        if tool.idempotency_key_argument:
            existing = arguments.get(tool.idempotency_key_argument)
            if existing is not None and str(existing) != str(effect_id):
                raise EffectIdentityConflict(
                    f"Argument {tool.idempotency_key_argument!r} conflicts with effect_id."
                )

        receipt, claimed = await asyncio.to_thread(
            ledger.claim,
            effect_id,
            tool.name,
            arguments,
        )
        attributes["branchpoint.execution.effect_id"] = effect_id
        attributes["branchpoint.execution.effect_hash"] = receipt.effect_hash

        if not claimed:
            if receipt.status == "succeeded":
                if self.telemetry is not None:
                    self.telemetry.event(
                        "branchpoint.execution.replayed",
                        {
                            "branchpoint.execution.effect_id": effect_id,
                            "branchpoint.execution.effect_hash": receipt.effect_hash,
                            "gen_ai.tool.name": tool.name,
                        },
                    )
                return receipt.result
            if receipt.status == "failed":
                raise PreviousExecutionFailed(
                    f"Effect {effect_id!r} previously failed with "
                    f"{receipt.error_type or 'error'}: {receipt.error_message or ''}"
                )
            raise ExecutionInProgress(
                f"Effect {effect_id!r} is already in_flight. "
                "Reconcile the authoritative world outcome before retrying."
            )

        call_arguments = dict(arguments)
        if tool.idempotency_key_argument:
            call_arguments[tool.idempotency_key_argument] = effect_id
            attributes["branchpoint.execution.downstream_idempotency"] = True
        else:
            attributes["branchpoint.execution.downstream_idempotency"] = False

        try:
            result = await self._invoke_async(tool, call_arguments, attributes)
            await asyncio.to_thread(ledger.complete, effect_id, result)
            if self.telemetry is not None:
                self.telemetry.event(
                    "branchpoint.execution.completed",
                    {
                        "branchpoint.execution.effect_id": effect_id,
                        "branchpoint.execution.effect_hash": receipt.effect_hash,
                        "gen_ai.tool.name": tool.name,
                    },
                )
            return result
        except Exception as exc:
            await asyncio.to_thread(ledger.fail, effect_id, exc)
            raise
