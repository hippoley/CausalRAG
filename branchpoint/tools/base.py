from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from branchpoint.agent.temporal import TemporalEffectContract
    from branchpoint.experiments import (
        DecisionPreferences,
        ExperimentContract,
        InterventionContract,
    )
    from branchpoint.observability import CausalTelemetry
    from branchpoint.execution import SQLiteExecutionLedger


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


class ToolRegistry:
    def __init__(
        self,
        tools: Optional[Iterable[ToolSpec]] = None,
        decision_preferences: Optional["DecisionPreferences"] = None,
        telemetry: Optional["CausalTelemetry"] = None,
        execution_ledger: Optional["SQLiteExecutionLedger"] = None,
    ) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.decision_preferences = decision_preferences
        self.telemetry = telemetry
        self.execution_ledger = execution_ledger
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
    ) -> Any:
        tool = self.get(name)
        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "branchpoint.tool.kind": tool.metadata.get("kind", "tool"),
            "branchpoint.tool.cost": float(tool.cost),
            "branchpoint.tool.risk": float(tool.risk),
            "branchpoint.tool.reversible": bool(tool.reversible),
            "branchpoint.tool.argument_names": sorted(str(key) for key in arguments),
        }

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

        from branchpoint.execution import ExecutionInProgress, PreviousExecutionFailed

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
            existing = call_arguments.get(tool.idempotency_key_argument)
            if existing is not None and str(existing) != str(effect_id):
                from branchpoint.execution import EffectIdentityConflict
                raise EffectIdentityConflict(
                    f"Argument {tool.idempotency_key_argument!r} conflicts with effect_id."
                )
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
