from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from causalrag.agent.temporal import TemporalEffectContract
    from causalrag.experiments import (
        DecisionPreferences,
        ExperimentContract,
        InterventionContract,
    )
    from causalrag.observability import CausalTelemetry


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


class ToolRegistry:
    def __init__(
        self,
        tools: Optional[Iterable[ToolSpec]] = None,
        decision_preferences: Optional["DecisionPreferences"] = None,
        telemetry: Optional["CausalTelemetry"] = None,
    ) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.decision_preferences = decision_preferences
        self.telemetry = telemetry
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

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get(name)
        if self.telemetry is None:
            return tool.handler(**arguments)

        attributes: Dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "causalrag.tool.kind": tool.metadata.get("kind", "tool"),
            "causalrag.tool.cost": float(tool.cost),
            "causalrag.tool.risk": float(tool.risk),
            "causalrag.tool.reversible": bool(tool.reversible),
            "causalrag.tool.argument_names": sorted(str(key) for key in arguments),
        }
        if self.telemetry.capture_content:
            attributes["causalrag.tool.arguments"] = dict(arguments)

        with self.telemetry.span(f"execute_tool {tool.name}", attributes) as span:
            result = tool.handler(**arguments)
            if self.telemetry.capture_content:
                span.set_attribute("causalrag.tool.result", result)
            return result
