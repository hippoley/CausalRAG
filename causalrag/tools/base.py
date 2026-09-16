from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    risk: float = 0.0
    cost: float = 0.0
    reversible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[ToolSpec]] = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
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
        return tool.handler(**arguments)
