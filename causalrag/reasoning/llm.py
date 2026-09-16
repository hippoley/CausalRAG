from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from causalrag.agent.actions import ActionKind, CandidateAction
from causalrag.agent.state import AgentState
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel


class LLMCausalReasoner:
    """Model-backed reasoner that proposes actions but does not execute them."""

    def __init__(self, llm, tools: ToolRegistry, max_candidates: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_candidates = max_candidates
        self._last_uncertainty: Optional[str] = None

    def propose(self, state: AgentState, world_model: CausalWorldModel) -> Sequence[CandidateAction]:
        prompt = self._build_prompt(state, world_model)
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=1400, json_mode=True)
        payload = self._parse_json(raw)
        self._last_uncertainty = payload.get("uncertainty")

        candidates: List[CandidateAction] = []
        for item in payload.get("candidates", [])[: self.max_candidates]:
            try:
                kind = ActionKind(str(item.get("kind", "stop")))
                name = str(item.get("name") or kind.value)
                if kind not in (ActionKind.STOP, ActionKind.WAIT) and name not in self.tools.specs():
                    continue
                candidates.append(
                    CandidateAction(
                        kind=kind,
                        name=name,
                        arguments=dict(item.get("arguments") or {}),
                        expected_goal_gain=float(item.get("expected_goal_gain", 0.0)),
                        expected_information_gain=float(item.get("expected_information_gain", 0.0)),
                        cost=float(item.get("cost", 0.0)),
                        risk=float(item.get("risk", 0.0)),
                        irreversibility=float(item.get("irreversibility", 0.0)),
                        rationale=str(item.get("rationale") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue

        if not candidates:
            candidates.append(
                CandidateAction(
                    kind=ActionKind.STOP,
                    name="stop",
                    arguments={"answer": payload.get("answer", "Unable to identify a safe useful action.")},
                    rationale="No valid candidate action was produced.",
                )
            )
        return candidates

    def uncertainty(self, state: AgentState, world_model: CausalWorldModel) -> Optional[str]:
        return self._last_uncertainty

    def _build_prompt(self, state: AgentState, world_model: CausalWorldModel) -> str:
        tools: List[Dict[str, Any]] = []
        for name, spec in self.tools.specs().items():
            tools.append(
                {
                    "name": name,
                    "description": spec.description,
                    "cost": spec.cost,
                    "risk": spec.risk,
                    "reversible": spec.reversible,
                    "metadata": spec.metadata,
                }
            )

        observations = [
            {"action": obs.action_name, "result": obs.result}
            for obs in state.observations[-6:]
        ]
        context = {
            "goal": state.goal,
            "step": state.step,
            "budget_remaining": state.budget_remaining(),
            "beliefs": world_model.snapshot(),
            "recent_observations": observations,
            "available_tools": tools,
        }
        return f"""You are the decision proposer inside a causal agent runtime.
You do NOT execute tools. Propose up to {self.max_candidates} candidate actions.

Optimize for goal progress and information gain while minimizing cost, risk, and irreversible commitments.
Prefer observing/retrieving evidence when uncertainty is high. Use intervene only when evidence is sufficient.
WAIT is valid when an earlier intervention needs time to produce an observable effect.
STOP when the goal can be answered/completed from current evidence; include the final user-facing answer in arguments.answer.
Never invent a tool name. Non-WAIT/non-STOP actions must use one of available_tools.

Return ONLY a JSON object of this exact shape:
{{
  "uncertainty": "largest decision-relevant uncertainty or null",
  "candidates": [
    {{
      "kind": "observe|retrieve|ask|intervene|wait|stop",
      "name": "tool name, wait, or stop",
      "arguments": {{}},
      "expected_goal_gain": 0.0,
      "expected_information_gain": 0.0,
      "cost": 0.0,
      "risk": 0.0,
      "irreversibility": 0.0,
      "rationale": "short reason"
    }}
  ],
  "answer": "optional fallback final answer"
}}

STATE:
{json.dumps(context, ensure_ascii=False, default=str)}
"""

    @staticmethod
    def _parse_json(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
