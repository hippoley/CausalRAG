from __future__ import annotations

import json
from typing import Any, Dict, List

from branchpoint.agent.actions import DecisionRecord
from branchpoint.agent.state import AgentState, Observation
from branchpoint.world_model.models import CausalWorldModel, Evidence


class LLMBeliefUpdater:
    """Turn new observations into defeasible causal-belief updates.

    The model proposes updates; CausalWorldModel owns the persistent belief
    state and applies bounded updates. Retrieved statements are treated as
    evidence, not as ground-truth interventions.
    """

    def __init__(self, llm, max_updates: int = 6) -> None:
        self.llm = llm
        self.max_updates = max_updates

    def __call__(
        self,
        state: AgentState,
        world_model: CausalWorldModel,
        decision: DecisionRecord,
        observation: Observation,
    ) -> None:
        prompt = f"""Extract decision-relevant directional causal claims from the latest observation.
Do not convert mere co-occurrence into causation. If evidence contradicts a causal claim, use a negative weight.
For retrieved text, keep weights conservative because it is reported evidence rather than an intervention performed by this agent.

Return ONLY JSON:
{{
  "updates": [
    {{
      "cause": "specific variable/event",
      "effect": "specific variable/outcome",
      "prior": 0.5,
      "weight": 0.0,
      "statement": "short evidence statement",
      "kind": "retrieved|observation|intervention|counterevidence",
      "mechanism": "optional mechanism or null",
      "temporal_lag": "optional lag or null"
    }}
  ]
}}

Goal: {state.goal}
Selected action: {decision.selected.name}
Action kind: {decision.selected.kind.value}
Observation: {json.dumps(observation.result, ensure_ascii=False, default=str)}
Current beliefs: {json.dumps(world_model.snapshot(), ensure_ascii=False, default=str)}
"""
        raw = self.llm.generate(prompt, temperature=0.0, max_tokens=1200, json_mode=True)
        payload = self._parse_json(raw)
        for item in payload.get("updates", [])[: self.max_updates]:
            cause = str(item.get("cause") or "").strip()
            effect = str(item.get("effect") or "").strip()
            if not cause or not effect or cause == effect:
                continue
            try:
                prior = float(item.get("prior", 0.5))
                weight = float(item.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
            evidence = Evidence(
                source=observation.action_name,
                statement=str(item.get("statement") or observation.result),
                weight=max(-1.0, min(1.0, weight)),
                kind=str(item.get("kind") or "observation"),
                metadata={"step": decision.step},
            )
            belief = world_model.update_belief(cause, effect, evidence=evidence, prior=prior)
            if item.get("mechanism"):
                belief.mechanism = str(item["mechanism"])
            if item.get("temporal_lag"):
                belief.temporal_lag = str(item["temporal_lag"])

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
