from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from causalrag.agent.actions import DecisionRecord
from causalrag.agent.state import AgentState, Observation
from causalrag.world_model.models import CausalWorldModel, Evidence


@dataclass
class HypothesisProposal:
    hypothesis_id: str
    statement: str
    probability: float = 0.5
    rationale: str = ""
    falsifiers: List[str] = field(default_factory=list)
    experiment_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)


class LLMHypothesisUpdater:
    """Update competing hypotheses from the latest action outcome.

    The model only proposes signed evidence updates. Persistent probability,
    support/conflict history, and status live in CausalWorldModel. When an
    action explicitly declares which hypotheses it tests, updates are scoped to
    those hypotheses so evidence remains traceable to the experiment that
    produced it.
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
        hypotheses = world_model.snapshot().get("hypotheses", [])
        if not hypotheses:
            return

        declared_tests = {
            str(hypothesis_id)
            for hypothesis_id in decision.selected.tests_hypotheses
            if str(hypothesis_id).strip()
        }

        prompt = f"""Evaluate the latest observation against the current competing hypotheses.
Update only hypotheses for which this observation is actually diagnostic.
Use a positive weight for supporting evidence and a negative weight for conflicting/falsifying evidence.
Do not reward a hypothesis merely because the observation is compatible with it; compatibility is weaker than discrimination.
Strong negative weights should be reserved for observations that match an explicit falsifier or clearly contradict the hypothesis.
If the action explicitly lists hypotheses it tests, do not update hypotheses outside that list.

Return ONLY JSON:
{{
  "updates": [
    {{
      "id": "existing hypothesis id",
      "weight": 0.0,
      "statement": "what this observation implies for the hypothesis",
      "kind": "support|conflict|falsifier"
    }}
  ]
}}

Goal: {state.goal}
Selected action: {decision.selected.name}
Action kind: {decision.selected.kind.value}
Hypotheses tested by action: {json.dumps(decision.selected.tests_hypotheses, ensure_ascii=False)}
Falsification target: {decision.selected.falsification_target}
Observation: {json.dumps(observation.result, ensure_ascii=False, default=str)}
Current hypotheses: {json.dumps(hypotheses, ensure_ascii=False, default=str)}
"""
        raw = self.llm.generate(prompt, temperature=0.0, max_tokens=1000, json_mode=True)
        payload = self._parse_json(raw)
        for item in payload.get("updates", [])[: self.max_updates]:
            hypothesis_id = str(item.get("id") or "").strip()
            if not hypothesis_id or world_model.get_hypothesis(hypothesis_id) is None:
                continue
            if declared_tests and hypothesis_id not in declared_tests:
                continue
            try:
                weight = float(item.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
            evidence = Evidence(
                source=observation.action_name,
                statement=str(item.get("statement") or observation.result),
                weight=max(-1.0, min(1.0, weight)),
                kind=str(item.get("kind") or "observation"),
                metadata={
                    "step": decision.step,
                    "tests_hypotheses": list(decision.selected.tests_hypotheses),
                    "falsification_target": decision.selected.falsification_target,
                },
            )
            world_model.update_hypothesis(hypothesis_id, evidence)

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
