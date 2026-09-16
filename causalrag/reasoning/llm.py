from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from causalrag.agent.actions import ActionKind, CandidateAction
from causalrag.agent.state import AgentState
from causalrag.reasoning.hypothesis import HypothesisProposal
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel


class LLMCausalReasoner:
    """Model-backed reasoner that proposes hypotheses and actions, but executes neither."""

    def __init__(self, llm, tools: ToolRegistry, max_candidates: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_candidates = max_candidates
        self._last_uncertainty: Optional[str] = None
        self._last_hypotheses: List[HypothesisProposal] = []

    def propose(self, state: AgentState, world_model: CausalWorldModel) -> Sequence[CandidateAction]:
        prompt = self._build_prompt(state, world_model)
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=1800, json_mode=True)
        payload = self._parse_json(raw)
        self._last_uncertainty = payload.get("uncertainty")
        self._last_hypotheses = self._parse_hypotheses(payload)

        known_hypothesis_ids = {
            hypothesis.hypothesis_id for hypothesis in world_model.hypotheses()
        }
        known_hypothesis_ids.update(
            hypothesis.hypothesis_id for hypothesis in self._last_hypotheses
        )

        candidates: List[CandidateAction] = []
        specs = self.tools.specs()
        for item in payload.get("candidates", [])[: self.max_candidates]:
            try:
                requested_kind = ActionKind(str(item.get("kind", "stop")))
                name = str(item.get("name") or requested_kind.value)
                kind = requested_kind
                cost = float(item.get("cost", 0.0))
                risk = float(item.get("risk", 0.0))
                irreversibility = float(item.get("irreversibility", 0.0))

                if kind not in (ActionKind.STOP, ActionKind.WAIT):
                    if name not in specs:
                        continue
                    spec = specs[name]
                    declared_kind = spec.metadata.get("kind")
                    if declared_kind in {member.value for member in ActionKind}:
                        kind = ActionKind(declared_kind)
                    cost = max(cost, float(spec.cost))
                    risk = max(risk, float(spec.risk))
                    if not spec.reversible:
                        irreversibility = max(irreversibility, 1.0)

                tests_hypotheses = [
                    str(hypothesis_id)
                    for hypothesis_id in (item.get("tests_hypotheses") or [])
                    if str(hypothesis_id) in known_hypothesis_ids
                ]
                falsification_target = item.get("falsification_target")
                if falsification_target is not None:
                    falsification_target = str(falsification_target)
                    if falsification_target not in known_hypothesis_ids:
                        falsification_target = None

                candidates.append(
                    CandidateAction(
                        kind=kind,
                        name=name,
                        arguments=dict(item.get("arguments") or {}),
                        expected_goal_gain=float(item.get("expected_goal_gain", 0.0)),
                        expected_information_gain=float(item.get("expected_information_gain", 0.0)),
                        cost=cost,
                        risk=risk,
                        irreversibility=irreversibility,
                        rationale=str(item.get("rationale") or ""),
                        tests_hypotheses=tests_hypotheses,
                        falsification_target=falsification_target,
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

    def hypothesis_proposals(
        self,
        state: AgentState,
        world_model: CausalWorldModel,
    ) -> Sequence[HypothesisProposal]:
        return list(self._last_hypotheses)

    def _parse_hypotheses(self, payload: Dict[str, Any]) -> List[HypothesisProposal]:
        proposals: List[HypothesisProposal] = []
        seen = set()
        for item in payload.get("hypotheses", [])[:6]:
            hypothesis_id = str(item.get("id") or "").strip()
            statement = str(item.get("statement") or "").strip()
            if not hypothesis_id or not statement or hypothesis_id in seen:
                continue
            try:
                probability = float(item.get("probability", 0.5))
            except (TypeError, ValueError):
                probability = 0.5
            proposals.append(
                HypothesisProposal(
                    hypothesis_id=hypothesis_id,
                    statement=statement,
                    probability=max(0.001, min(0.999, probability)),
                    rationale=str(item.get("rationale") or ""),
                    falsifiers=[
                        str(value)
                        for value in (item.get("falsifiers") or [])
                        if str(value).strip()
                    ],
                )
            )
            seen.add(hypothesis_id)
        return proposals

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
            "world_model": world_model.snapshot(),
            "recent_observations": observations,
            "available_tools": tools,
        }
        return f"""You are the decision proposer inside a causal agent runtime.
You do NOT execute tools and you do NOT get final authority over actions.

Maintain explicit competing hypotheses when there is unresolved causal uncertainty.
- Prefer 2-4 genuinely competing explanations rather than one favored story.
- Reuse existing hypothesis IDs from world_model when the explanation is unchanged.
- Every hypothesis must include concrete falsifiers: observations that would count against it.
- Do not raise a hypothesis probability merely because evidence is compatible with it.
- Prefer actions whose possible outcomes discriminate among hypotheses or could falsify the current leader.

Propose up to {self.max_candidates} candidate actions.
Optimize for goal progress and information gain while minimizing cost, risk, and irreversible commitments.
Prefer observing/retrieving evidence when uncertainty is high. Use intervene only when evidence is sufficient.
WAIT is valid when an earlier intervention needs time to produce an observable effect.
STOP when the goal can be answered/completed from current evidence; include the final user-facing answer in arguments.answer.
Never invent a tool name. Non-WAIT/non-STOP actions must use one of available_tools.
Do not try to lower a tool's declared cost/risk in order to make it win selection; capability metadata is enforced outside the model.

For every information-seeking action, list the hypothesis IDs it tests. Set falsification_target when the action is specifically useful because an outcome could disconfirm one hypothesis.

Return ONLY a JSON object of this exact shape:
{{
  "uncertainty": "largest decision-relevant uncertainty or null",
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "specific falsifiable explanation",
      "probability": 0.5,
      "rationale": "why it remains plausible",
      "falsifiers": ["concrete observation that would count against it"]
    }}
  ],
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
      "tests_hypotheses": ["H1", "H2"],
      "falsification_target": "H1 or null",
      "rationale": "short reason including how outcomes would change the decision"
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
