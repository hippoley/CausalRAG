from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from causalrag.agent.actions import ActionKind, ActionScore, CandidateAction
from causalrag.world_model.models import CausalWorldModel, Hypothesis


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def hypothesis_discrimination_score(
    action: CandidateAction,
    world_model: Optional[CausalWorldModel],
) -> Optional[float]:
    """Estimate how much an action can discriminate among explicit hypotheses.

    This is deliberately a deterministic baseline rather than a claim of true
    expected information gain. It uses only runtime-owned hypothesis state and
    action test metadata:

    - coverage: how much active hypothesis credence the action tests;
    - ambiguity: normalized entropy among the tested hypotheses;
    - falsification leverage: whether the action challenges a high-credence
      hypothesis explicitly named as its falsification target.

    Returns ``None`` only when the action did not declare hypothesis tests. If
    it declared tests but none resolve to active hypotheses, the score is 0.0
    so the runtime does not fall back to an ungrounded model self-score.
    """
    if not action.tests_hypotheses:
        return None
    if world_model is None:
        return 0.0

    active = [
        hypothesis
        for hypothesis in world_model.hypotheses(include_rejected=False)
        if hypothesis.status != "rejected"
    ]
    if not active:
        return 0.0

    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in active}
    tested: List[Hypothesis] = []
    seen = set()
    for hypothesis_id in action.tests_hypotheses:
        hypothesis_id = str(hypothesis_id)
        if hypothesis_id in seen:
            continue
        hypothesis = by_id.get(hypothesis_id)
        if hypothesis is not None:
            tested.append(hypothesis)
            seen.add(hypothesis_id)

    if not tested:
        return 0.0

    total_mass = sum(max(0.001, hypothesis.probability) for hypothesis in active)
    tested_mass = sum(max(0.001, hypothesis.probability) for hypothesis in tested)
    coverage = min(1.0, tested_mass / total_mass) if total_mass > 0 else 0.0

    discrimination = 0.0
    if len(tested) >= 2 and tested_mass > 0:
        distribution = [
            max(0.001, hypothesis.probability) / tested_mass
            for hypothesis in tested
        ]
        entropy = -sum(p * math.log(p) for p in distribution if p > 0)
        max_entropy = math.log(len(distribution))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        discrimination = coverage * normalized_entropy

    falsification_leverage = 0.0
    if action.falsification_target:
        target = by_id.get(str(action.falsification_target))
        if target is not None and target in tested:
            falsification_leverage = coverage * _clamp01(target.probability)

    return _clamp01(max(discrimination, falsification_leverage))


def score_action(
    action: CandidateAction,
    world_model: Optional[CausalWorldModel] = None,
    candidate_index: int = 0,
) -> ActionScore:
    """Score one candidate using runtime evidence when it is available."""
    model_information_gain = _clamp01(action.expected_information_gain)
    discrimination = hypothesis_discrimination_score(action, world_model)

    if discrimination is None:
        information_gain = model_information_gain
        information_source = "model_estimate"
    else:
        information_gain = discrimination
        information_source = "runtime_hypothesis_discrimination"

    goal_gain = float(action.expected_goal_gain)
    total_utility = (
        goal_gain
        + information_gain
        - float(action.cost)
        - float(action.risk)
        - float(action.irreversibility)
    )

    return ActionScore(
        candidate_index=candidate_index,
        action_name=action.name,
        action_kind=action.kind,
        total_utility=total_utility,
        goal_gain=goal_gain,
        information_gain=information_gain,
        information_source=information_source,
        model_information_gain=model_information_gain,
        discrimination_score=discrimination,
        cost=float(action.cost),
        risk=float(action.risk),
        irreversibility=float(action.irreversibility),
    )


def rank_actions(
    candidates: Sequence[CandidateAction],
    world_model: Optional[CausalWorldModel] = None,
) -> List[Tuple[CandidateAction, ActionScore]]:
    """Return candidates ordered by the runtime score used for selection."""
    ranked = [
        (
            action,
            score_action(action, world_model=world_model, candidate_index=index),
        )
        for index, action in enumerate(candidates)
    ]
    return sorted(ranked, key=lambda pair: pair[1].total_utility, reverse=True)


def select_action(
    candidates: Sequence[CandidateAction],
    world_model: Optional[CausalWorldModel] = None,
) -> CandidateAction:
    """Select an action using runtime causal/epistemic utility."""
    if not candidates:
        return CandidateAction(
            kind=ActionKind.STOP,
            name="stop",
            rationale="No valid candidate actions were proposed.",
        )
    return rank_actions(candidates, world_model=world_model)[0][0]
