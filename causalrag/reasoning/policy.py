from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from causalrag.agent.actions import ActionKind, ActionScore, CandidateAction
from causalrag.experiments import (
    contract_applicable,
    expected_information_gain,
    experiment_decision_value,
    intervention_value,
)
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel, Hypothesis


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _has_active_hypotheses(world_model: Optional[CausalWorldModel]) -> bool:
    return bool(world_model and world_model.hypotheses(include_rejected=False))


def hypothesis_discrimination_score(action: CandidateAction, world_model: Optional[CausalWorldModel]) -> Optional[float]:
    if not action.tests_hypotheses:
        return None
    if world_model is None:
        return 0.0
    active = [h for h in world_model.hypotheses(include_rejected=False) if h.status != "rejected"]
    if not active:
        return 0.0
    by_id = {h.hypothesis_id: h for h in active}
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
    total_mass = sum(max(0.001, h.probability) for h in active)
    tested_mass = sum(max(0.001, h.probability) for h in tested)
    coverage = min(1.0, tested_mass / total_mass) if total_mass > 0 else 0.0
    discrimination = 0.0
    if len(tested) >= 2 and tested_mass > 0:
        distribution = [max(0.001, h.probability) / tested_mass for h in tested]
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


def _tool_spec_for_action(action: CandidateAction, tools: Optional[ToolRegistry]):
    if tools is None or action.kind in (ActionKind.STOP, ActionKind.WAIT):
        return None
    try:
        return tools.get(action.name)
    except KeyError:
        return None


def score_action(action: CandidateAction, world_model: Optional[CausalWorldModel] = None, candidate_index: int = 0, tools: Optional[ToolRegistry] = None) -> ActionScore:
    model_information_gain = _clamp01(action.expected_information_gain)
    discrimination = hypothesis_discrimination_score(action, world_model)
    bayesian_information_gain = None
    tool_spec = _tool_spec_for_action(action, tools)
    experiment_contract = None if tool_spec is None else tool_spec.experiment_contract
    intervention_contract = None if tool_spec is None else tool_spec.intervention_contract

    if experiment_contract is not None and world_model is not None and contract_applicable(experiment_contract, world_model):
        bayesian_information_gain = expected_information_gain(experiment_contract, world_model)
        information_gain = bayesian_information_gain
        information_source = "runtime_bayesian_eig"
    elif discrimination is not None:
        information_gain = discrimination
        information_source = "runtime_hypothesis_discrimination"
    elif _has_active_hypotheses(world_model) and action.kind in (ActionKind.OBSERVE, ActionKind.RETRIEVE, ActionKind.ASK):
        information_gain = 0.0
        information_source = "unanchored_model_estimate"
    else:
        information_gain = model_information_gain
        information_source = "model_estimate"

    goal_gain = _clamp01(action.expected_goal_gain)
    cost = float(action.cost)
    risk = float(action.risk)
    irreversibility = float(action.irreversibility)
    if tool_spec is not None:
        cost = max(cost, float(tool_spec.cost))
        risk = max(risk, float(tool_spec.risk))
        if not tool_spec.reversible:
            irreversibility = max(irreversibility, 1.0)

    decision_value = None
    decision_value_source = None
    evsi = None
    net_sampling = None

    if world_model is not None and tools is not None and intervention_contract is not None:
        intervention = intervention_value(
            action.name,
            intervention_contract,
            world_model,
            capability_cost=cost,
        )
        if intervention is not None:
            decision_value = intervention.net_value - risk - irreversibility
            decision_value_source = "runtime_expected_intervention_utility"
    elif world_model is not None and tools is not None and experiment_contract is not None:
        experiment_value = experiment_decision_value(
            action.name,
            experiment_contract,
            world_model,
            tools,
            experiment_cost=cost,
        )
        if experiment_value is not None:
            decision_value = (
                experiment_value.expected_best_value_after
                - experiment_value.experiment_cost
                - risk
                - irreversibility
            )
            decision_value_source = "runtime_expected_decision_value_after_sampling"
            evsi = experiment_value.evsi
            net_sampling = experiment_value.net_value_of_sampling

    if decision_value is not None:
        total_utility = decision_value
    else:
        total_utility = goal_gain + information_gain - cost - risk - irreversibility

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
        bayesian_information_gain=bayesian_information_gain,
        cost=cost,
        risk=risk,
        irreversibility=irreversibility,
        decision_value=decision_value,
        decision_value_source=decision_value_source,
        expected_value_of_sample_information=evsi,
        net_value_of_sampling=net_sampling,
    )


def rank_actions(candidates: Sequence[CandidateAction], world_model: Optional[CausalWorldModel] = None, tools: Optional[ToolRegistry] = None) -> List[Tuple[CandidateAction, ActionScore]]:
    ranked = [(action, score_action(action, world_model=world_model, candidate_index=index, tools=tools)) for index, action in enumerate(candidates)]
    return sorted(ranked, key=lambda pair: pair[1].total_utility, reverse=True)


def select_action(candidates: Sequence[CandidateAction], world_model: Optional[CausalWorldModel] = None, tools: Optional[ToolRegistry] = None) -> CandidateAction:
    if not candidates:
        return CandidateAction(kind=ActionKind.STOP, name="stop", rationale="No valid candidate actions were proposed.")
    return rank_actions(candidates, world_model=world_model, tools=tools)[0][0]
