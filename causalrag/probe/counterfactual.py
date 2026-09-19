from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping

from causalrag.agent import ActionKind, CandidateAction, create_ablation_agent
from causalrag.observability import CausalTelemetry

from .runtime import ProbeRunConfig
from .scenarios import build_scenario_runtime, scenario_metrics


class ScriptedForkReasoner:
    """Replay a frozen prefix, then inject one counterfactual candidate.

    Runtime arbitration and temporal guards remain active. Only proposer choice
    is scripted so the branch stays attributable to the changed decision.
    """

    def __init__(self, actions: Iterable[CandidateAction]) -> None:
        self.actions = list(actions)

    def propose(self, state, world_model):
        index = int(state.step)
        if index < len(self.actions):
            return [self.actions[index]]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "counterfactual_branch_complete"},
                rationale="Stop after the requested one-step counterfactual branch.",
            )
        ]

    def hypothesis_proposals(self, state, world_model):
        return []

    def uncertainty(self, state, world_model):
        return "counterfactual replay"


def candidate_from_payload(payload: Mapping[str, Any]) -> CandidateAction:
    return CandidateAction(
        kind=ActionKind(str(payload.get("kind"))),
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        expected_goal_gain=float(payload.get("expected_goal_gain", 0.0) or 0.0),
        expected_information_gain=float(payload.get("expected_information_gain", 0.0) or 0.0),
        cost=float(payload.get("cost", 0.0) or 0.0),
        risk=float(payload.get("risk", 0.0) or 0.0),
        irreversibility=float(payload.get("irreversibility", 0.0) or 0.0),
        rationale=str(payload.get("rationale") or ""),
        tests_hypotheses=[
            str(value) for value in (payload.get("tests_hypotheses") or [])
        ],
        falsification_target=payload.get("falsification_target"),
    )


def counterfactual_support(config: ProbeRunConfig) -> Dict[str, Any]:
    if config.outcome_mode == "deterministic":
        return {
            "available": True,
            "replay_mode": "deterministic_exact_rebuild",
            "paired_randomness": "not_applicable",
        }
    if config.stochastic_coupling == "action_indexed":
        return {
            "available": True,
            "replay_mode": "action_indexed_common_random_numbers",
            "paired_randomness": "seed + experiment_id + occurrence",
        }
    return {
        "available": False,
        "replay_mode": "unsupported_sequence_randomness",
        "paired_randomness": None,
        "reason": (
            "A stochastic sequence-coupled session cannot support a fair frozen-step "
            "fork because changing the action also changes which random draw is consumed. "
            "Re-run the session with stochastic_coupling='action_indexed'."
        ),
    }


def run_one_step_counterfactual(
    config: ProbeRunConfig,
    ledger: list[Dict[str, Any]],
    *,
    step: int,
    candidate_index: int,
) -> Dict[str, Any]:
    support = counterfactual_support(config)
    if not support.get("available"):
        return {**support, "step": int(step), "candidate_index": int(candidate_index)}

    target = next(
        (row for row in ledger if int(row.get("step", -1)) == int(step)),
        None,
    )
    if target is None:
        raise KeyError(f"step not found: {step}")

    candidates = target.get("candidates") or []
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError("candidate_index out of range")
    alternative_payload = candidates[candidate_index]
    if not bool(alternative_payload.get("runtime_valid", True)):
        raise ValueError("candidate was rejected by runtime validation")

    actual = target.get("selected") or {}
    if (
        str(actual.get("name")) == str(alternative_payload.get("name"))
        and str(actual.get("kind")) == str(alternative_payload.get("kind"))
        and dict(actual.get("arguments") or {}) == dict(alternative_payload.get("arguments") or {})
    ):
        raise ValueError("candidate is the actual selected action, not a counterfactual")

    replay_config = config
    if config.outcome_mode == "stochastic":
        replay_config = replace(config, stochastic_coupling="action_indexed")

    prefix_rows = sorted(
        [row for row in ledger if int(row.get("step", -1)) < int(step)],
        key=lambda row: int(row.get("step", -1)),
    )
    scripted = [
        candidate_from_payload(row.get("selected") or {})
        for row in prefix_rows
    ]
    scripted.append(candidate_from_payload(alternative_payload))

    scenario = build_scenario_runtime(replay_config)
    telemetry = CausalTelemetry(capture_content=False)
    kwargs: Dict[str, Any] = {
        "reasoner": ScriptedForkReasoner(scripted),
        "world_model": scenario.world_model,
        "tools": scenario.tools,
        "capabilities": replay_config.resolved_capabilities(),
        "telemetry": telemetry,
    }
    if scenario.time_driver is not None:
        kwargs["time_driver"] = scenario.time_driver
    if scenario.mismatch_policy is not None:
        kwargs["mismatch_policy"] = scenario.mismatch_policy

    agent = create_ablation_agent(**kwargs)
    result = agent.run(
        str(replay_config.goal or scenario.goal),
        max_steps=max(int(step) + 2, len(scripted) + 1),
    )
    payload = result.to_dict()
    metrics = scenario_metrics(scenario.environment, result)

    branch_index = min(int(step), len(payload.get("decisions") or []) - 1)
    branch_decision = (
        (payload.get("decisions") or [])[branch_index]
        if branch_index >= 0
        else None
    )
    branch_observation = (
        (payload.get("observations") or [])[branch_index]
        if branch_index >= 0 and branch_index < len(payload.get("observations") or [])
        else None
    )
    branch_transition = (
        (payload.get("transitions") or [])[branch_index]
        if branch_index >= 0 and branch_index < len(payload.get("transitions") or [])
        else None
    )

    actual_after = {
        str(row.get("id") or row.get("hypothesis_id")): float(row.get("probability", 0.0))
        for row in (target.get("posterior") or [])
    }
    branch_after = {
        str(row.get("id") or row.get("hypothesis_id")): float(row.get("probability", 0.0))
        for row in (payload.get("hypotheses") or [])
    }
    posterior_delta_vs_actual = {
        hypothesis_id: branch_after.get(hypothesis_id, 0.0) - probability
        for hypothesis_id, probability in actual_after.items()
        if hypothesis_id in branch_after
    }

    return {
        **support,
        "step": int(step),
        "candidate_index": int(candidate_index),
        "actual": {
            "selected": actual,
            "observation": target.get("observation"),
            "posterior": target.get("posterior"),
        },
        "counterfactual": {
            "requested_candidate": alternative_payload,
            "runtime_effective_decision": branch_decision,
            "observation": branch_observation,
            "transition": branch_transition,
            "posterior": payload.get("hypotheses") or [],
            "stop_reason": payload.get("stop_reason"),
            "metrics": metrics,
            "trace_id": payload.get("trace_id"),
        },
        "posterior_delta_counterfactual_minus_actual": posterior_delta_vs_actual,
        "prefix_replayed_steps": len(prefix_rows),
        "truthfulness": {
            "environment_rebuilt_from_same_config": True,
            "runtime_guards_reapplied": True,
            "prefix_actions_replayed": True,
            "front_end_simulation": False,
        },
    }
