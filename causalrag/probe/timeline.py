from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def _hypotheses(snapshot: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not snapshot:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in snapshot.get("hypotheses", []) or []:
        hypothesis_id = str(row.get("id") or row.get("hypothesis_id") or "").strip()
        if hypothesis_id:
            result[hypothesis_id] = dict(row)
    return result


def hypothesis_delta(
    before: Optional[Mapping[str, Any]],
    after: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return per-hypothesis probability/state movement between two snapshots."""

    left = _hypotheses(before)
    right = _hypotheses(after)
    result: Dict[str, Dict[str, Any]] = {}
    for hypothesis_id in sorted(set(left) | set(right)):
        a = left.get(hypothesis_id)
        b = right.get(hypothesis_id)
        p0 = None if a is None else a.get("probability")
        p1 = None if b is None else b.get("probability")
        delta = None
        if p0 is not None and p1 is not None:
            try:
                delta = float(p1) - float(p0)
            except (TypeError, ValueError):
                delta = None
        result[hypothesis_id] = {
            "before": p0,
            "after": p1,
            "delta": delta,
            "status_before": None if a is None else a.get("status"),
            "status_after": None if b is None else b.get("status"),
            "origin_before": None if a is None else a.get("origin"),
            "origin_after": None if b is None else b.get("origin"),
            "validated_before": None if a is None else a.get("validated"),
            "validated_after": None if b is None else b.get("validated"),
            "created": a is None and b is not None,
            "removed": a is not None and b is None,
        }
    return result


def _next_world_after(
    gates: List[Dict[str, Any]],
    index: int,
    *,
    executed: bool,
    final_world: Mapping[str, Any],
) -> Mapping[str, Any]:
    current_step = int(gates[index].get("step", 0))
    if executed:
        # Replans can create several previews at the same state step. The first
        # gate with a larger step is the post-action world.
        for candidate in gates[index + 1 :]:
            if int(candidate.get("step", current_step)) > current_step:
                return candidate.get("world_before") or final_world
        return final_world

    # A discarded preview did not execute a tool. Its successor may be another
    # gate at the same step after a human/world-model mutation.
    if index + 1 < len(gates):
        return gates[index + 1].get("world_before") or final_world
    return final_world


def _next_observation(
    observations: List[Dict[str, Any]],
    cursor: int,
    action_name: Optional[str],
):
    for index in range(cursor, len(observations)):
        row = observations[index]
        if action_name is None or row.get("action_name") == action_name:
            return row, index + 1
    return None, cursor


def _next_transition(
    transitions: List[Dict[str, Any]],
    cursor: int,
    action_name: Optional[str],
):
    for index in range(cursor, len(transitions)):
        row = transitions[index]
        if row.get("action") == "missed_observation_window":
            continue
        if action_name is None or row.get("action") == action_name:
            return row, index + 1
    return None, cursor


def build_episode_timeline(
    gate_history: Iterable[Mapping[str, Any]],
    result_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build semantic, replayable frames from live gates + canonical runtime state.

    A frame represents what the proposer/runtime/human knew before execution,
    the actual canonical decision (if any), the environment observation, and the
    world-model change visible by the next decision point.

    Discarded replans remain as first-class frames and consume no canonical
    decision/observation/transition.
    """

    gates = [dict(row) for row in gate_history]
    decisions = [dict(row) for row in (result_payload.get("decisions") or [])]
    observations = [dict(row) for row in (result_payload.get("observations") or [])]
    transitions = [dict(row) for row in (result_payload.get("transitions") or [])]
    final_world = dict(result_payload.get("beliefs") or {})
    if not final_world:
        final_world = {
            "hypotheses": list(result_payload.get("hypotheses") or []),
            "open_world": dict(result_payload.get("open_world") or {}),
        }

    frames: List[Dict[str, Any]] = []
    decision_cursor = 0
    observation_cursor = 0
    transition_cursor = 0

    for gate_index, gate in enumerate(gates):
        status = str(gate.get("status") or "waiting")
        discarded = status == "discarded_before_execution"
        executed = not discarded and status != "waiting"

        decision = None
        selected = None
        observation = None
        transition = None

        if executed and decision_cursor < len(decisions):
            decision = decisions[decision_cursor]
            decision_cursor += 1
            selected = dict(decision.get("selected") or {})
            kind = str(selected.get("kind") or "")
            action_name = selected.get("name")

            if kind != "stop":
                observation, observation_cursor = _next_observation(
                    observations,
                    observation_cursor,
                    action_name,
                )
                transition, transition_cursor = _next_transition(
                    transitions,
                    transition_cursor,
                    action_name,
                )

        world_before = dict(gate.get("world_before") or {})
        world_after = dict(
            _next_world_after(
                gates,
                gate_index,
                executed=executed,
                final_world=final_world,
            )
            or {}
        )
        effects = {} if transition is None else dict(transition.get("expected_effects") or {})

        frames.append(
            {
                "frame_index": gate_index,
                "frame_id": gate.get("gate_id"),
                "step": gate.get("step"),
                "status": status,
                "executed": bool(executed),
                "model": {
                    "uncertainty": gate.get("uncertainty"),
                    "hypothesis_proposals": list(gate.get("hypothesis_proposals") or []),
                    "candidates": list(gate.get("candidates") or []),
                },
                "runtime": {
                    "selected_before_human": dict(gate.get("runtime_selected") or {}),
                    "action_scores": list(gate.get("action_scores") or []),
                },
                "human": gate.get("human_intervention"),
                "operator_messages": list(gate.get("operator_messages") or []),
                "actual": {
                    "selected": selected,
                    "observation": observation,
                    "transition": transition,
                },
                "world_before": world_before,
                "world_after": world_after,
                "hypothesis_delta": hypothesis_delta(world_before, world_after),
                "causal_effects": effects,
            }
        )

    return frames
