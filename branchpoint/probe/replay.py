from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


SEMANTIC_REPLAY_SCHEMA = "branchpoint.semantic-replay.v1"


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value


def build_semantic_replay(
    gate_history: Iterable[Mapping[str, Any]],
    episode_ledger: Iterable[Mapping[str, Any]],
    *,
    final_decisions: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Align every decision preview with the canonical outcome, if any.

    The episode ledger remains the source of truth for executed actions.
    Gate history may contain multiple previews for one runtime step because
    a human can mutate context and request replanning before any tool runs.
    """

    previews = [_copy(dict(row)) for row in gate_history]
    ledger = [_copy(dict(row)) for row in episode_ledger]
    decisions = [_copy(dict(row)) for row in final_decisions]

    ledger_cursor = 0
    decision_cursor = 0
    frames = []

    for index, preview in enumerate(previews):
        status = str(preview.get("status") or "unknown")
        step = int(preview.get("step", -1))
        frame: Dict[str, Any] = {
            "frame_index": index,
            "gate_id": preview.get("gate_id"),
            "step": step,
            "preview_status": status,
            "proposer": preview.get("proposer"),
            "proposer_attempts": preview.get("proposer_attempts") or [],
            "hypothesis_proposals": preview.get("hypothesis_proposals") or [],
            "runtime_selected": preview.get("runtime_selected"),
            "candidates": preview.get("candidates") or [],
            "action_scores": preview.get("action_scores") or [],
            "score_provenance": preview.get("score_provenance") or {},
            "decision_inspector": preview.get("decision_inspector") or {},
            "world_before": preview.get("world_before") or {
                "hypotheses": preview.get("hypotheses") or []
            },
            "world_after_gate": preview.get("world_after_gate"),
            "human_intervention": preview.get("human_intervention"),
            "human_events": preview.get("human_events") or [],
            "operator_messages": preview.get("operator_messages") or [],
            "human_hypothesis_events": preview.get("human_hypothesis_events") or [],
            "executed": False,
            "actual_selected": None,
            "observation": None,
            "transition": None,
            "world_after": None,
            "posterior": None,
            "posterior_delta": {},
            "hypothesis_changes": {},
            "terminal_decision": False,
        }

        if status == "discarded_before_execution":
            frame["outcome"] = "discarded_before_execution"
            frames.append(frame)
            continue

        matched = None
        for candidate_index in range(ledger_cursor, len(ledger)):
            candidate = ledger[candidate_index]
            candidate_step = int(candidate.get("step", -1))
            if candidate_step < step:
                ledger_cursor = candidate_index + 1
                continue
            if candidate_step == step:
                matched = candidate
                ledger_cursor = candidate_index + 1
            break

        if matched is not None:
            frame.update(
                {
                    "outcome": "executed",
                    "executed": True,
                    "actual_selected": matched.get("selected"),
                    "observation": matched.get("observation"),
                    "transition": matched.get("transition"),
                    "world_after": matched.get("world_after"),
                    "posterior": matched.get("posterior"),
                    "posterior_delta": matched.get("posterior_delta") or {},
                    "hypothesis_changes": matched.get("hypothesis_changes") or {},
                    "canonical_episode": matched,
                }
            )
            frames.append(frame)
            continue

        terminal = None
        for candidate_index in range(decision_cursor, len(decisions)):
            candidate = decisions[candidate_index]
            candidate_step = int(candidate.get("step", -1))
            if candidate_step < step:
                decision_cursor = candidate_index + 1
                continue
            if candidate_step == step:
                selected = candidate.get("selected") or {}
                if selected.get("kind") == "stop":
                    terminal = candidate
                    decision_cursor = candidate_index + 1
            break

        if terminal is not None:
            frame.update(
                {
                    "outcome": "terminal_without_tool_execution",
                    "actual_selected": terminal.get("selected"),
                    "terminal_decision": True,
                }
            )
        elif status in {"waiting", "pending"}:
            frame["outcome"] = "pending"
        else:
            # Partial archives are visible as uncertainty rather than guessed
            # into an execution that the canonical ledger cannot prove.
            frame["outcome"] = "released_without_canonical_outcome"

        frames.append(frame)

    return {
        "schema_version": SEMANTIC_REPLAY_SCHEMA,
        "frame_count": len(frames),
        "executed_count": sum(1 for row in frames if row["executed"]),
        "discarded_count": sum(
            1
            for row in frames
            if row["outcome"] == "discarded_before_execution"
        ),
        "frames": frames,
    }
