from __future__ import annotations

import json
import os
import math
import queue
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional

from branchpoint.agent.actions import ActionKind, CandidateAction, DecisionRecord
from branchpoint.agent.loop import DecisionGateReplan
from branchpoint.observability import CausalTelemetry, CausalTraceRecord
from branchpoint.world_model import CausalWorldModel
from branchpoint.experiments import (
    expanded_experiment_contract,
    experiment_decision_value,
    intervention_value,
    posterior_for_outcome,
)

from .runtime import ProbeRunConfig, build_probe_agent
from .archive import SQLiteSessionArchive
from .counterfactual import counterfactual_support, run_trajectory_counterfactual
from .replay import build_semantic_replay


def _jsonable(value: Any) -> Any:
    if isinstance(value, ActionKind):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


def _candidate_payload(candidate: CandidateAction, index: int) -> Dict[str, Any]:
    return {
        "index": int(index),
        "kind": candidate.kind.value,
        "name": candidate.name,
        "arguments": _jsonable(candidate.arguments),
        "expected_goal_gain": float(candidate.expected_goal_gain),
        "expected_information_gain": float(candidate.expected_information_gain),
        "cost": float(candidate.cost),
        "risk": float(candidate.risk),
        "irreversibility": float(candidate.irreversibility),
        "tests_hypotheses": list(candidate.tests_hypotheses),
        "falsification_target": candidate.falsification_target,
        "rationale": candidate.rationale,
    }


def _same_candidate(left: CandidateAction, right: CandidateAction) -> bool:
    return (
        left.name == right.name
        and left.kind == right.kind
        and dict(left.arguments) == dict(right.arguments)
    )


def _decision_inspector(decision: DecisionRecord) -> Dict[str, Any]:
    """Explain runtime arbitration without exposing private model reasoning."""

    score_by_index = {
        int(score.candidate_index): score
        for score in decision.action_scores
    }
    proposer_first = decision.candidates[0] if decision.candidates else None
    selected_index = next(
        (
            index
            for index, candidate in enumerate(decision.candidates)
            if _same_candidate(candidate, decision.selected)
        ),
        None,
    )
    proposer_score = score_by_index.get(0)
    selected_score = score_by_index.get(selected_index) if selected_index is not None else None

    if proposer_first is None:
        divergence_kind = "no_proposer_candidate"
        diverged = True
    elif _same_candidate(proposer_first, decision.selected):
        divergence_kind = "none"
        diverged = False
    elif selected_index is None:
        divergence_kind = "runtime_guard_override"
        diverged = True
    else:
        divergence_kind = "runtime_reordered"
        diverged = True

    reasons: list[Dict[str, Any]] = []
    if proposer_first is not None and proposer_score is None and decision.action_scores:
        reasons.append(
            {
                "code": "proposer_first_not_runtime_ranked",
                "detail": "The proposer's first candidate did not survive runtime scoring/validation.",
            }
        )

    if selected_score is not None:
        if selected_score.information_source != "model_estimate":
            reasons.append(
                {
                    "code": "runtime_information_source",
                    "source": selected_score.information_source,
                    "model_information_gain": selected_score.model_information_gain,
                    "runtime_information_gain": selected_score.information_gain,
                }
            )
        if selected_score.decision_value_source:
            reasons.append(
                {
                    "code": "runtime_decision_value",
                    "source": selected_score.decision_value_source,
                    "decision_value": selected_score.decision_value,
                    "evsi": selected_score.expected_value_of_sample_information,
                    "net_value_of_sampling": selected_score.net_value_of_sampling,
                }
            )
        if proposer_score is not None and selected_index != 0:
            reasons.append(
                {
                    "code": "higher_runtime_utility",
                    "proposer_first_utility": proposer_score.total_utility,
                    "runtime_selected_utility": selected_score.total_utility,
                    "delta": selected_score.total_utility - proposer_score.total_utility,
                }
            )

    rationale = str(decision.selected.rationale or "")
    if divergence_kind == "runtime_guard_override" and "guard" in rationale.lower():
        reasons.append(
            {
                "code": "runtime_safety_or_temporal_guard",
                "detail": rationale,
            }
        )

    return {
        "diverged": diverged,
        "divergence_kind": divergence_kind,
        "proposer_first": (
            None if proposer_first is None else _candidate_payload(proposer_first, 0)
        ),
        "proposer_first_score": _jsonable(proposer_score),
        "runtime_selected": _candidate_payload(decision.selected, -1),
        "runtime_selected_candidate_index": selected_index,
        "runtime_selected_score": _jsonable(selected_score),
        "reasons": reasons,
    }


def _hypothesis_map(snapshot: Dict[str, Any]) -> Dict[str, float]:
    return {
        str(row.get("id") or row.get("hypothesis_id")): float(row.get("probability", 0.0))
        for row in (snapshot.get("hypotheses") or [])
        if row.get("id") or row.get("hypothesis_id")
    }


def _world_from_snapshot(snapshot: Dict[str, Any]) -> CausalWorldModel:
    world = CausalWorldModel()
    for row in snapshot.get("hypotheses") or []:
        hypothesis_id = str(row.get("id") or row.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            continue
        world.upsert_hypothesis(
            hypothesis_id,
            str(row.get("statement") or hypothesis_id),
            probability=float(row.get("probability", 0.5)),
            rationale=str(row.get("rationale") or ""),
            falsifiers=row.get("falsifiers") or [],
            origin=str(row.get("origin") or "authored"),
            validated=bool(row.get("validated", True)),
            experiment_predictions=row.get("experiment_predictions") or {},
        )
    return world


def _entropy(values) -> float:
    return -sum(float(p) * math.log(float(p)) for p in values if float(p) > 0.0)


def _score_provenance(decision: DecisionRecord, tools=None) -> Dict[str, Any]:
    if tools is None:
        return {}
    world = _world_from_snapshot(decision.beliefs_before)
    score_by_index = {int(score.candidate_index): score for score in decision.action_scores}
    provenance: Dict[str, Any] = {}

    for index, candidate in enumerate(decision.candidates):
        score = score_by_index.get(index)
        row: Dict[str, Any] = {
            "candidate_index": index,
            "action_name": candidate.name,
            "action_kind": candidate.kind.value,
            "formula": (
                "decision_value"
                if score is not None and score.decision_value is not None
                else "goal_gain + information_gain - cost - risk - irreversibility"
            ),
            "score": _jsonable(score),
        }
        if score is None or candidate.kind in {ActionKind.STOP, ActionKind.WAIT}:
            provenance[str(index)] = row
            continue

        try:
            tool_spec = tools.get(candidate.name)
        except KeyError:
            provenance[str(index)] = row
            continue

        experiment = getattr(tool_spec, "experiment_contract", None)
        intervention = getattr(tool_spec, "intervention_contract", None)

        if experiment is not None:
            expanded = expanded_experiment_contract(experiment, world)
            ids = expanded.hypothesis_ids()
            raw_prior = {
                hypothesis_id: max(
                    0.0,
                    float(world.get_hypothesis(hypothesis_id).probability),
                )
                for hypothesis_id in ids
                if world.get_hypothesis(hypothesis_id) is not None
            }
            total = sum(raw_prior.values())
            prior = (
                {hypothesis_id: value / total for hypothesis_id, value in raw_prior.items()}
                if total > 0.0
                else ({hypothesis_id: 1.0 / len(raw_prior) for hypothesis_id in raw_prior} if raw_prior else {})
            )
            prior_entropy = _entropy(prior.values())
            expected_posterior_entropy = 0.0
            outcomes = []
            for outcome in expanded.outcomes:
                outcome_probability = sum(
                    prior.get(hypothesis_id, 0.0) * float(outcome.likelihoods[hypothesis_id])
                    for hypothesis_id in prior
                )
                posterior = posterior_for_outcome(expanded, world, outcome.outcome)
                posterior_entropy = _entropy(posterior.values())
                expected_posterior_entropy += outcome_probability * posterior_entropy
                outcomes.append(
                    {
                        "outcome": outcome.outcome,
                        "predictive_probability": outcome_probability,
                        "likelihoods": {
                            str(hypothesis_id): float(value)
                            for hypothesis_id, value in outcome.likelihoods.items()
                        },
                        "posterior": posterior,
                        "posterior_entropy": posterior_entropy,
                    }
                )
            normalized_eig = (
                max(0.0, min(1.0, (prior_entropy - expected_posterior_entropy) / prior_entropy))
                if prior_entropy > 0.0
                else 0.0
            )
            row["experiment"] = {
                "experiment_id": expanded.experiment_id,
                "outcome_key": expanded.outcome_key,
                "prior": prior,
                "prior_entropy": prior_entropy,
                "outcomes": outcomes,
                "expected_posterior_entropy": expected_posterior_entropy,
                "normalized_eig": normalized_eig,
            }
            if score.expected_value_of_sample_information is not None:
                value = experiment_decision_value(
                    candidate.name,
                    expanded,
                    world,
                    tools,
                    experiment_cost=float(score.cost),
                )
                if value is not None:
                    row["evsi"] = _jsonable(value)

        if intervention is not None:
            value = intervention_value(
                candidate.name,
                intervention,
                world,
                capability_cost=float(score.cost),
            )
            if value is not None:
                row["intervention_value"] = _jsonable(value)

        provenance[str(index)] = row

    return provenance


def _hypothesis_changes(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Describe structural hypothesis-model changes across one completed step."""

    def rows(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for row in snapshot.get("hypotheses") or []:
            hypothesis_id = str(row.get("id") or row.get("hypothesis_id") or "").strip()
            if hypothesis_id:
                result[hypothesis_id] = row
        return result

    before_rows = rows(before)
    after_rows = rows(after)
    added = [
        _jsonable(after_rows[hypothesis_id])
        for hypothesis_id in after_rows
        if hypothesis_id not in before_rows
    ]
    removed = [
        _jsonable(before_rows[hypothesis_id])
        for hypothesis_id in before_rows
        if hypothesis_id not in after_rows
    ]
    validation_changed = []
    for hypothesis_id in sorted(set(before_rows).intersection(after_rows)):
        left = bool(before_rows[hypothesis_id].get("validated", True))
        right = bool(after_rows[hypothesis_id].get("validated", True))
        if left != right:
            validation_changed.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "before": left,
                    "after": right,
                    "statement": after_rows[hypothesis_id].get("statement"),
                    "origin": after_rows[hypothesis_id].get("origin"),
                }
            )
    return {
        "added": added,
        "removed": removed,
        "validation_changed": validation_changed,
        "structural_change": bool(added or removed or validation_changed),
    }


def _episode_ledger(state, world_model: CausalWorldModel, tools=None) -> list[Dict[str, Any]]:
    """Build a human-readable ledger from canonical runtime state.

    Only completed actions are included. The currently paused decision is not
    treated as executed until an observation exists.
    """
    ledger: list[Dict[str, Any]] = []
    human_history = list(state.scratch.get("human_gate_history", []))
    proposer_history = list(state.scratch.get("proposer_traces", []))
    current_snapshot = world_model.snapshot()

    transition_cursor = 0
    transitions = list(world_model.transitions)

    for index, observation in enumerate(state.observations):
        if index >= len(state.decisions):
            break
        decision = state.decisions[index]

        matched_transition = None
        for transition_index in range(transition_cursor, len(transitions)):
            candidate_transition = transitions[transition_index]
            if candidate_transition.action == decision.selected.name:
                matched_transition = candidate_transition
                transition_cursor = transition_index + 1
                break
        before = decision.beliefs_before
        if index + 1 < len(state.decisions):
            after = state.decisions[index + 1].beliefs_before
        else:
            after = current_snapshot

        before_map = _hypothesis_map(before)
        after_map = _hypothesis_map(after)
        posterior_delta = {
            hypothesis_id: after_map.get(hypothesis_id, 0.0) - probability
            for hypothesis_id, probability in before_map.items()
            if hypothesis_id in after_map
        }
        gate_events = [
            row for row in human_history
            if int(row.get("step", -1)) == int(decision.step)
        ]
        effective_human = next(
            (
                row
                for row in reversed(gate_events)
                if row.get("action") in {"approve", "choose"}
            ),
            None,
        )

        selected_score = next(
            (
                score
                for score in decision.action_scores
                if score.action_name == decision.selected.name
                and score.action_kind == decision.selected.kind
            ),
            None,
        )
        proposer_attempts = [
            row for row in proposer_history
            if int(row.get("step", -1)) == int(decision.step)
        ]
        effective_proposer = proposer_attempts[-1] if proposer_attempts else None
        ledger.append(
            {
                "step": int(decision.step),
                "uncertainty": decision.uncertainty,
                "proposer": _jsonable(effective_proposer),
                "proposer_attempts": _jsonable(proposer_attempts),
                "prior": before.get("hypotheses", []),
                "world_before": _jsonable(before),
                "candidates": [
                    _candidate_payload(candidate, candidate_index)
                    for candidate_index, candidate in enumerate(decision.candidates)
                ],
                "action_scores": [_jsonable(score) for score in decision.action_scores],
                "score_provenance": _score_provenance(decision, tools),
                "selected": _candidate_payload(decision.selected, -1),
                "runtime_score": _jsonable(selected_score),
                "decision_inspector": _decision_inspector(decision),
                "human": _jsonable(effective_human),
                "observation": _jsonable(observation),
                "transition": _jsonable(matched_transition),
                "posterior": after.get("hypotheses", []),
                "world_after": _jsonable(after),
                "posterior_delta": posterior_delta,
                "hypothesis_changes": _hypothesis_changes(before, after),
            }
        )
    return ledger


class InteractiveDecisionGate:
    """Blocking pre-execution gate used by the Playable Probe.

    The agent thread reaches this gate only after the runtime has proposed,
    ranked, and temporally guarded the next action. No tool has executed yet.
    A human may approve the runtime selection or choose another proposed
    candidate. Candidate overrides still pass through runtime temporal safety.
    """

    def __init__(
        self,
        telemetry: CausalTelemetry,
        *,
        timeout_seconds: float = 900.0,
        change_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.telemetry = telemetry
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.change_callback = change_callback
        self._condition = threading.Condition(threading.RLock())
        self._pending: Optional[Dict[str, Any]] = None
        self._decision: Optional[DecisionRecord] = None
        self._response: Optional[Dict[str, Any]] = None
        self._state = None
        self._history: list[Dict[str, Any]] = []
        self._closed = False
        self.tools = None

    def _changed(self) -> None:
        callback = self.change_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:
            # Persistence must never make the control gate fail open or crash.
            pass

    def __call__(
        self,
        state,
        world_model: CausalWorldModel,
        decision: DecisionRecord,
    ) -> Optional[CandidateAction]:
        scores = [_jsonable(score) for score in decision.action_scores]
        valid_indexes = (
            {int(score.candidate_index) for score in decision.action_scores}
            if decision.action_scores
            else set(range(len(decision.candidates)))
        )
        candidates = []
        for index, candidate in enumerate(decision.candidates):
            row = _candidate_payload(candidate, index)
            row["runtime_valid"] = index in valid_indexes
            candidates.append(row)
        world_before = _jsonable(world_model.snapshot())
        pending = {
            "gate_id": uuid.uuid4().hex,
            "step": int(state.step),
            "status": "waiting",
            "uncertainty": decision.uncertainty,
            "runtime_selected": _candidate_payload(decision.selected, -1),
            "proposer": _jsonable(state.scratch.get("last_proposer_trace")),
            "proposer_attempts": _jsonable(
                [
                    row
                    for row in state.scratch.get("proposer_traces", [])
                    if int(row.get("step", -1)) == int(state.step)
                ]
            ),
            "hypothesis_proposals": _jsonable(
                state.scratch.get("last_hypothesis_proposals", [])
            ),
            "candidates": candidates,
            "action_scores": scores,
            "score_provenance": _score_provenance(decision, self.tools),
            "decision_inspector": _decision_inspector(decision),
            "hypotheses": world_before.get("hypotheses", []),
            "world_before": world_before,
            "world_after_gate": None,
            "human_intervention": None,
            "human_events": [],
            "operator_messages": [],
            "human_hypothesis_events": [],
            "episode_ledger": _episode_ledger(state, world_model, self.tools),
        }
        with self._condition:
            self._pending = pending
            self._history.append(pending)
            self._decision = decision
            self._state = state
            self._response = None
            self._condition.notify_all()
        self._changed()

        self.telemetry.event(
            "branchpoint.human_gate.waiting",
            {
                "branchpoint.step": int(state.step),
                "branchpoint.human_gate.gate_id": pending["gate_id"],
                "branchpoint.action.name": decision.selected.name,
                "branchpoint.action.kind": decision.selected.kind.value,
                "branchpoint.candidate.count": len(candidates),
                "branchpoint.candidate.names": [row["name"] for row in candidates],
            },
        )

        deadline = time.monotonic() + self.timeout_seconds
        with self._condition:
            while self._response is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(remaining, 1.0))
            response = self._response
            if response is None:
                pending["status"] = "released_by_timeout"
            elif response.get("action") == "replan":
                pending["status"] = "discarded_before_execution"
            else:
                pending["status"] = "released_for_execution"
            pending["world_after_gate"] = _jsonable(world_model.snapshot())
            self._pending = None
            self._decision = None
            self._state = None
            self._response = None

        if response is None:
            self.telemetry.event(
                "branchpoint.human_gate.timeout",
                {
                    "branchpoint.step": int(state.step),
                    "branchpoint.action.name": decision.selected.name,
                },
            )
            self._changed()
            return None

        if response["action"] == "approve":
            state.scratch.setdefault("human_gate_history", []).append(
                {
                    "step": int(state.step),
                    "gate_id": pending["gate_id"],
                    "action": "approve",
                    "selected": decision.selected.name,
                }
            )
            self.telemetry.event(
                "branchpoint.human_gate.approved",
                {
                    "branchpoint.step": int(state.step),
                    "branchpoint.action.name": decision.selected.name,
                },
            )
            self._changed()
            return None

        if response["action"] == "replan":
            state.scratch.setdefault("human_gate_history", []).append(
                {
                    "step": int(state.step),
                    "gate_id": pending["gate_id"],
                    "action": "replan",
                    "selected": decision.selected.name,
                }
            )
            self.telemetry.event(
                "branchpoint.human_gate.replan",
                {
                    "branchpoint.step": int(state.step),
                    "branchpoint.action.previous": decision.selected.name,
                },
            )
            self._changed()
            raise DecisionGateReplan()

        index = int(response["candidate_index"])
        candidate = decision.candidates[index]
        state.scratch.setdefault("human_gate_history", []).append(
            {
                "step": int(state.step),
                "gate_id": pending["gate_id"],
                "action": "choose",
                "candidate_index": index,
                "selected": candidate.name,
                "runtime_original": decision.selected.name,
            }
        )
        self.telemetry.event(
            "branchpoint.human_gate.override",
            {
                "branchpoint.step": int(state.step),
                "branchpoint.action.original": decision.selected.name,
                "branchpoint.action.name": candidate.name,
                "branchpoint.action.kind": candidate.kind.value,
                "branchpoint.human_gate.candidate_index": index,
            },
        )
        self._changed()
        return candidate

    def pending(self) -> Optional[Dict[str, Any]]:
        with self._condition:
            # A submitted response is single-assignment. Hide the gate from
            # pollers immediately so a fast UI refresh cannot overwrite a
            # choose/replan with a second approve before the agent thread wakes.
            if self._response is not None:
                return None
            return _jsonable(self._pending) if self._pending is not None else None

    def history(self) -> list[Dict[str, Any]]:
        with self._condition:
            return _jsonable(self._history)


    def live_state(self):
        with self._condition:
            return self._state

    def respond(self, action: str, candidate_index: Optional[int] = None) -> None:
        action = str(action).strip().lower()
        if action not in {"approve", "choose", "replan"}:
            raise ValueError("action must be approve, choose, or replan")
        with self._condition:
            if self._pending is None or self._decision is None:
                raise RuntimeError("no decision is waiting for human input")
            if self._response is not None:
                raise RuntimeError("human decision has already been submitted")
            response: Dict[str, Any] = {"action": action}
            intervention: Dict[str, Any] = {"action": action}
            if action == "choose":
                if candidate_index is None:
                    raise ValueError("candidate_index is required for choose")
                index = int(candidate_index)
                if index < 0 or index >= len(self._decision.candidates):
                    raise ValueError("candidate_index is out of range")
                candidate_rows = self._pending.get("candidates", [])
                if candidate_rows and not bool(candidate_rows[index].get("runtime_valid", False)):
                    raise ValueError("candidate was rejected by runtime validation")
                response["candidate_index"] = index
                intervention["candidate_index"] = index
                intervention["candidate"] = _candidate_payload(
                    self._decision.candidates[index],
                    index,
                )
            self._pending["human_intervention"] = intervention
            self._pending.setdefault("human_events", []).append(
                _jsonable(intervention)
            )
            self._response = response
            self._condition.notify_all()
        self._changed()


    def add_operator_message(self, message: str, *, replan: bool = True) -> Dict[str, Any]:
        """Inject a human message into the live agent state before execution."""
        message = str(message).strip()
        if not message:
            raise ValueError("operator message must be non-empty")
        with self._condition:
            if self._pending is None or self._decision is None or self._state is None:
                raise RuntimeError("operator messages require a paused decision")
            if self._response is not None:
                raise RuntimeError("human decision has already been submitted")
            row = {
                "step": int(self._state.step),
                "message": message,
            }
            self._state.scratch.setdefault("operator_messages", []).append(row)
            self._pending.setdefault("operator_messages", []).append(
                _jsonable(row)
            )
            event = {
                "action": "operator_message",
                "message": message,
                "replan": bool(replan),
            }
            self._pending["human_intervention"] = event
            self._pending.setdefault("human_events", []).append(
                _jsonable(event)
            )
            if replan:
                self._response = {"action": "replan"}
                self._condition.notify_all()

        self.telemetry.event(
            "branchpoint.human.operator_message",
            {
                "branchpoint.step": row["step"],
                "branchpoint.human.message_characters": len(message),
                "branchpoint.human.replan": bool(replan),
            },
        )
        self._changed()
        return row

    def record_hypothesis_added(self, event: Dict[str, Any]) -> None:
        with self._condition:
            if self._pending is None:
                return
            row = _jsonable(event)
            self._pending.setdefault("human_hypothesis_events", []).append(row)
            self._pending.setdefault("human_events", []).append(
                {"action": "add_hypothesis", **row}
            )
        self._changed()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._changed()


class ProbeSession:
    def __init__(
        self,
        config: ProbeRunConfig,
        *,
        persist_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self.config = config
        self._persist_callback = persist_callback
        self.telemetry = CausalTelemetry(capture_content=False)
        self.gate = InteractiveDecisionGate(
            self.telemetry,
            change_callback=self._persist,
        )
        environment, agent, goal, capabilities = build_probe_agent(
            config,
            telemetry=self.telemetry,
            decision_gate=self.gate,
        )
        self.environment = environment
        self.agent = agent
        self.gate.tools = self.agent.loop.tools
        self.goal = goal
        self.capabilities = capabilities
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._result: Optional[Dict[str, Any]] = None
        self._agent_state = None
        self._error: Optional[str] = None
        self._started = False
        self._completed = False

    def _persist(self) -> None:
        callback = self._persist_callback
        if callback is None:
            return
        try:
            callback(_jsonable(self.export_payload()))
        except Exception:
            # Archival failure must not grant execution authority or kill the run.
            pass

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"branchpoint-probe-{self.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()
        self._persist()

    def _run(self) -> None:
        self.telemetry.event(
            "branchpoint.probe.session_started",
            {
                "branchpoint.probe.session_id": self.session_id,
                "branchpoint.probe.proposer_family": self.config.proposer_family,
                "branchpoint.probe.model": self.config.model or "",
            },
        )
        try:
            result = self.agent.run(self.goal, max_steps=int(self.config.max_steps))
            self._agent_state = result.state
            metrics = self.environment.metrics(result)
            payload = result.to_dict()
            final = {
                "config": self.config.to_dict(),
                "metrics": metrics.to_dict(),
                "answer": payload["answer"],
                "trace_id": payload["trace_id"],
                "hypotheses": payload["hypotheses"],
                "open_world": payload["open_world"],
                "decisions": payload["decisions"],
                "observations": payload["observations"],
                "transitions": payload["transitions"],
                "causal_trace": payload["causal_trace"],
                "runtime_capabilities": self.capabilities.to_dict(),
                "proposer_traces": _jsonable(result.state.scratch.get("proposer_traces", [])),
                "episode_ledger": _episode_ledger(result.state, result.world_model, self.agent.loop.tools),
            }
            with self._lock:
                self._result = final
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self.telemetry.event(
                "branchpoint.probe.session_failed",
                {
                    "branchpoint.probe.session_id": self.session_id,
                    "error.type": type(exc).__name__,
                },
            )
        finally:
            with self._lock:
                self._completed = True
            self.telemetry.event(
                "branchpoint.probe.session_finished",
                {
                    "branchpoint.probe.session_id": self.session_id,
                    "branchpoint.probe.failed": bool(self._error),
                },
            )
            self._persist()

    def status(self) -> str:
        if self._completed:
            return "failed" if self._error else "completed"
        if self.gate.pending() is not None:
            return "waiting_for_human"
        return "running" if self._started else "created"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            result = _jsonable(self._result)
            error = self._error
        return {
            "session_id": self.session_id,
            "status": self.status(),
            "config": self.config.to_dict(),
            "goal": self.goal,
            "pending_decision": self.gate.pending(),
            "hypotheses": self.agent.world_model.snapshot().get("hypotheses", []),
            "open_world": self.agent.world_model.snapshot().get("open_world", {}),
            "result": result,
            "error": error,
            "trace_count": self.telemetry.count(),
        }

    def export_payload(self) -> Dict[str, Any]:
        """Export a self-contained, replayable experiment session."""

        live_state = self.gate.live_state()
        state = live_state or self._agent_state
        with self._lock:
            result = _jsonable(self._result)
            error = self._error

        scratch = {} if state is None else state.scratch
        ledger = (
            _episode_ledger(state, self.agent.world_model, self.agent.loop.tools)
            if state is not None
            else ((result or {}).get("episode_ledger") or [])
        )
        return {
            "schema_version": "branchpoint.playable_probe.session.v1",
            "session_id": self.session_id,
            "status": self.status(),
            "config": self.config.to_dict(),
            "goal": self.goal,
            "runtime_capabilities": self.capabilities.to_dict(),
            "proposer_traces": _jsonable(scratch.get("proposer_traces", [])),
            "pending_decision": self.gate.pending(),
            "world_model": _jsonable(self.agent.world_model.snapshot()),
            "episode_ledger": _jsonable(ledger),
            "interactions": {
                "human_gate_history": _jsonable(
                    scratch.get("human_gate_history", [])
                ),
                "operator_messages": _jsonable(
                    scratch.get("operator_messages", [])
                ),
                "human_hypothesis_events": _jsonable(
                    scratch.get("human_hypothesis_events", [])
                ),
            },
            "trace": self.telemetry.records(),
            "result": result,
            "error": error,
        }

    def step_context(self, step: int) -> Dict[str, Any]:
        """Return one frozen step context for IDE-style debugging."""
        target = int(step)
        live_state = self.gate.live_state()
        state = live_state or self._agent_state
        pending = self.gate.pending()

        if pending is not None and int(pending.get("step", -1)) == target:
            context = {
                **_jsonable(pending),
                "status": "pending",
                "world_before": {
                    **self.agent.world_model.snapshot(),
                    "hypotheses": _jsonable(pending.get("hypotheses", [])),
                },
                "world_after": None,
                "observation": None,
                "transition": None,
                "posterior": None,
                "posterior_delta": {},
                "human": None,
            }
        else:
            ledger = (
                _episode_ledger(state, self.agent.world_model, self.agent.loop.tools)
                if state is not None
                else ((_jsonable(self._result) or {}).get("episode_ledger") or [])
            )
            row = next((item for item in ledger if int(item.get("step", -1)) == target), None)
            if row is None:
                raise KeyError(target)
            context = {**_jsonable(row), "status": "completed"}

        trace_rows = []
        for record in self.telemetry.records():
            row = record.to_dict() if hasattr(record, "to_dict") else _jsonable(record)
            attrs = row.get("attributes") or {}
            if int(attrs.get("branchpoint.step", -1)) == target:
                trace_rows.append(row)

        scratch = {} if state is None else state.scratch
        human_events = [
            row
            for row in scratch.get("human_gate_history", [])
            if int(row.get("step", -1)) == target
        ]
        operator_messages = [
            row
            for row in scratch.get("operator_messages", [])
            if int(row.get("step", -1)) == target
        ]
        hypothesis_events = [
            row
            for row in scratch.get("human_hypothesis_events", [])
            if int(row.get("step", -1)) == target
        ]

        alternatives = [
            row
            for row in (context.get("candidates") or [])
            if row.get("runtime_valid", True)
            and row.get("name") != (context.get("selected") or context.get("runtime_selected") or {}).get("name")
        ]
        context["telemetry"] = trace_rows
        context["operator_context"] = {
            "human_gate_history": _jsonable(human_events),
            "operator_messages": _jsonable(operator_messages),
            "human_hypothesis_events": _jsonable(hypothesis_events),
        }
        support = counterfactual_support(self.config)
        if context.get("status") == "pending":
            support = {
                "available": False,
                "replay_mode": "live_override_preferred",
                "reason": (
                    "This step has not executed yet. Use the live Human Gate to choose an "
                    "alternative candidate or replan; a counterfactual fork would duplicate "
                    "an action you can still take for real."
                ),
            }
        context["counterfactual"] = {
            **support,
            "alternatives": _jsonable(alternatives),
            "whole_run_ab_available": True,
            "step_level_scope": "one_step_fork_then_stop",
        }
        return context

    def run_counterfactual(self, step: int, candidate_index: int) -> Dict[str, Any]:
        live_state = self.gate.live_state()
        state = live_state or self._agent_state
        if state is None:
            raise RuntimeError("counterfactual requires an active or completed session")
        ledger = _episode_ledger(state, self.agent.world_model, self.agent.loop.tools)
        fork = run_trajectory_counterfactual(
            self.config,
            ledger,
            step=int(step),
            candidate_index=int(candidate_index),
        )
        if not fork.get("available"):
            return fork

        actual_result = _jsonable(self._result) or {}
        actual_decisions = actual_result.get("decisions") or []
        actual_observations = actual_result.get("observations") or []
        target_step = int(step)
        actual_suffix = {
            "decisions": actual_decisions[target_step:],
            "observations": actual_observations[target_step:],
            "metrics": actual_result.get("metrics"),
            "answer": actual_result.get("answer"),
            "hypotheses": actual_result.get("hypotheses") or [],
            "open_world": actual_result.get("open_world") or {},
        }
        fork["actual_trajectory"] = actual_suffix

        cf = fork.get("counterfactual") or {}
        cf_decisions = cf.get("decisions") or []
        first_divergence = None
        width = max(len(actual_decisions), len(cf_decisions))
        for index in range(target_step, width):
            left = actual_decisions[index] if index < len(actual_decisions) else None
            right = cf_decisions[index] if index < len(cf_decisions) else None
            left_selected = None if left is None else (left.get("selected") or {})
            right_selected = None if right is None else (right.get("selected") or {})
            left_key = None if left_selected is None else (
                left_selected.get("kind"),
                left_selected.get("name"),
                left_selected.get("arguments") or {},
            )
            right_key = None if right_selected is None else (
                right_selected.get("kind"),
                right_selected.get("name"),
                right_selected.get("arguments") or {},
            )
            if left_key != right_key:
                first_divergence = {
                    "step": index,
                    "actual": left_selected,
                    "counterfactual": right_selected,
                }
                break
        fork["first_future_divergence"] = first_divergence

        actual_metrics = actual_result.get("metrics") or {}
        cf_metrics = cf.get("metrics") or {}
        metric_deltas: Dict[str, float] = {}
        for name in sorted(set(actual_metrics).intersection(cf_metrics)):
            left, right = actual_metrics[name], cf_metrics[name]
            if isinstance(left, bool) and isinstance(right, bool):
                metric_deltas[name] = float(int(right) - int(left))
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                metric_deltas[name] = float(right) - float(left)
        fork["metric_deltas_counterfactual_minus_actual"] = metric_deltas
        return fork

    def resolve_decision(self, action: str, candidate_index: Optional[int] = None) -> None:
        self.gate.respond(action, candidate_index)
        self._persist()

    def add_operator_message(self, message: str, *, replan: bool = True) -> Dict[str, Any]:
        row = self.gate.add_operator_message(message, replan=replan)
        self._persist()
        return row

    def add_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        *,
        probability: float = 0.2,
        rationale: str = "Added by human operator during Playable Probe.",
    ) -> Dict[str, Any]:
        if self.gate.pending() is None:
            raise RuntimeError("human hypotheses can only be added while a decision is paused")
        hypothesis_id = str(hypothesis_id).strip()
        statement = str(statement).strip()
        if not hypothesis_id or not statement:
            raise ValueError("hypothesis_id and statement must be non-empty")
        if self.agent.world_model.get_hypothesis(hypothesis_id) is not None:
            raise ValueError(f"hypothesis already exists: {hypothesis_id}")
        hypothesis = self.agent.world_model.upsert_hypothesis(
            hypothesis_id,
            statement,
            probability=max(0.01, min(0.4, float(probability))),
            rationale=rationale,
            origin="human",
            validated=False,
        )
        if self.gate._state is not None:
            self.gate._state.scratch.setdefault("human_hypothesis_events", []).append(
                {
                    "step": int(self.gate._state.step),
                    "hypothesis_id": hypothesis_id,
                    "statement": statement,
                    "probability": hypothesis.probability,
                }
            )
        self.telemetry.event(
            "branchpoint.human.hypothesis_added",
            {
                "branchpoint.hypothesis.id": hypothesis.hypothesis_id,
                "branchpoint.hypothesis.probability": hypothesis.probability,
            },
        )
        self._persist()
        return _jsonable(hypothesis)


class ProbeSessionManager:
    def __init__(self, archive: Optional[SQLiteSessionArchive] = None) -> None:
        self._sessions: Dict[str, ProbeSession] = {}
        self._lock = threading.RLock()
        self.archive = archive

    def create(self, config: ProbeRunConfig) -> ProbeSession:
        callback = None if self.archive is None else self.archive.save
        session = ProbeSession(config, persist_callback=callback)
        with self._lock:
            self._sessions[session.session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> ProbeSession:
        with self._lock:
            session = self._sessions.get(str(session_id))
        if session is None:
            raise KeyError(session_id)
        return session

    def archived(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            live = self._sessions.get(str(session_id))
        if live is not None:
            payload = _jsonable(live.export_payload())
            if self.archive is not None:
                self.archive.save(payload)
            return payload
        if self.archive is None:
            raise KeyError(session_id)
        return self.archive.get(session_id)

    def archive_index(self, *, limit: int = 100):
        if self.archive is None:
            return []
        return self.archive.list(limit=limit)


def _configured_archive() -> Optional[SQLiteSessionArchive]:
    path = str(os.getenv("BRANCHPOINT_PROBE_SESSION_DB", "")).strip()
    return SQLiteSessionArchive(path) if path else None


SESSION_MANAGER = ProbeSessionManager(_configured_archive())


def sse_stream(session: ProbeSession):
    """Yield canonical telemetry as Server-Sent Events until the session ends."""

    events: queue.Queue[Dict[str, Any]] = queue.Queue()

    def receive(record: CausalTraceRecord) -> None:
        events.put(record.to_dict())

    subscription_id = session.telemetry.subscribe(receive, replay_existing=True)
    try:
        while True:
            try:
                record = events.get(timeout=10.0)
                yield "event: trace\n" + "data: " + json.dumps(record, ensure_ascii=False) + "\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
            if session.status() in {"completed", "failed"} and events.empty():
                snapshot = session.snapshot()
                yield "event: session\n" + "data: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
                break
    finally:
        session.telemetry.unsubscribe(subscription_id)
