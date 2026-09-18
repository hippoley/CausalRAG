from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from causalrag.agent.actions import ActionKind, CandidateAction, DecisionRecord
from causalrag.agent.loop import DecisionGateReplan
from causalrag.observability import CausalTelemetry, CausalTraceRecord
from causalrag.world_model import CausalWorldModel

from .runtime import ProbeRunConfig, build_probe_agent


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
        "tests_hypotheses": list(candidate.tests_hypotheses),
        "falsification_target": candidate.falsification_target,
        "rationale": candidate.rationale,
    }


class InteractiveDecisionGate:
    """Blocking pre-execution gate used by the Playable Probe.

    The agent thread reaches this gate only after the runtime has proposed,
    ranked, and temporally guarded the next action. No tool has executed yet.
    A human may approve the runtime selection or choose another proposed
    candidate. Candidate overrides still pass through runtime temporal safety.
    """

    def __init__(self, telemetry: CausalTelemetry, *, timeout_seconds: float = 900.0) -> None:
        self.telemetry = telemetry
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._condition = threading.Condition(threading.RLock())
        self._pending: Optional[Dict[str, Any]] = None
        self._decision: Optional[DecisionRecord] = None
        self._response: Optional[Dict[str, Any]] = None
        self._closed = False

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
        pending = {
            "gate_id": uuid.uuid4().hex,
            "step": int(state.step),
            "uncertainty": decision.uncertainty,
            "runtime_selected": _candidate_payload(decision.selected, -1),
            "candidates": candidates,
            "action_scores": scores,
            "hypotheses": world_model.snapshot().get("hypotheses", []),
        }
        with self._condition:
            self._pending = pending
            self._decision = decision
            self._response = None
            self._condition.notify_all()

        self.telemetry.event(
            "causalrag.human_gate.waiting",
            {
                "causalrag.step": int(state.step),
                "causalrag.human_gate.gate_id": pending["gate_id"],
                "causalrag.action.name": decision.selected.name,
                "causalrag.action.kind": decision.selected.kind.value,
                "causalrag.candidate.count": len(candidates),
                "causalrag.candidate.names": [row["name"] for row in candidates],
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
            self._pending = None
            self._decision = None
            self._response = None

        if response is None:
            self.telemetry.event(
                "causalrag.human_gate.timeout",
                {
                    "causalrag.step": int(state.step),
                    "causalrag.action.name": decision.selected.name,
                },
            )
            return None

        if response["action"] == "approve":
            self.telemetry.event(
                "causalrag.human_gate.approved",
                {
                    "causalrag.step": int(state.step),
                    "causalrag.action.name": decision.selected.name,
                },
            )
            return None

        if response["action"] == "replan":
            self.telemetry.event(
                "causalrag.human_gate.replan",
                {
                    "causalrag.step": int(state.step),
                    "causalrag.action.previous": decision.selected.name,
                },
            )
            raise DecisionGateReplan()

        index = int(response["candidate_index"])
        candidate = decision.candidates[index]
        self.telemetry.event(
            "causalrag.human_gate.override",
            {
                "causalrag.step": int(state.step),
                "causalrag.action.original": decision.selected.name,
                "causalrag.action.name": candidate.name,
                "causalrag.action.kind": candidate.kind.value,
                "causalrag.human_gate.candidate_index": index,
            },
        )
        return candidate

    def pending(self) -> Optional[Dict[str, Any]]:
        with self._condition:
            # A submitted response is single-assignment. Hide the gate from
            # pollers immediately so a fast UI refresh cannot overwrite a
            # choose/replan with a second approve before the agent thread wakes.
            if self._response is not None:
                return None
            return _jsonable(self._pending) if self._pending is not None else None

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
            self._response = response
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class ProbeSession:
    def __init__(self, config: ProbeRunConfig) -> None:
        self.session_id = uuid.uuid4().hex
        self.config = config
        self.telemetry = CausalTelemetry(capture_content=False)
        self.gate = InteractiveDecisionGate(self.telemetry)
        environment, agent, goal, capabilities = build_probe_agent(
            config,
            telemetry=self.telemetry,
            decision_gate=self.gate,
        )
        self.environment = environment
        self.agent = agent
        self.goal = goal
        self.capabilities = capabilities
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._started = False
        self._completed = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"causalrag-probe-{self.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        self.telemetry.event(
            "causalrag.probe.session_started",
            {
                "causalrag.probe.session_id": self.session_id,
                "causalrag.probe.proposer_family": self.config.proposer_family,
                "causalrag.probe.model": self.config.model or "",
            },
        )
        try:
            result = self.agent.run(self.goal, max_steps=int(self.config.max_steps))
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
            }
            with self._lock:
                self._result = final
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self.telemetry.event(
                "causalrag.probe.session_failed",
                {
                    "causalrag.probe.session_id": self.session_id,
                    "error.type": type(exc).__name__,
                },
            )
        finally:
            with self._lock:
                self._completed = True
            self.telemetry.event(
                "causalrag.probe.session_finished",
                {
                    "causalrag.probe.session_id": self.session_id,
                    "causalrag.probe.failed": bool(self._error),
                },
            )

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

    def resolve_decision(self, action: str, candidate_index: Optional[int] = None) -> None:
        self.gate.respond(action, candidate_index)

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
        self.telemetry.event(
            "causalrag.human.hypothesis_added",
            {
                "causalrag.hypothesis.id": hypothesis.hypothesis_id,
                "causalrag.hypothesis.probability": hypothesis.probability,
            },
        )
        return _jsonable(hypothesis)


class ProbeSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, ProbeSession] = {}
        self._lock = threading.RLock()

    def create(self, config: ProbeRunConfig) -> ProbeSession:
        session = ProbeSession(config)
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


SESSION_MANAGER = ProbeSessionManager()


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
