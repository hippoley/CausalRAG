from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from threading import Condition, Thread
from typing import Any, Dict, List, Optional

from causalrag.agent import ActionKind, CandidateAction, DecisionRecord

from .runtime import ProbeRunConfig, _build_probe_agent


def _action_dict(action: CandidateAction) -> Dict[str, Any]:
    row = asdict(action)
    row["kind"] = action.kind.value
    return row


def _score_dict(score) -> Dict[str, Any]:
    row = asdict(score)
    row["action_kind"] = score.action_kind.value
    return row


@dataclass
class ProbePreview:
    preview_id: str
    step: int
    uncertainty: Optional[str]
    proposer_preference: Optional[str]
    runtime_preference: Optional[str]
    disagreement: bool
    candidates: List[Dict[str, Any]]
    ranking: List[Dict[str, Any]]
    hypotheses: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    explanation: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InteractiveProbeSession:
    """Pause the canonical CausalAgentLoop at each decision for human review.

    The agent still owns proposal parsing, runtime ranking, temporal guards,
    tool execution, Bayesian updates, open-world mismatch, hypothesis discovery,
    belief updates, and telemetry. This class only supplies a blocking
    DecisionHook and exposes the pending DecisionRecord to a UI/API.
    """

    def __init__(self, config: ProbeRunConfig) -> None:
        self.session_id = uuid.uuid4().hex
        self.config = config
        self._condition = Condition()
        self._preview: Optional[ProbePreview] = None
        self._pending_decision: Optional[DecisionRecord] = None
        self._selection: Optional[str] = None
        self._human_note = ""
        self._cancelled = False
        self._thread: Optional[Thread] = None
        self._error: Optional[BaseException] = None
        self._result = None
        self.state = None
        self.human_events: List[Dict[str, Any]] = []

        built = _build_probe_agent(config, decision_hook=self._decision_gate)
        self.scenario = built["scenario"]
        self.environment = built["environment"]
        self.agent = built["agent"]
        self.telemetry = built["telemetry"]
        self.capabilities = built["capabilities"]
        self.world_model = self.agent.world_model
        self.goal = built["goal"]

    @property
    def done(self) -> bool:
        with self._condition:
            return bool(self._result is not None or self._error is not None or self._cancelled)

    def _hypotheses(self) -> List[Dict[str, Any]]:
        return [asdict(item) for item in self.world_model.hypotheses()]

    def _observations(self) -> List[Dict[str, Any]]:
        state = self.state
        if state is None:
            return []
        return [
            {
                "action_name": item.action_name,
                "result": item.result,
                "metadata": item.metadata,
            }
            for item in state.observations
        ]

    def _explain_ranking(self, decision: DecisionRecord) -> List[str]:
        proposer = decision.candidates[0] if decision.candidates else None
        runtime = decision.selected
        score = next(
            (
                row
                for row in decision.action_scores
                if row.action_name == runtime.name and row.action_kind == runtime.kind
            ),
            None,
        )
        notes: List[str] = []
        if proposer is not None and proposer.name != runtime.name:
            notes.append(f"Runtime overrides proposer: {proposer.name} -> {runtime.name}.")
        if score is not None:
            notes.append(
                f"Information value from {score.information_source}: "
                f"{score.information_gain:.3f}; model claimed {score.model_information_gain:.3f}."
            )
            if score.decision_value is not None:
                text = f"Runtime expected decision utility: {score.decision_value:.3f}"
                if score.expected_value_of_sample_information is not None:
                    text += f"; EVSI {score.expected_value_of_sample_information:.3f}"
                notes.append(text + ".")
            notes.append(
                "Runtime-enforced cost/risk/irreversibility: "
                f"{score.cost:.3f}/{score.risk:.3f}/{score.irreversibility:.3f}."
            )
        if not notes:
            notes.append("No runtime disagreement: proposer and runtime currently align.")
        return notes

    def _make_preview(self, state, decision: DecisionRecord) -> ProbePreview:
        proposer = decision.candidates[0] if decision.candidates else None
        runtime = decision.selected
        ranking_by_index = {
            score.candidate_index: score for score in decision.action_scores
        }
        ranking: List[Dict[str, Any]] = []
        if decision.action_scores:
            for score in sorted(
                decision.action_scores,
                key=lambda row: row.total_utility,
                reverse=True,
            ):
                action = decision.candidates[score.candidate_index]
                ranking.append(
                    {"action": _action_dict(action), "score": _score_dict(score)}
                )
        else:
            for index, action in enumerate(decision.candidates):
                ranking.append(
                    {
                        "action": _action_dict(action),
                        "score": _score_dict(ranking_by_index[index])
                        if index in ranking_by_index
                        else None,
                        "rank_reason": "causal_selection disabled; proposer order preserved",
                        "proposer_index": index,
                    }
                )

        return ProbePreview(
            preview_id=uuid.uuid4().hex,
            step=state.step,
            uncertainty=decision.uncertainty,
            proposer_preference=proposer.name if proposer else None,
            runtime_preference=runtime.name if runtime else None,
            disagreement=bool(
                proposer is not None and runtime is not None and proposer.name != runtime.name
            ),
            candidates=[_action_dict(item) for item in decision.candidates],
            ranking=ranking,
            hypotheses=self._hypotheses(),
            observations=self._observations(),
            explanation=self._explain_ranking(decision),
        )

    def _resolve_choice(self, decision: DecisionRecord, selection: str) -> CandidateAction:
        selection = str(selection or "runtime")
        if selection == "runtime":
            return decision.selected
        if selection == "proposer":
            if decision.candidates:
                return decision.candidates[0]
            return decision.selected
        for action in decision.candidates:
            if action.name == selection:
                return action
        raise ValueError(f"unknown candidate selection: {selection}")

    def _decision_gate(self, state, world_model, decision: DecisionRecord) -> Optional[CandidateAction]:
        with self._condition:
            self.state = state
            self._pending_decision = decision
            self._preview = self._make_preview(state, decision)
            self._selection = None
            self._human_note = ""
            self._condition.notify_all()

            while self._selection is None and not self._cancelled:
                self._condition.wait()

            if self._cancelled:
                chosen = CandidateAction(
                    kind=ActionKind.STOP,
                    name="stop",
                    rationale="Human closed the interactive probe session.",
                )
                selection = "cancel"
                note = ""
            else:
                selection = str(self._selection)
                note = self._human_note
                chosen = self._resolve_choice(decision, selection)

            proposer_name = decision.candidates[0].name if decision.candidates else None
            runtime_name = decision.selected.name
            event = {
                "step": state.step,
                "selection": selection,
                "selected_action": chosen.name,
                "proposer_preference": proposer_name,
                "runtime_preference": runtime_name,
                "human_note": note,
                "overrode_runtime": chosen.name != runtime_name,
                "overrode_proposer": bool(proposer_name and chosen.name != proposer_name),
            }
            self.human_events.append(event)
            # This runs on the same background thread and inside the active
            # invoke_agent span, so the human choice stays trace-correlated.
            self.telemetry.event(
                "causalrag.probe.human_choice",
                {"probe.session_id": self.session_id, **event},
            )
            self._preview = None
            self._pending_decision = None
            self._selection = None
            self._human_note = ""
            self._condition.notify_all()
            return chosen

    def _run(self) -> None:
        try:
            result = self.agent.run(self.goal, max_steps=int(self.config.max_steps))
            with self._condition:
                self._result = result
                self.state = result.state
                self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def start(self, *, timeout: float = 120.0) -> Dict[str, Any]:
        with self._condition:
            if self._thread is None:
                self._thread = Thread(
                    target=self._run,
                    name=f"causalrag-probe-{self.session_id[:8]}",
                    daemon=True,
                )
                self._thread.start()
            ready = self._condition.wait_for(
                lambda: self._preview is not None
                or self._result is not None
                or self._error is not None,
                timeout=timeout,
            )
            if not ready:
                raise TimeoutError("timed out waiting for proposer/runtime decision")
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            return self.snapshot()

    def preview(self, *, force_replan: bool = False) -> Dict[str, Any]:
        if force_replan:
            raise ValueError(
                "force_replan is intentionally disabled in canonical interactive mode; "
                "replanning without a world action would consume another proposer sample "
                "and break paired experiment accounting."
            )
        with self._condition:
            if self._preview is None:
                if self._result is not None:
                    return self.snapshot()
                raise RuntimeError("session is not paused at a decision")
            return self._preview.to_dict()

    def commit(
        self,
        selection: str = "runtime",
        *,
        human_note: str = "",
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            if self._result is not None:
                return self.snapshot()
            if self._preview is None:
                raise RuntimeError("session is not waiting for a human decision")
            previous_step = self._preview.step
            self._selection = str(selection or "runtime")
            self._human_note = str(human_note or "")
            self._condition.notify_all()

            advanced = self._condition.wait_for(
                lambda: self._error is not None
                or self._result is not None
                or (
                    self._preview is not None
                    and self._preview.step > previous_step
                ),
                timeout=timeout,
            )
            if not advanced:
                raise TimeoutError("timed out waiting for the next agent decision")
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            return self.snapshot()

    def close(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        state = self.state
        if self._result is not None:
            metrics = self.environment.metrics(self._result)
            payload = self._result.to_dict()
            trace = payload["causal_trace"]
            trace_id = payload["trace_id"]
        else:
            result_proxy = type(
                "_ProbeResult",
                (),
                {"world_model": self.world_model, "state": state},
            )()
            metrics = self.environment.metrics(result_proxy) if state is not None else None
            trace = self.telemetry.records()
            trace_id = trace[0]["trace_id"] if trace else None

        return {
            "session_id": self.session_id,
            "done": self._result is not None,
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "step": state.step if state is not None else 0,
            "stop_reason": state.stop_reason if state is not None else None,
            "trace_id": trace_id,
            "config": self.config.to_dict(),
            "metrics": metrics.to_dict() if metrics is not None else None,
            "hypotheses": self._hypotheses(),
            "observations": self._observations(),
            "human_events": list(self.human_events),
            "trace": trace,
            "preview": self._preview.to_dict() if self._preview else None,
        }
