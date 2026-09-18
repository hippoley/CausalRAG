from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from causalrag.agent import ActionKind, AgentState, CandidateAction, DecisionRecord, Observation
from causalrag.experiments import apply_experiment_observation, expanded_experiment_contract
from causalrag.reasoning.policy import rank_actions
from causalrag.world_model import Transition

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
    """Human-in-the-loop Playable Probe session.

    Preview asks the proposer for candidate actions and lets the runtime score
    them, but executes nothing. Commit records the human choice, executes one
    chosen candidate, applies the validated experiment contract, and returns
    the resulting belief change.
    """

    def __init__(self, config: ProbeRunConfig) -> None:
        self.session_id = uuid.uuid4().hex
        self.config = config
        built = _build_probe_agent(config)
        self.scenario = built["scenario"]
        self.environment = built["environment"]
        self.agent = built["agent"]
        self.telemetry = built["telemetry"]
        self.capabilities = config.resolved_capabilities()
        self.world_model = self.agent.world_model
        self.loop = self.agent.loop
        self.state = AgentState(goal=built["goal"], max_steps=int(config.max_steps))
        self.state.scratch["runtime_capabilities"] = self.capabilities.to_dict()
        self._preview: Optional[ProbePreview] = None
        self._candidate_objects: List[CandidateAction] = []
        self._ranked = []
        self.human_events: List[Dict[str, Any]] = []

    @property
    def done(self) -> bool:
        return bool(self.state.done or self.state.step >= self.state.max_steps)

    def _hypotheses(self) -> List[Dict[str, Any]]:
        return [asdict(item) for item in self.world_model.hypotheses()]

    def _observations(self) -> List[Dict[str, Any]]:
        return [
            {"action_name": item.action_name, "result": item.result, "metadata": item.metadata}
            for item in self.state.observations
        ]

    def _explain_ranking(self, proposer: Optional[CandidateAction], runtime, score) -> List[str]:
        notes: List[str] = []
        if proposer is not None and runtime is not None and proposer.name != runtime.name:
            notes.append(f"Runtime overrides proposer: {proposer.name} -> {runtime.name}.")
        if score is not None:
            if score.information_source:
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
                f"Runtime-enforced cost/risk/irreversibility: "
                f"{score.cost:.3f}/{score.risk:.3f}/{score.irreversibility:.3f}."
            )
        if not notes:
            notes.append("No runtime disagreement: proposer and runtime currently align.")
        return notes

    def preview(self, *, force_replan: bool = False) -> Dict[str, Any]:
        if self.done:
            return self.snapshot()
        if self._preview is not None and not force_replan:
            return self._preview.to_dict()

        candidates = list(self.loop.reasoner.propose(self.state, self.world_model))
        proposal_method = getattr(self.loop.reasoner, "hypothesis_proposals", None)
        if self.capabilities.causal_updates and callable(proposal_method):
            self.world_model.sync_hypotheses(proposal_method(self.state, self.world_model))

        if self.capabilities.causal_selection:
            ranked = rank_actions(
                candidates,
                world_model=self.world_model,
                tools=self.loop.tools,
                capabilities=self.capabilities,
            )
        else:
            ranked = [(action, None) for action in candidates]

        proposer = candidates[0] if candidates else None
        runtime = ranked[0][0] if ranked else proposer
        top_score = ranked[0][1] if ranked and ranked[0][1] is not None else None

        ranking: List[Dict[str, Any]] = []
        if self.capabilities.causal_selection:
            for action, score in ranked:
                ranking.append({"action": _action_dict(action), "score": _score_dict(score)})
        else:
            for index, action in enumerate(candidates):
                ranking.append({
                    "action": _action_dict(action),
                    "score": None,
                    "rank_reason": "causal_selection disabled; proposer order preserved",
                    "proposer_index": index,
                })

        uncertainty = self.loop.reasoner.uncertainty(self.state, self.world_model)
        preview = ProbePreview(
            preview_id=uuid.uuid4().hex,
            step=self.state.step,
            uncertainty=uncertainty,
            proposer_preference=proposer.name if proposer else None,
            runtime_preference=runtime.name if runtime else None,
            disagreement=bool(proposer is not None and runtime is not None and proposer.name != runtime.name),
            candidates=[_action_dict(item) for item in candidates],
            ranking=ranking,
            hypotheses=self._hypotheses(),
            observations=self._observations(),
            explanation=self._explain_ranking(proposer, runtime, top_score),
        )
        self._preview = preview
        self._candidate_objects = candidates
        self._ranked = ranked
        self.telemetry.event("causalrag.probe.preview", {
            "probe.session_id": self.session_id,
            "probe.step": self.state.step,
            "probe.proposer_preference": preview.proposer_preference,
            "probe.runtime_preference": preview.runtime_preference,
            "probe.disagreement": preview.disagreement,
        })
        return preview.to_dict()

    def _resolve_choice(self, selection: str) -> CandidateAction:
        if self._preview is None:
            self.preview()
        if not self._candidate_objects:
            return CandidateAction(kind=ActionKind.STOP, name="stop", rationale="No candidate action.")
        selection = str(selection or "runtime")
        if selection == "runtime":
            return self._ranked[0][0] if self._ranked else self._candidate_objects[0]
        if selection == "proposer":
            return self._candidate_objects[0]
        for action in self._candidate_objects:
            if action.name == selection:
                return action
        raise ValueError(f"unknown candidate selection: {selection}")

    def commit(self, selection: str = "runtime", *, human_note: str = "") -> Dict[str, Any]:
        if self.done:
            return self.snapshot()
        if self._preview is None:
            self.preview()

        selected = self._resolve_choice(selection)
        proposer_name = self._preview.proposer_preference if self._preview else None
        runtime_name = self._preview.runtime_preference if self._preview else None
        human_event = {
            "step": self.state.step,
            "selection": selection,
            "selected_action": selected.name,
            "proposer_preference": proposer_name,
            "runtime_preference": runtime_name,
            "human_note": str(human_note or ""),
            "overrode_runtime": bool(runtime_name and selected.name != runtime_name),
            "overrode_proposer": bool(proposer_name and selected.name != proposer_name),
        }
        self.human_events.append(human_event)
        self.telemetry.event("causalrag.probe.human_choice", {
            "probe.session_id": self.session_id,
            **human_event,
        })

        score = None
        for action, action_score in self._ranked:
            if action is selected:
                score = action_score
                break

        decision = DecisionRecord(
            step=self.state.step,
            uncertainty=self._preview.uncertainty if self._preview else None,
            candidates=list(self._candidate_objects),
            selected=selected,
            beliefs_before=self.world_model.snapshot(),
            rationale=selected.rationale,
            action_scores=[
                action_score for _action, action_score in self._ranked if action_score is not None
            ],
        )
        self.state.decisions.append(decision)

        if selected.kind == ActionKind.STOP:
            self.state.done = True
            self.state.stop_reason = selected.rationale or "human_committed_stop"
            answer = selected.arguments.get("answer")
            if answer is not None:
                self.state.scratch["answer"] = answer
            self._preview = None
            return self.snapshot()

        if selected.kind == ActionKind.WAIT:
            requested = float(selected.arguments.get("seconds", 0) or 0)
            result: Any = {
                "waited": self.state.advance_time(requested),
                "virtual_time_seconds": self.state.virtual_time_seconds,
            }
            tool_spec = None
        else:
            tool_spec = self.loop.tools.get(selected.name)
            result = self.loop.tools.execute(selected.name, selected.arguments)

        observation = Observation(action_name=selected.name, result=result)
        self.state.observations.append(observation)

        experiment_update = None
        if tool_spec is not None and tool_spec.experiment_contract is not None and self.capabilities.bayesian_updates:
            contract = expanded_experiment_contract(tool_spec.experiment_contract, self.world_model)
            experiment_update = apply_experiment_observation(
                contract,
                self.world_model,
                result,
                source=selected.name,
                metadata={
                    "step": self.state.step,
                    "probe_session_id": self.session_id,
                    "human_selection": selection,
                },
            )

        self.world_model.record_transition(Transition(
            action=selected.name,
            arguments=dict(selected.arguments),
            observation=result,
            expected_effects={
                "probe_session_id": self.session_id,
                "human_selection": selection,
                "proposer_preference": proposer_name,
                "runtime_preference": runtime_name,
                "runtime_score": _score_dict(score) if score is not None else None,
                "posterior": experiment_update.posterior if experiment_update else None,
            },
        ))

        self.telemetry.event("causalrag.probe.observation", {
            "probe.session_id": self.session_id,
            "probe.step": self.state.step,
            "probe.action": selected.name,
            "probe.result": result,
            "probe.posterior": experiment_update.posterior if experiment_update else None,
        })

        self.state.step += 1
        if self.state.step >= self.state.max_steps:
            self.state.done = True
            self.state.stop_reason = "budget_exhausted"

        self._preview = None
        self._candidate_objects = []
        self._ranked = []
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        result_proxy = type("_ProbeResult", (), {"world_model": self.world_model, "state": self.state})()
        metrics = self.environment.metrics(result_proxy)
        return {
            "session_id": self.session_id,
            "done": self.done,
            "step": self.state.step,
            "stop_reason": self.state.stop_reason,
            "config": self.config.to_dict(),
            "metrics": metrics.to_dict(),
            "hypotheses": self._hypotheses(),
            "observations": self._observations(),
            "human_events": list(self.human_events),
            "trace": self.telemetry.records(),
            "preview": self._preview.to_dict() if self._preview else None,
        }
