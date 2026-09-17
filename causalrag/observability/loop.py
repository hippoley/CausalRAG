from __future__ import annotations

from typing import Optional

from causalrag.agent.actions import ActionKind, DecisionRecord
from causalrag.agent.features import RuntimeFeatureFlags
from causalrag.agent.loop import CausalAgentLoop
from causalrag.agent.state import AgentState, Observation
from causalrag.experiments import (
    apply_experiment_observation,
    assess_model_mismatch,
    expanded_experiment_contract,
    maybe_resolve_model_mismatch,
)
from causalrag.reasoning.policy import rank_actions, select_action
from causalrag.world_model.models import Transition

from .events import ProbeEmitter, ProbeEventSink


class ObservableCausalAgentLoop(CausalAgentLoop):
    """CausalAgentLoop that emits a replayable causal decision event stream."""

    def __init__(
        self,
        *args,
        event_sink: Optional[ProbeEventSink] = None,
        features: Optional[RuntimeFeatureFlags] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.event_sink = event_sink
        self.features = features or RuntimeFeatureFlags()
        self.last_run_id: Optional[str] = None

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        emitter = ProbeEmitter(self.event_sink)
        self.last_run_id = emitter.run_id
        state = AgentState(goal=goal, max_steps=max_steps)
        self._sync_state_time(state)
        state.scratch["run_id"] = emitter.run_id
        state.scratch["runtime_features"] = self.features.to_dict()
        emitter.emit(
            "run.started",
            payload={
                "goal": goal,
                "max_steps": max_steps,
                "features": self.features.to_dict(),
                "hypotheses": self.world_model.snapshot().get("hypotheses", []),
            },
        )

        while not state.done and state.step < state.max_steps:
            self._sync_state_time(state)
            self._record_expired_effects(state)
            if self.goal_evaluator and self.goal_evaluator(state, self.world_model):
                state.done = True
                state.stop_reason = "goal_reached"
                break

            candidates = list(self.reasoner.propose(state, self.world_model))
            proposal_method = getattr(self.reasoner, "hypothesis_proposals", None)
            if callable(proposal_method):
                self.world_model.sync_hypotheses(proposal_method(state, self.world_model))

            emitter.emit(
                "proposal",
                step=state.step,
                payload={
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "hypotheses": self.world_model.snapshot().get("hypotheses", []),
                },
            )

            ranked = rank_actions(
                candidates,
                world_model=self.world_model,
                tools=self.tools,
                features=self.features,
            )
            if ranked:
                proposed_selected = ranked[0][0]
                selected_score = ranked[0][1]
                action_scores = [score for _action, score in ranked]
            else:
                proposed_selected = select_action(
                    candidates,
                    world_model=self.world_model,
                    tools=self.tools,
                    features=self.features,
                )
                selected_score = None
                action_scores = []

            if self.features.temporal_attribution:
                selected = self._temporal_guard(proposed_selected, state)
                temporal_guard_applied = selected is not proposed_selected
            else:
                selected = proposed_selected
                temporal_guard_applied = False
            if temporal_guard_applied:
                selected_score = None

            uncertainty = self.reasoner.uncertainty(state, self.world_model)
            decision = DecisionRecord(
                step=state.step,
                uncertainty=uncertainty,
                candidates=candidates,
                selected=selected,
                beliefs_before=self.world_model.snapshot(),
                rationale=selected.rationale,
                action_scores=action_scores,
            )
            state.decisions.append(decision)
            emitter.emit(
                "decision",
                step=state.step,
                payload={
                    "uncertainty": uncertainty,
                    "selected": selected,
                    "action_scores": action_scores,
                    "temporal_guard_applied": temporal_guard_applied,
                    "features": self.features.to_dict(),
                },
            )

            if selected.kind == ActionKind.STOP:
                answer = selected.arguments.get("answer")
                if answer is not None:
                    state.scratch["answer"] = answer
                state.done = True
                state.stop_reason = selected.rationale or "reasoner_stopped"
                emitter.emit(
                    "run.completed",
                    step=state.step,
                    payload={
                        "stop_reason": state.stop_reason,
                        "answer": answer,
                        "world_model": self.world_model.snapshot(),
                        "features": self.features.to_dict(),
                    },
                )
                break

            operation_id = f"step-{state.step}:{selected.name}"
            if selected.kind == ActionKind.WAIT:
                requested = float(selected.arguments.get("seconds", 0) or 0)
                if requested <= 0.0:
                    requested = state.next_effect_ready_in() or 0.0
                emitter.emit(
                    "wait.started",
                    step=state.step,
                    payload={
                        "operation_id": operation_id,
                        "action_name": selected.name,
                        "requested_seconds": requested,
                    },
                )
                waited = float(self.time_driver.advance(requested))
                self._sync_state_time(state)
                self._record_expired_effects(state)
                result = {
                    "waited": waited,
                    "virtual_time_seconds": state.virtual_time_seconds,
                }
                tool_spec = None
                emitter.emit(
                    "wait.completed",
                    step=state.step,
                    payload={
                        "operation_id": operation_id,
                        "action_name": selected.name,
                        "waited_seconds": waited,
                        "virtual_time_seconds": state.virtual_time_seconds,
                    },
                )
            else:
                tool_spec = self.tools.get(selected.name)
                emitter.emit(
                    "tool.started",
                    step=state.step,
                    payload={
                        "operation_id": operation_id,
                        "action_name": selected.name,
                        "action_kind": selected.kind.value,
                        "arguments": selected.arguments,
                        "cost": tool_spec.cost,
                        "risk": tool_spec.risk,
                        "reversible": tool_spec.reversible,
                    },
                )
                try:
                    result = self.tools.execute(selected.name, selected.arguments)
                except Exception as exc:
                    emitter.emit(
                        "tool.failed",
                        step=state.step,
                        payload={
                            "operation_id": operation_id,
                            "action_name": selected.name,
                            "error": str(exc),
                        },
                    )
                    emitter.emit(
                        "run.completed",
                        step=state.step,
                        payload={"stop_reason": "tool_error", "error": str(exc)},
                    )
                    raise
                emitter.emit(
                    "tool.completed",
                    step=state.step,
                    payload={
                        "operation_id": operation_id,
                        "action_name": selected.name,
                        "result": result,
                    },
                )

            temporal_evaluations = []
            if self.features.temporal_attribution:
                self._schedule_temporal_effect(tool_spec, selected, state)
                temporal_evaluations = self._evaluate_temporal_observation(selected, result, state)
                for evaluation in temporal_evaluations:
                    emitter.emit("attribution", step=state.step, payload=evaluation)

            observation_metadata = {}
            if temporal_evaluations:
                observation_metadata["temporal_effects"] = temporal_evaluations
            observation = Observation(action_name=selected.name, result=result, metadata=observation_metadata)
            state.observations.append(observation)
            emitter.emit(
                "observation",
                step=state.step,
                payload={"action_name": selected.name, "result": result},
            )

            experiment_update = None
            mismatch_assessment = None
            discovered_hypotheses: list[str] = []
            if self.features.causal_runtime and tool_spec is not None and tool_spec.experiment_contract is not None:
                experiment_contract = expanded_experiment_contract(
                    tool_spec.experiment_contract,
                    self.world_model,
                )

                if self.features.open_world_discovery:
                    mismatch_assessment = assess_model_mismatch(
                        experiment_contract,
                        self.world_model,
                        result,
                        policy=self.mismatch_policy,
                        metadata={"step": state.step, "action": selected.name},
                    )
                    if mismatch_assessment is not None:
                        observation.metadata["model_mismatch"] = {
                            "predictive_probability": mismatch_assessment.predictive_probability,
                            "surprisal": mismatch_assessment.surprisal,
                            "suspicious": mismatch_assessment.suspicious,
                            "hard_mismatch": mismatch_assessment.hard_mismatch,
                            "escalate": mismatch_assessment.escalate,
                            "mismatch_id": mismatch_assessment.mismatch_id,
                        }
                        if mismatch_assessment.suspicious:
                            emitter.emit(
                                "model_mismatch",
                                step=state.step,
                                payload=observation.metadata["model_mismatch"],
                            )

                should_suppress = bool(
                    mismatch_assessment is not None
                    and mismatch_assessment.suppress_closed_world_posterior
                )
                if not should_suppress:
                    experiment_update = apply_experiment_observation(
                        experiment_contract,
                        self.world_model,
                        result,
                        source=selected.name,
                        metadata={"step": state.step},
                    )
                    if experiment_update is not None:
                        maybe_resolve_model_mismatch(self.world_model)
                        emitter.emit(
                            "posterior.updated",
                            step=state.step,
                            payload={
                                "experiment_id": experiment_update.experiment_id,
                                "outcome": experiment_update.outcome,
                                "prior": experiment_update.prior,
                                "posterior": experiment_update.posterior,
                                "predictive_probability": experiment_update.predictive_probability,
                                "surprisal": experiment_update.surprisal,
                            },
                        )
                elif self.features.open_world_discovery:
                    discovered_hypotheses = self._discover_after_mismatch(state, mismatch_assessment)
                    if discovered_hypotheses:
                        emitter.emit(
                            "hypothesis.discovered",
                            step=state.step,
                            payload={
                                "hypothesis_ids": discovered_hypotheses,
                                "trigger": observation.metadata.get("model_mismatch", {}),
                                "hypotheses": self.world_model.snapshot().get("hypotheses", []),
                            },
                        )

            runtime_information_gain = selected_score.information_gain if selected_score is not None else selected.expected_information_gain
            information_source = selected_score.information_source if selected_score is not None else "model_estimate"
            expected_effects = {
                "goal_gain": selected_score.goal_gain if selected_score is not None else selected.expected_goal_gain,
                "information_gain": runtime_information_gain,
                "information_source": information_source,
                "model_information_gain": selected.expected_information_gain,
                "tests_hypotheses": list(selected.tests_hypotheses),
                "falsification_target": selected.falsification_target,
                "virtual_time_seconds": state.virtual_time_seconds,
                "runtime_features": self.features.to_dict(),
            }
            if experiment_update is not None:
                expected_effects["experiment_id"] = experiment_update.experiment_id
                expected_effects["observed_outcome"] = experiment_update.outcome
                expected_effects["posterior"] = experiment_update.posterior
                expected_effects["predictive_probability"] = experiment_update.predictive_probability
                expected_effects["surprisal"] = experiment_update.surprisal
            if mismatch_assessment is not None:
                expected_effects["model_mismatch"] = {
                    "predictive_probability": mismatch_assessment.predictive_probability,
                    "surprisal": mismatch_assessment.surprisal,
                    "suspicious": mismatch_assessment.suspicious,
                    "hard_mismatch": mismatch_assessment.hard_mismatch,
                    "escalate": mismatch_assessment.escalate,
                    "posterior_suppressed": mismatch_assessment.suppress_closed_world_posterior,
                    "discovered_hypotheses": discovered_hypotheses,
                }
            if temporal_evaluations:
                expected_effects["temporal_effects"] = temporal_evaluations

            self.world_model.record_transition(
                Transition(
                    action=selected.name,
                    arguments=selected.arguments,
                    observation=result,
                    expected_effects=expected_effects,
                )
            )

            if self.features.causal_runtime and self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)
            mismatch_escalated = bool(mismatch_assessment and mismatch_assessment.escalate)
            if (
                self.features.causal_runtime
                and self.hypothesis_updater
                and experiment_update is None
                and not temporal_evaluations
                and not mismatch_escalated
            ):
                self.hypothesis_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"
            emitter.emit(
                "run.completed",
                step=state.step,
                payload={
                    "stop_reason": state.stop_reason,
                    "world_model": self.world_model.snapshot(),
                    "features": self.features.to_dict(),
                },
            )
        return state
