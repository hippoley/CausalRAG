from __future__ import annotations

from typing import Callable, Optional

from causalrag.experiments import (
    ModelMismatchPolicy,
    apply_experiment_observation,
    assess_model_mismatch,
    expanded_experiment_contract,
    maybe_resolve_model_mismatch,
)
from causalrag.reasoning.policy import rank_actions, select_action
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel, Evidence, Transition

from .actions import ActionKind, CandidateAction, DecisionRecord
from .state import AgentState, Observation
from .temporal import TimeDriver, VirtualTimeDriver, pending_effect_from_contract

BeliefUpdater = Callable[[AgentState, CausalWorldModel, DecisionRecord, Observation], None]
HypothesisUpdater = Callable[[AgentState, CausalWorldModel, DecisionRecord, Observation], None]
GoalEvaluator = Callable[[AgentState, CausalWorldModel], bool]


class CausalAgentLoop:
    """Thin runtime for goal-directed causal learning and action."""

    def __init__(
        self,
        reasoner,
        tools: Optional[ToolRegistry] = None,
        world_model: Optional[CausalWorldModel] = None,
        belief_updater: Optional[BeliefUpdater] = None,
        hypothesis_updater: Optional[HypothesisUpdater] = None,
        goal_evaluator: Optional[GoalEvaluator] = None,
        time_driver: Optional[TimeDriver] = None,
        mismatch_policy: Optional[ModelMismatchPolicy] = None,
    ) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.hypothesis_updater = hypothesis_updater
        self.goal_evaluator = goal_evaluator
        self.time_driver = time_driver or VirtualTimeDriver()
        self.mismatch_policy = mismatch_policy or ModelMismatchPolicy()

    def _event(self, name: str, attributes: Optional[dict] = None) -> None:
        """Emit a causal event at the moment it happens, when telemetry is attached."""
        telemetry = getattr(self.tools, "telemetry", None)
        if telemetry is not None:
            telemetry.event(name, attributes or {})

    def _sync_state_time(self, state: AgentState) -> None:
        state.virtual_time_seconds = float(self.time_driver.now_seconds)
        for effect in state.pending_effects:
            effect.refresh(state.virtual_time_seconds)

    def _record_expired_effects(self, state: AgentState) -> None:
        now = state.virtual_time_seconds
        for effect in state.pending_effects:
            effect.refresh(now)
            if not effect.expired or effect.expiry_recorded:
                continue
            event = {
                "kind": "missed_observation_window",
                "effect_id": effect.effect_id,
                "intervention": effect.intervention,
                "observe_with": effect.observe_with,
                "ready_at": effect.ready_at,
                "expires_at": effect.expires_at,
                "detected_at": now,
            }
            state.scratch.setdefault("temporal_events", []).append(event)
            self._event(
                "causalrag.temporal_window_missed",
                {
                    "causalrag.temporal.effect_id": effect.effect_id,
                    "causalrag.action.name": effect.intervention,
                    "causalrag.temporal.observe_with": effect.observe_with,
                    "causalrag.temporal.ready_at": effect.ready_at,
                    "causalrag.temporal.expires_at": effect.expires_at,
                    "causalrag.virtual_time_seconds": now,
                },
            )
            self.world_model.record_transition(
                Transition(
                    action="missed_observation_window",
                    arguments={"effect_id": effect.effect_id},
                    observation=event,
                    expected_effects={"temporal_failure": True, "belief_update": False},
                )
            )
            effect.expiry_recorded = True

    def _observation_for_effect(self, effect, rationale: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.OBSERVE,
            name=effect.observe_with,
            arguments=dict(effect.observe_arguments),
            rationale=rationale,
        )

    def _wait_for_effect(self, effect, state: AgentState, rationale: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.WAIT,
            name="wait_for_effect_window",
            arguments={"seconds": effect.seconds_until_ready(state.virtual_time_seconds)},
            rationale=rationale,
        )

    def _protected_effect(self, state: AgentState):
        effects = [effect for effect in state.active_pending_effects() if effect.protect_attribution]
        if not effects:
            return None
        return min(effects, key=lambda effect: (effect.expires_at, effect.ready_at))

    def _temporal_guard(self, selected: CandidateAction, state: AgentState) -> CandidateAction:
        now = state.virtual_time_seconds
        matching = [
            effect
            for effect in state.active_pending_effects()
            if effect.observe_with == selected.name and effect.is_premature(now)
        ]
        if matching:
            effect = min(matching, key=lambda item: item.ready_at)
            return self._wait_for_effect(effect, state, f"Runtime temporal guard: {selected.name} is premature; wait for the causal observation window.")

        protected = self._protected_effect(state)
        if protected is None:
            return selected

        if selected.kind in (ActionKind.INTERVENE, ActionKind.STOP):
            if protected.is_ready(now):
                return self._observation_for_effect(protected, "Runtime attribution guard: observe the unresolved intervention effect before another intervention or stop.")
            return self._wait_for_effect(protected, state, "Runtime attribution guard: wait for the unresolved intervention effect before another intervention or stop.")

        if selected.kind == ActionKind.WAIT:
            if protected.is_ready(now):
                return self._observation_for_effect(protected, "Runtime attribution guard: the protected effect is ready; observe it before waiting longer.")
            requested = float(selected.arguments.get("seconds", 0) or 0)
            until_ready = protected.seconds_until_ready(now)
            if requested <= 0.0 or requested > until_ready:
                return self._wait_for_effect(protected, state, "Runtime attribution guard: cap WAIT at the first valid observation time.")
        return selected

    def _schedule_temporal_effect(self, tool_spec, selected: CandidateAction, state: AgentState) -> None:
        contract = None if tool_spec is None else tool_spec.temporal_effect_contract
        if selected.kind != ActionKind.INTERVENE or contract is None:
            return
        effect = pending_effect_from_contract(selected.name, contract, state.virtual_time_seconds)
        active = self.world_model.hypotheses(include_rejected=False)
        if active:
            prediction = max(active, key=lambda item: item.probability)
            effect.metadata["prediction_hypothesis"] = prediction.hypothesis_id
            effect.metadata["prediction_probability"] = prediction.probability
            effect.metadata["expected_outcome"] = effect.expected_for(prediction.hypothesis_id)
        state.schedule_effect(effect)
        self._event(
            "causalrag.temporal_effect_scheduled",
            {
                "causalrag.step": state.step,
                "causalrag.temporal.effect_id": effect.effect_id,
                "causalrag.action.name": selected.name,
                "causalrag.temporal.ready_at": effect.ready_at,
                "causalrag.temporal.expires_at": effect.expires_at,
            },
        )

    def _evaluate_temporal_observation(self, selected: CandidateAction, result, state: AgentState):
        evaluations = []
        now = state.virtual_time_seconds
        for effect in state.pending_effects:
            effect.refresh(now)
            if effect.observed or effect.expired or effect.observe_with != selected.name:
                continue
            if not effect.is_ready(now):
                continue
            if not isinstance(result, dict) or effect.observation_key not in result:
                continue
            observed = result[effect.observation_key]
            hypothesis_id = effect.metadata.get("prediction_hypothesis")
            expected = effect.expected_for(str(hypothesis_id)) if hypothesis_id else None
            matched = expected is None or observed == expected
            effect.observed = True
            effect.observed_value = observed
            effect.matched_prediction = matched
            evaluation = {
                "effect_id": effect.effect_id,
                "intervention": effect.intervention,
                "prediction_hypothesis": hypothesis_id,
                "expected": expected,
                "observed": observed,
                "matched_prediction": matched,
                "lag_seconds": now - effect.started_at,
                "within_window": effect.ready_at <= now <= effect.expires_at,
            }
            evaluations.append(evaluation)
            self._event(
                "causalrag.temporal_attribution",
                {
                    "causalrag.step": state.step,
                    "causalrag.temporal.effect_id": effect.effect_id,
                    "causalrag.action.name": effect.intervention,
                    "causalrag.temporal.prediction_hypothesis": hypothesis_id,
                    "causalrag.temporal.matched_prediction": matched,
                    "causalrag.temporal.lag_seconds": evaluation["lag_seconds"],
                    "causalrag.temporal.within_window": evaluation["within_window"],
                },
            )
            if hypothesis_id and expected is not None:
                weight = effect.falsification_weight if matched else -effect.falsification_weight
                self.world_model.update_hypothesis(
                    str(hypothesis_id),
                    Evidence(
                        source=selected.name,
                        statement=f"Temporal effect {effect.effect_id}: expected {expected!r}, observed {observed!r} after {now - effect.started_at:.3f}s.",
                        weight=weight,
                        kind="temporal_intervention_outcome",
                        metadata=evaluation,
                    ),
                )
        return evaluations

    def _discover_after_mismatch(self, state: AgentState, assessment) -> list[str]:
        discover = getattr(self.reasoner, "discover_hypotheses", None)
        if not callable(discover) or assessment is None or not assessment.escalate:
            return []
        unresolved_provisional = [
            hypothesis
            for hypothesis in self.world_model.hypotheses(include_rejected=False)
            if hypothesis.origin == "discovered" and not hypothesis.validated
        ]
        if unresolved_provisional:
            return []

        context = {
            "experiment_id": assessment.experiment_id,
            "outcome": assessment.outcome,
            "predictive_probability": assessment.predictive_probability,
            "surprisal": assessment.surprisal,
            "hard_mismatch": assessment.hard_mismatch,
            "recent_mismatches": self.world_model.snapshot().get("open_world", {}).get("recent_mismatches", []),
        }
        proposals = list(discover(state, self.world_model, context))
        added: list[str] = []
        for proposal in proposals:
            hypothesis = self.world_model.add_discovered_hypothesis(
                hypothesis_id=getattr(proposal, "hypothesis_id", ""),
                statement=getattr(proposal, "statement", ""),
                rationale=getattr(proposal, "rationale", ""),
                falsifiers=getattr(proposal, "falsifiers", []),
                experiment_predictions=getattr(proposal, "experiment_predictions", {}),
                initial_probability=self.mismatch_policy.discovered_initial_probability,
            )
            if hypothesis is not None:
                added.append(hypothesis.hypothesis_id)

        if added:
            discovery_event = {"step": state.step, "trigger": context, "hypotheses": list(added)}
            state.scratch.setdefault("hypothesis_discovery_events", []).append(discovery_event)
            self._event(
                "causalrag.hypothesis_discovery",
                {
                    "causalrag.step": state.step,
                    "causalrag.discovery.hypothesis_ids": list(added),
                    "causalrag.discovery.trigger_experiment": assessment.experiment_id,
                    "causalrag.discovery.trigger_surprisal": assessment.surprisal,
                },
            )
        return added

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        state = AgentState(goal=goal, max_steps=max_steps)
        self._sync_state_time(state)
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

            ranked = rank_actions(candidates, world_model=self.world_model, tools=self.tools)
            if ranked:
                proposed_selected = ranked[0][0]
                selected_score = ranked[0][1]
                action_scores = [score for _action, score in ranked]
            else:
                proposed_selected = select_action(candidates, world_model=self.world_model, tools=self.tools)
                selected_score = None
                action_scores = []

            selected = self._temporal_guard(proposed_selected, state)
            if selected is not proposed_selected:
                selected_score = None

            uncertainty = self.reasoner.uncertainty(state, self.world_model)
            decision = DecisionRecord(step=state.step, uncertainty=uncertainty, candidates=candidates, selected=selected, beliefs_before=self.world_model.snapshot(), rationale=selected.rationale, action_scores=action_scores)
            state.decisions.append(decision)
            decision_attributes = {
                "causalrag.step": state.step,
                "causalrag.action.name": selected.name,
                "causalrag.action.kind": selected.kind.value,
                "causalrag.candidate.count": len(candidates),
                "causalrag.action.tests_hypotheses": list(selected.tests_hypotheses),
            }
            if selected_score is not None:
                decision_attributes.update(
                    {
                        "causalrag.decision.total_utility": selected_score.total_utility,
                        "causalrag.decision.information_gain": selected_score.information_gain,
                        "causalrag.decision.information_source": selected_score.information_source,
                        "causalrag.decision.cost": selected_score.cost,
                        "causalrag.decision.risk": selected_score.risk,
                        "causalrag.decision.irreversibility": selected_score.irreversibility,
                    }
                )
                if selected_score.expected_value_of_sample_information is not None:
                    decision_attributes["causalrag.decision.evsi"] = selected_score.expected_value_of_sample_information
            self._event("causalrag.decision", decision_attributes)

            if selected.kind == ActionKind.STOP:
                answer = selected.arguments.get("answer")
                if answer is not None:
                    state.scratch["answer"] = answer
                state.done = True
                state.stop_reason = selected.rationale or "reasoner_stopped"
                self._event("causalrag.stop", {"causalrag.step": state.step, "causalrag.stop_reason": state.stop_reason})
                break

            if selected.kind == ActionKind.WAIT:
                requested = float(selected.arguments.get("seconds", 0) or 0)
                if requested <= 0.0:
                    requested = state.next_effect_ready_in() or 0.0
                waited = float(self.time_driver.advance(requested))
                self._sync_state_time(state)
                self._record_expired_effects(state)
                result = {"waited": waited, "virtual_time_seconds": state.virtual_time_seconds}
                tool_spec = None
            else:
                tool_spec = self.tools.get(selected.name)
                result = self.tools.execute(selected.name, selected.arguments)

            self._schedule_temporal_effect(tool_spec, selected, state)
            temporal_evaluations = self._evaluate_temporal_observation(selected, result, state)

            observation_metadata = {}
            if temporal_evaluations:
                observation_metadata["temporal_effects"] = temporal_evaluations
            observation = Observation(action_name=selected.name, result=result, metadata=observation_metadata)
            state.observations.append(observation)
            self._event(
                "causalrag.observation",
                {
                    "causalrag.step": state.step,
                    "causalrag.action.name": selected.name,
                    "causalrag.observation.has_temporal_effects": bool(temporal_evaluations),
                },
            )

            experiment_update = None
            mismatch_assessment = None
            discovered_hypotheses: list[str] = []
            if tool_spec is not None and tool_spec.experiment_contract is not None:
                experiment_contract = expanded_experiment_contract(tool_spec.experiment_contract, self.world_model)
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
                    self._event(
                        "causalrag.model_mismatch",
                        {
                            "causalrag.step": state.step,
                            "causalrag.experiment.id": mismatch_assessment.experiment_id,
                            "causalrag.experiment.outcome": mismatch_assessment.outcome,
                            "causalrag.predictive_probability": mismatch_assessment.predictive_probability,
                            "causalrag.surprisal": mismatch_assessment.surprisal,
                            "causalrag.model_mismatch.suspicious": mismatch_assessment.suspicious,
                            "causalrag.model_mismatch.hard": mismatch_assessment.hard_mismatch,
                            "causalrag.model_mismatch.escalated": mismatch_assessment.escalate,
                            "causalrag.model_mismatch.posterior_suppressed": mismatch_assessment.suppress_closed_world_posterior,
                        },
                    )

                if not (mismatch_assessment is not None and mismatch_assessment.suppress_closed_world_posterior):
                    experiment_update = apply_experiment_observation(
                        experiment_contract,
                        self.world_model,
                        result,
                        source=selected.name,
                        metadata={"step": state.step},
                    )
                    if experiment_update is not None:
                        self._event(
                            "causalrag.posterior.updated",
                            {
                                "causalrag.step": state.step,
                                "causalrag.experiment.id": experiment_update.experiment_id,
                                "causalrag.experiment.outcome": experiment_update.outcome,
                                "causalrag.predictive_probability": experiment_update.predictive_probability,
                                "causalrag.surprisal": experiment_update.surprisal,
                                "causalrag.posterior": experiment_update.posterior,
                            },
                        )
                        maybe_resolve_model_mismatch(self.world_model)
                else:
                    discovered_hypotheses = self._discover_after_mismatch(state, mismatch_assessment)

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
            }
            if experiment_update is not None:
                expected_effects.update(
                    {
                        "experiment_id": experiment_update.experiment_id,
                        "observed_outcome": experiment_update.outcome,
                        "posterior": experiment_update.posterior,
                        "predictive_probability": experiment_update.predictive_probability,
                        "surprisal": experiment_update.surprisal,
                    }
                )
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

            self.world_model.record_transition(Transition(action=selected.name, arguments=selected.arguments, observation=result, expected_effects=expected_effects))

            if self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)
            mismatch_escalated = bool(mismatch_assessment and mismatch_assessment.escalate)
            if self.hypothesis_updater and experiment_update is None and not temporal_evaluations and not mismatch_escalated:
                self.hypothesis_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"
            self._event("causalrag.stop", {"causalrag.step": state.step, "causalrag.stop_reason": state.stop_reason})
        return state
