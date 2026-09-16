from __future__ import annotations

from typing import Callable, Optional

from causalrag.experiments import (
    TemporalDecisionPreferences,
    apply_experiment_observation,
    best_temporal_observation_value,
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

    def __init__(self, reasoner, tools: Optional[ToolRegistry] = None, world_model: Optional[CausalWorldModel] = None, belief_updater: Optional[BeliefUpdater] = None, hypothesis_updater: Optional[HypothesisUpdater] = None, goal_evaluator: Optional[GoalEvaluator] = None, time_driver: Optional[TimeDriver] = None, temporal_decision_preferences: Optional[TemporalDecisionPreferences] = None) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.hypothesis_updater = hypothesis_updater
        self.goal_evaluator = goal_evaluator
        self.time_driver = time_driver or VirtualTimeDriver()
        self.temporal_decision_preferences = temporal_decision_preferences or TemporalDecisionPreferences()

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
            self.world_model.record_transition(
                Transition(
                    action="missed_observation_window",
                    arguments={"effect_id": effect.effect_id},
                    observation=event,
                    expected_effects={"temporal_failure": True, "belief_update": False},
                )
            )
            effect.expiry_recorded = True

    def _planned_observation_at(self, effect) -> float:
        return (
            float(effect.planned_observation_at)
            if effect.planned_observation_at is not None
            else float(effect.ready_at)
        )

    def _observation_for_effect(self, effect, rationale: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.OBSERVE,
            name=effect.observe_with,
            arguments=dict(effect.observe_arguments),
            rationale=rationale,
        )

    def _wait_for_effect(self, effect, state: AgentState, rationale: str) -> CandidateAction:
        target = self._planned_observation_at(effect)
        return CandidateAction(
            kind=ActionKind.WAIT,
            name="wait_for_effect_window",
            arguments={"seconds": max(0.0, target - state.virtual_time_seconds)},
            rationale=rationale,
        )

    def _protected_effect(self, state: AgentState):
        effects = [effect for effect in state.active_pending_effects() if effect.protect_attribution]
        if not effects:
            return None
        return min(
            effects,
            key=lambda effect: (self._planned_observation_at(effect), effect.expires_at),
        )

    def _temporal_guard(self, selected: CandidateAction, state: AgentState) -> CandidateAction:
        now = state.virtual_time_seconds

        matching = [
            effect
            for effect in state.active_pending_effects()
            if effect.observe_with == selected.name
            and now < self._planned_observation_at(effect)
        ]
        if matching:
            effect = min(matching, key=self._planned_observation_at)
            return self._wait_for_effect(
                effect,
                state,
                f"Runtime temporal decision: {selected.name} is more valuable at the planned causal observation time.",
            )

        protected = self._protected_effect(state)
        if protected is None:
            return selected

        target = self._planned_observation_at(protected)
        if selected.kind in (ActionKind.INTERVENE, ActionKind.STOP):
            if now >= target:
                return self._observation_for_effect(
                    protected,
                    "Runtime attribution guard: observe the unresolved intervention effect before another intervention or stop.",
                )
            return self._wait_for_effect(
                protected,
                state,
                "Runtime attribution guard: wait until the selected causal observation time before another intervention or stop.",
            )

        if selected.kind == ActionKind.WAIT:
            if now >= target:
                return self._observation_for_effect(
                    protected,
                    "Runtime temporal decision: the selected observation time has arrived.",
                )
            requested = float(selected.arguments.get("seconds", 0) or 0)
            until_target = max(0.0, target - now)
            if requested <= 0.0 or requested > until_target:
                return self._wait_for_effect(
                    protected,
                    state,
                    "Runtime temporal decision: cap WAIT at the selected observation time.",
                )

        return selected

    def _plan_temporal_effect(self, effect, state: AgentState) -> None:
        if not effect.observation_points:
            return
        plan = best_temporal_observation_value(
            effect=effect,
            world_model=self.world_model,
            tools=self.tools,
            now=state.virtual_time_seconds,
            preferences=self.temporal_decision_preferences,
        )
        if plan is None:
            return
        effect.planned_observation_at = plan.observation_at
        effect.planned_experiment_contract = plan.experiment_contract
        effect.planned_temporal_value = plan.net_value
        effect.metadata["temporal_decision"] = {
            "experiment_id": plan.experiment_id,
            "offset_seconds": plan.offset_seconds,
            "observation_at": plan.observation_at,
            "wait_seconds": plan.wait_seconds,
            "evsi": plan.evsi,
            "measurement_cost": plan.measurement_cost,
            "wait_cost": plan.wait_cost,
            "net_value": plan.net_value,
        }

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
        self._plan_temporal_effect(effect, state)
        state.schedule_effect(effect)

    def _evaluate_temporal_observation(self, selected: CandidateAction, result, state: AgentState):
        evaluations = []
        now = state.virtual_time_seconds
        for effect in state.pending_effects:
            effect.refresh(now)
            if effect.observed or effect.expired or effect.observe_with != selected.name:
                continue
            target = self._planned_observation_at(effect)
            if now < target:
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
                "planned_observation_at": effect.planned_observation_at,
                "planned_temporal_value": effect.planned_temporal_value,
            }

            bayesian_update = None
            if effect.planned_experiment_contract is not None:
                bayesian_update = apply_experiment_observation(
                    effect.planned_experiment_contract,
                    self.world_model,
                    result,
                    source=selected.name,
                    metadata={
                        "effect_id": effect.effect_id,
                        "lag_seconds": now - effect.started_at,
                        "temporal_decision": True,
                    },
                )
                if bayesian_update is not None:
                    evaluation["experiment_id"] = bayesian_update.experiment_id
                    evaluation["posterior"] = bayesian_update.posterior
                    evaluation["expected_information_gain"] = bayesian_update.expected_information_gain

            if bayesian_update is None and hypothesis_id and expected is not None:
                weight = effect.falsification_weight if matched else -effect.falsification_weight
                self.world_model.update_hypothesis(
                    str(hypothesis_id),
                    Evidence(
                        source=selected.name,
                        statement=(
                            f"Temporal effect {effect.effect_id}: expected {expected!r}, "
                            f"observed {observed!r} after {now - effect.started_at:.3f}s."
                        ),
                        weight=weight,
                        kind="temporal_intervention_outcome",
                        metadata=evaluation,
                    ),
                )
            evaluations.append(evaluation)
        return evaluations

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

            if selected.kind == ActionKind.STOP:
                answer = selected.arguments.get("answer")
                if answer is not None:
                    state.scratch["answer"] = answer
                state.done = True
                state.stop_reason = selected.rationale or "reasoner_stopped"
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

            experiment_update = None
            temporal_bayesian_update = any(
                "posterior" in evaluation for evaluation in temporal_evaluations
            )
            if (
                not temporal_bayesian_update
                and tool_spec is not None
                and tool_spec.experiment_contract is not None
            ):
                experiment_update = apply_experiment_observation(
                    tool_spec.experiment_contract,
                    self.world_model,
                    result,
                    source=selected.name,
                    metadata={"step": state.step},
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
            }
            if experiment_update is not None:
                expected_effects["experiment_id"] = experiment_update.experiment_id
                expected_effects["observed_outcome"] = experiment_update.outcome
                expected_effects["posterior"] = experiment_update.posterior
            if temporal_evaluations:
                expected_effects["temporal_effects"] = temporal_evaluations

            self.world_model.record_transition(Transition(action=selected.name, arguments=selected.arguments, observation=result, expected_effects=expected_effects))

            if self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)
            if self.hypothesis_updater and experiment_update is None and not temporal_evaluations:
                self.hypothesis_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"
        return state
