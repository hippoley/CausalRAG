from __future__ import annotations

from typing import Callable, Optional

from causalrag.experiments import apply_experiment_observation
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

    def __init__(self, reasoner, tools: Optional[ToolRegistry] = None, world_model: Optional[CausalWorldModel] = None, belief_updater: Optional[BeliefUpdater] = None, hypothesis_updater: Optional[HypothesisUpdater] = None, goal_evaluator: Optional[GoalEvaluator] = None, time_driver: Optional[TimeDriver] = None) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.hypothesis_updater = hypothesis_updater
        self.goal_evaluator = goal_evaluator
        self.time_driver = time_driver or VirtualTimeDriver()

    def _sync_state_time(self, state: AgentState) -> None:
        state.virtual_time_seconds = float(self.time_driver.now_seconds)
        for effect in state.pending_effects:
            effect.refresh(state.virtual_time_seconds)

    def _temporal_guard(self, selected: CandidateAction, state: AgentState) -> CandidateAction:
        if selected.kind in (ActionKind.STOP, ActionKind.WAIT):
            return selected
        matching = [
            effect
            for effect in state.active_pending_effects()
            if effect.observe_with == selected.name and effect.is_premature(state.virtual_time_seconds)
        ]
        if not matching:
            return selected
        wait_seconds = min(
            effect.seconds_until_ready(state.virtual_time_seconds) for effect in matching
        )
        return CandidateAction(
            kind=ActionKind.WAIT,
            name="wait_for_effect_window",
            arguments={"seconds": wait_seconds},
            rationale=(
                f"Runtime temporal guard: {selected.name} is premature; "
                f"wait {wait_seconds:.3f}s for the causal observation window."
            ),
        )

    def _schedule_temporal_effect(self, tool_spec, selected: CandidateAction, state: AgentState) -> None:
        contract = None if tool_spec is None else tool_spec.temporal_effect_contract
        if selected.kind != ActionKind.INTERVENE or contract is None:
            return
        effect = pending_effect_from_contract(
            selected.name,
            contract,
            state.virtual_time_seconds,
        )
        active = self.world_model.hypotheses(include_rejected=False)
        if active:
            prediction = max(active, key=lambda item: item.probability)
            effect.metadata["prediction_hypothesis"] = prediction.hypothesis_id
            effect.metadata["prediction_probability"] = prediction.probability
            effect.metadata["expected_outcome"] = effect.expected_for(prediction.hypothesis_id)
        state.schedule_effect(effect)

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
            if hypothesis_id and expected is not None:
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
        return evaluations

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        state = AgentState(goal=goal, max_steps=max_steps)
        self._sync_state_time(state)
        while not state.done and state.step < state.max_steps:
            self._sync_state_time(state)
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
                result = {
                    "waited": waited,
                    "virtual_time_seconds": state.virtual_time_seconds,
                }
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
            if tool_spec is not None and tool_spec.experiment_contract is not None:
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
