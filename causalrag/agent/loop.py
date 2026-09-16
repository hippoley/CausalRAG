from __future__ import annotations

from typing import Callable, Optional

from causalrag.reasoning.policy import rank_actions, select_action
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel, Transition

from .actions import ActionKind, DecisionRecord
from .state import AgentState, Observation


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
    ) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.hypothesis_updater = hypothesis_updater
        self.goal_evaluator = goal_evaluator

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        state = AgentState(goal=goal, max_steps=max_steps)

        while not state.done and state.step < state.max_steps:
            if self.goal_evaluator and self.goal_evaluator(state, self.world_model):
                state.done = True
                state.stop_reason = "goal_reached"
                break

            candidates = list(self.reasoner.propose(state, self.world_model))
            proposal_method = getattr(self.reasoner, "hypothesis_proposals", None)
            if callable(proposal_method):
                self.world_model.sync_hypotheses(
                    proposal_method(state, self.world_model)
                )

            ranked = rank_actions(candidates, world_model=self.world_model)
            if ranked:
                selected = ranked[0][0]
                action_scores = [score for _action, score in ranked]
                selected_score = ranked[0][1]
            else:
                selected = select_action(candidates, world_model=self.world_model)
                action_scores = []
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

            if selected.kind == ActionKind.STOP:
                answer = selected.arguments.get("answer")
                if answer is not None:
                    state.scratch["answer"] = answer
                state.done = True
                state.stop_reason = selected.rationale or "reasoner_stopped"
                break

            if selected.kind == ActionKind.WAIT:
                result = {"waited": selected.arguments.get("seconds", 0)}
            else:
                result = self.tools.execute(selected.name, selected.arguments)

            observation = Observation(action_name=selected.name, result=result)
            state.observations.append(observation)

            runtime_information_gain = (
                selected_score.information_gain
                if selected_score is not None
                else selected.expected_information_gain
            )
            information_source = (
                selected_score.information_source
                if selected_score is not None
                else "model_estimate"
            )

            self.world_model.record_transition(
                Transition(
                    action=selected.name,
                    arguments=selected.arguments,
                    observation=result,
                    expected_effects={
                        "goal_gain": (
                            selected_score.goal_gain
                            if selected_score is not None
                            else selected.expected_goal_gain
                        ),
                        "information_gain": runtime_information_gain,
                        "information_source": information_source,
                        "model_information_gain": selected.expected_information_gain,
                        "tests_hypotheses": list(selected.tests_hypotheses),
                        "falsification_target": selected.falsification_target,
                    },
                )
            )

            if self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)
            if self.hypothesis_updater:
                self.hypothesis_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"

        return state
