from __future__ import annotations

from typing import Callable, Optional

from causalrag.reasoning.policy import select_action
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel, Transition

from .actions import ActionKind, DecisionRecord
from .state import AgentState, Observation


BeliefUpdater = Callable[[AgentState, CausalWorldModel, DecisionRecord, Observation], None]
GoalEvaluator = Callable[[AgentState, CausalWorldModel], bool]


class CausalAgentLoop:
    """Thin runtime for goal-directed causal learning and action."""

    def __init__(
        self,
        reasoner,
        tools: Optional[ToolRegistry] = None,
        world_model: Optional[CausalWorldModel] = None,
        belief_updater: Optional[BeliefUpdater] = None,
        goal_evaluator: Optional[GoalEvaluator] = None,
    ) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.goal_evaluator = goal_evaluator

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        state = AgentState(goal=goal, max_steps=max_steps)

        while not state.done and state.step < state.max_steps:
            if self.goal_evaluator and self.goal_evaluator(state, self.world_model):
                state.done = True
                state.stop_reason = "goal_reached"
                break

            candidates = list(self.reasoner.propose(state, self.world_model))
            selected = select_action(candidates)
            uncertainty = self.reasoner.uncertainty(state, self.world_model)

            decision = DecisionRecord(
                step=state.step,
                uncertainty=uncertainty,
                candidates=candidates,
                selected=selected,
                beliefs_before=self.world_model.snapshot(),
                rationale=selected.rationale,
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

            self.world_model.record_transition(
                Transition(
                    action=selected.name,
                    arguments=selected.arguments,
                    observation=result,
                    expected_effects={
                        "goal_gain": selected.expected_goal_gain,
                        "information_gain": selected.expected_information_gain,
                    },
                )
            )

            if self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"

        return state
