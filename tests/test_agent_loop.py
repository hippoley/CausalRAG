from branchpoint.agent import ActionKind, CandidateAction, CausalAgentLoop
from branchpoint.tools import ToolRegistry, ToolSpec


class TwoStepReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_sensor",
                    expected_information_gain=0.8,
                    cost=0.1,
                    rationale="Reduce uncertainty before intervention.",
                )
            ]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", rationale="enough evidence")]

    def uncertainty(self, state, world_model):
        return "sensor state unknown" if state.step == 0 else None


def test_loop_records_decision_observation_and_transition():
    tools = ToolRegistry([ToolSpec(name="read_sensor", description="read", handler=lambda: 42)])
    agent = CausalAgentLoop(reasoner=TwoStepReasoner(), tools=tools)

    result = agent.run("understand the sensor", max_steps=3)

    assert result.done is True
    assert result.stop_reason == "enough evidence"
    assert len(result.decisions) == 2
    assert result.observations[0].result == 42
    assert len(agent.world_model.transitions) == 1


def test_policy_prefers_information_gain_when_goal_gain_equal():
    class Reasoner:
        def propose(self, state, world_model):
            return [
                CandidateAction(ActionKind.OBSERVE, "low_info", expected_information_gain=0.1),
                CandidateAction(ActionKind.OBSERVE, "high_info", expected_information_gain=0.7),
            ]

        def uncertainty(self, state, world_model):
            return "unknown"

    tools = ToolRegistry([
        ToolSpec(name="low_info", description="", handler=lambda: "low"),
        ToolSpec(name="high_info", description="", handler=lambda: "high"),
    ])
    agent = CausalAgentLoop(reasoner=Reasoner(), tools=tools)
    result = agent.run("learn", max_steps=1)

    assert result.observations[0].action_name == "high_info"
