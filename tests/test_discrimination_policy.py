from causalrag.agent.actions import ActionKind, CandidateAction
from causalrag.agent.loop import CausalAgentLoop
from causalrag.reasoning.policy import (
    hypothesis_discrimination_score,
    rank_actions,
    score_action,
    select_action,
)
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Filter is clogged", probability=0.6)
    world.upsert_hypothesis("H2", "Fan is weak", probability=0.4)
    return world


def test_balanced_multi_hypothesis_test_gets_high_discrimination_score():
    world = _world()
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="read_filter_pressure",
        tests_hypotheses=["H1", "H2"],
    )

    score = hypothesis_discrimination_score(action, world)

    assert score is not None
    assert score > 0.9


def test_runtime_discrimination_overrides_low_model_information_self_score():
    world = _world()
    diagnostic = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnostic_sensor",
        expected_information_gain=0.01,
        tests_hypotheses=["H1", "H2"],
        cost=0.05,
    )
    generic = CandidateAction(
        kind=ActionKind.RETRIEVE,
        name="generic_search",
        expected_information_gain=0.8,
        cost=0.05,
    )

    ranked = rank_actions([generic, diagnostic], world)

    assert ranked[0][0] is diagnostic
    assert ranked[0][1].information_source == "runtime_hypothesis_discrimination"
    assert ranked[0][1].model_information_gain == 0.01
    assert ranked[0][1].information_gain > 0.9


def test_unknown_declared_hypothesis_test_cannot_use_huge_model_self_score():
    world = _world()
    bogus = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="bogus_sensor",
        expected_information_gain=100.0,
        tests_hypotheses=["UNKNOWN"],
    )
    grounded = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="grounded_sensor",
        expected_information_gain=0.0,
        tests_hypotheses=["H1", "H2"],
        cost=0.1,
    )

    bogus_score = score_action(bogus, world)
    selected = select_action([bogus, grounded], world)

    assert bogus_score.model_information_gain == 1.0
    assert bogus_score.information_gain == 0.0
    assert bogus_score.information_source == "runtime_hypothesis_discrimination"
    assert selected is grounded


def test_single_high_credence_falsification_target_has_runtime_value():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Filter is clogged", probability=0.8)
    world.upsert_hypothesis("H2", "Fan is weak", probability=0.2)
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="read_filter_pressure",
        tests_hypotheses=["H1"],
        falsification_target="H1",
    )

    score = hypothesis_discrimination_score(action, world)

    assert score is not None
    assert score > 0.6


def test_model_goal_and_information_scores_are_bounded():
    action = CandidateAction(
        kind=ActionKind.RETRIEVE,
        name="search",
        expected_goal_gain=999.0,
        expected_information_gain=999.0,
    )

    score = score_action(action)

    assert score.goal_gain == 1.0
    assert score.model_information_gain == 1.0
    assert score.information_gain == 1.0
    assert score.total_utility == 2.0


class RuntimePolicyReasoner:
    def propose(self, state, world_model):
        return [
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="generic_sensor",
                expected_information_gain=0.8,
                cost=0.05,
            ),
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="diagnostic_sensor",
                expected_information_gain=0.01,
                tests_hypotheses=["H1", "H2"],
                falsification_target="H1",
                cost=0.05,
            ),
        ]

    def uncertainty(self, state, world_model):
        return "fault source"


def test_agent_loop_records_runtime_score_and_information_source():
    world = _world()
    tools = ToolRegistry(
        [
            ToolSpec(name="generic_sensor", description="generic", handler=lambda: "generic"),
            ToolSpec(name="diagnostic_sensor", description="diagnostic", handler=lambda: "diagnostic"),
        ]
    )
    loop = CausalAgentLoop(
        reasoner=RuntimePolicyReasoner(),
        tools=tools,
        world_model=world,
    )

    state = loop.run("diagnose", max_steps=1)

    assert state.observations[0].action_name == "diagnostic_sensor"
    assert state.decisions[0].action_scores[0].action_name == "diagnostic_sensor"
    assert (
        state.decisions[0].action_scores[0].information_source
        == "runtime_hypothesis_discrimination"
    )
    transition = world.transitions[0]
    assert transition.expected_effects["information_source"] == "runtime_hypothesis_discrimination"
    assert transition.expected_effects["model_information_gain"] == 0.01
    assert transition.expected_effects["information_gain"] > 0.9
