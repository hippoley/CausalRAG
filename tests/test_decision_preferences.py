import pytest

from causalrag import (
    ActionKind,
    CandidateAction,
    DecisionPreferences,
    ToolSpec,
    create_agent,
)
from causalrag.experiments import (
    ExperimentContract,
    InterventionContract,
    OutcomeLikelihood,
)
from causalrag.reasoning.policy import rank_actions
from causalrag.tools import ToolRegistry
from causalrag.world_model import CausalWorldModel


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "mechanism one", probability=0.70)
    world.upsert_hypothesis("H2", "mechanism two", probability=0.30)
    return world


def _tools():
    diagnostic = ExperimentContract(
        experiment_id="diagnostic",
        outcomes=[
            OutcomeLikelihood("leans_h1", {"H1": 0.75, "H2": 0.25}),
            OutcomeLikelihood("leans_h2", {"H1": 0.25, "H2": 0.75}),
        ],
    )
    return [
        ToolSpec(
            name="diagnose",
            description="diagnostic observation",
            handler=lambda: {"outcome": "leans_h1"},
            cost=0.08,
            experiment_contract=diagnostic,
        ),
        ToolSpec(
            name="fix_h1",
            description="intervene for H1",
            handler=lambda: {"ok": True},
            cost=0.20,
            intervention_contract=InterventionContract(
                "fix_h1", {"H1": 1.0, "H2": 0.0}
            ),
        ),
        ToolSpec(
            name="fix_h2",
            description="intervene for H2",
            handler=lambda: {"ok": True},
            cost=0.20,
            intervention_contract=InterventionContract(
                "fix_h2", {"H1": 0.0, "H2": 1.0}
            ),
        ),
    ]


def _candidates():
    return [
        CandidateAction(
            kind=ActionKind.OBSERVE,
            name="diagnose",
            tests_hypotheses=["H1", "H2"],
        ),
        CandidateAction(kind=ActionKind.INTERVENE, name="fix_h1"),
        CandidateAction(kind=ActionKind.INTERVENE, name="fix_h2"),
    ]


def test_preferences_change_action_ranking_without_changing_reasoner_or_tools():
    world = _world()
    raw_tools = _tools()
    neutral = ToolRegistry(raw_tools)
    neutral_ranked = rank_actions(_candidates(), world_model=world, tools=neutral)

    preferences = DecisionPreferences(
        intervention_utilities={
            "fix_h1": {"H1": 1.0, "H2": -1.0},
            "fix_h2": {"H1": -1.0, "H2": 1.0},
        }
    )
    preferred = ToolRegistry(raw_tools, decision_preferences=preferences)
    preferred_ranked = rank_actions(_candidates(), world_model=world, tools=preferred)

    assert neutral_ranked[0][0].name == "fix_h1"
    assert preferred_ranked[0][0].name == "diagnose"
    assert preferred_ranked[0][1].decision_value_source == (
        "runtime_expected_decision_value_after_sampling"
    )


def test_preferences_override_contracts_but_leave_original_tools_unchanged():
    raw_tools = _tools()
    original = raw_tools[1].intervention_contract
    preferences = DecisionPreferences(
        intervention_utilities={"fix_h1": {"H1": 2.0, "H2": -3.0}}
    )

    registry = ToolRegistry(raw_tools, decision_preferences=preferences)

    assert raw_tools[1].intervention_contract is original
    assert original.utility("H2") == pytest.approx(0.0)
    assert registry.get("fix_h1").intervention_contract.utility("H1") == pytest.approx(2.0)
    assert registry.get("fix_h1").intervention_contract.utility("H2") == pytest.approx(-3.0)


def test_create_agent_applies_preferences_at_runtime_boundary():
    class StaticReasoner:
        def propose(self, state, world_model):
            return _candidates()

        def uncertainty(self, state, world_model):
            return "H1 vs H2"

    preferences = DecisionPreferences(
        intervention_utilities={
            "fix_h1": {"H1": 1.0, "H2": -1.0},
            "fix_h2": {"H1": -1.0, "H2": 1.0},
        }
    )
    agent = create_agent(
        world_model=_world(),
        reasoner=StaticReasoner(),
        tools=_tools(),
        decision_preferences=preferences,
    )

    assert agent.tools.decision_preferences is preferences
    assert agent.tools.get("fix_h1").intervention_contract.utility("H2") == pytest.approx(-1.0)


def test_preferences_validate_intervention_domains():
    with pytest.raises(ValueError):
        DecisionPreferences(intervention_utilities={"fix_h1": {"H1": 1.0}})
