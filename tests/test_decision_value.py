import pytest

from branchpoint.agent import ActionKind, CandidateAction
from branchpoint.benchmarks import (
    DecisionValuePolicy,
    build_hvac_hidden_world,
    run_policy_episode,
)
from branchpoint.experiments import (
    ExperimentContract,
    InterventionContract,
    OutcomeLikelihood,
    best_intervention_value,
    experiment_decision_value,
)
from branchpoint.reasoning.policy import rank_actions
from branchpoint.tools import ToolRegistry, ToolSpec
from branchpoint.world_model import CausalWorldModel


def _tools():
    experiment = ExperimentContract(
        experiment_id="diagnostic",
        outcomes=[
            OutcomeLikelihood("leans_h1", {"H1": 0.9, "H2": 0.1}),
            OutcomeLikelihood("leans_h2", {"H1": 0.1, "H2": 0.9}),
        ],
    )
    return ToolRegistry(
        [
            ToolSpec(
                name="diagnose",
                description="diagnostic experiment",
                handler=lambda: {"outcome": "leans_h1"},
                cost=0.05,
                experiment_contract=experiment,
            ),
            ToolSpec(
                name="fix_h1",
                description="intervene for H1",
                handler=lambda: {"ok": True},
                cost=0.20,
                intervention_contract=InterventionContract(
                    "fix_h1",
                    {"H1": 1.0, "H2": 0.0},
                ),
            ),
            ToolSpec(
                name="fix_h2",
                description="intervene for H2",
                handler=lambda: {"ok": True},
                cost=0.20,
                intervention_contract=InterventionContract(
                    "fix_h2",
                    {"H1": 0.0, "H2": 1.0},
                ),
            ),
        ]
    )


def _world(h1=0.5, h2=0.5):
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "mechanism one", probability=h1)
    world.upsert_hypothesis("H2", "mechanism two", probability=h2)
    return world


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


def test_evsi_prefers_diagnostic_experiment_when_decision_is_uncertain():
    world = _world()
    tools = _tools()

    best_now = best_intervention_value(world, tools)
    experiment = experiment_decision_value(
        "diagnose",
        tools.get("diagnose").experiment_contract,
        world,
        tools,
        experiment_cost=tools.get("diagnose").cost,
    )
    ranked = rank_actions(_candidates(), world_model=world, tools=tools)

    assert best_now is not None
    assert best_now.net_value == pytest.approx(0.3)
    assert experiment is not None
    assert experiment.evsi == pytest.approx(0.4)
    assert experiment.net_value_of_sampling == pytest.approx(0.35)
    assert ranked[0][0].name == "diagnose"
    assert ranked[0][1].decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert ranked[0][1].expected_value_of_sample_information == pytest.approx(0.4)


def test_decision_value_prefers_intervention_when_additional_sample_is_not_worth_cost():
    world = _world(h1=0.95, h2=0.05)
    tools = _tools()

    ranked = rank_actions(_candidates(), world_model=world, tools=tools)

    assert ranked[0][0].name == "fix_h1"
    assert ranked[0][1].decision_value_source == "runtime_expected_intervention_utility"
    diagnose_score = next(score for action, score in ranked if action.name == "diagnose")
    assert diagnose_score.net_value_of_sampling is not None
    assert diagnose_score.net_value_of_sampling < 0.0


def test_decision_value_ignores_model_self_scores_when_contracts_are_available():
    world = _world()
    tools = _tools()
    candidates = _candidates()
    candidates[0].expected_information_gain = 0.0
    candidates[1].expected_goal_gain = 999.0

    ranked = rank_actions(candidates, world_model=world, tools=tools)

    assert ranked[0][0].name == "diagnose"
    assert ranked[0][1].decision_value is not None


def test_hiddenworld_decision_value_policy_is_runtime_arbitrated_not_thresholded():
    metrics, result = run_policy_episode(
        DecisionValuePolicy,
        build_hvac_hidden_world("H2"),
        seed=0,
        max_steps=7,
    )

    first = result.state.decisions[0]
    selected_score = next(
        score for score in first.action_scores if score.action_name == first.selected.name
    )
    assert first.selected.kind == ActionKind.OBSERVE
    assert selected_score.decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert selected_score.expected_value_of_sample_information is not None
    assert any(
        decision.selected.kind == ActionKind.INTERVENE
        for decision in result.state.decisions
    )
    assert metrics.interventions == 1
