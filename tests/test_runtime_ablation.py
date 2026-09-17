import pytest

from causalrag.agent import ActionKind, CandidateAction
from causalrag.agent.features import RuntimeFeatureFlags
from causalrag.experiments import ExperimentContract, InterventionContract, OutcomeLikelihood
from causalrag.reasoning.policy import score_action
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "cause one", probability=0.5)
    world.upsert_hypothesis("H2", "cause two", probability=0.5)
    return world


def _experiment():
    return ExperimentContract(
        experiment_id="diagnostic",
        outcomes=[
            OutcomeLikelihood("a", {"H1": 0.9, "H2": 0.1}),
            OutcomeLikelihood("b", {"H1": 0.1, "H2": 0.9}),
        ],
    )


def test_eig_switch_changes_actual_policy_information_source():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(name="diagnose", description="diagnose", handler=lambda: {"outcome": "a"}, experiment_contract=_experiment())
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
        expected_information_gain=0.01,
    )

    full = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(eig=True))
    no_eig = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(eig=False))

    assert full.information_source == "runtime_bayesian_eig"
    assert full.bayesian_information_gain is not None
    assert no_eig.information_source == "runtime_hypothesis_discrimination_eig_disabled"
    assert no_eig.bayesian_information_gain is None


def test_causal_runtime_off_uses_model_estimates_and_skips_runtime_bayes():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(name="diagnose", description="diagnose", handler=lambda: {"outcome": "a"}, experiment_contract=_experiment())
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
        expected_information_gain=0.23,
        expected_goal_gain=0.12,
    )

    score = score_action(
        action,
        world_model=world,
        tools=tools,
        features=RuntimeFeatureFlags(causal_runtime=False),
    )
    assert score.information_source == "model_estimate_causal_runtime_disabled"
    assert score.information_gain == pytest.approx(0.23)
    assert score.bayesian_information_gain is None
    assert score.decision_value is None


def test_evsi_switch_removes_sampling_decision_value_but_keeps_eig():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(
            name="diagnose",
            description="diagnose",
            handler=lambda: {"outcome": "a"},
            experiment_contract=_experiment(),
            cost=0.02,
        ),
        ToolSpec(
            name="fix_one",
            description="fix one",
            handler=lambda: {"ok": True},
            intervention_contract=InterventionContract(
                intervention_id="fix_one",
                utilities={"H1": 1.0, "H2": -1.0},
            ),
        ),
        ToolSpec(
            name="fix_two",
            description="fix two",
            handler=lambda: {"ok": True},
            intervention_contract=InterventionContract(
                intervention_id="fix_two",
                utilities={"H1": -1.0, "H2": 1.0},
            ),
        ),
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
    )

    with_evsi = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(evsi=True))
    without_evsi = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(evsi=False))

    assert with_evsi.information_source == "runtime_bayesian_eig"
    assert with_evsi.expected_value_of_sample_information is not None
    assert with_evsi.decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert without_evsi.information_source == "runtime_bayesian_eig"
    assert without_evsi.expected_value_of_sample_information is None
    assert without_evsi.decision_value is None
