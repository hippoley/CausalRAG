import pytest

from causalrag.agent import ActionKind, CandidateAction, CausalAgentLoop
from causalrag.experiments import (
    ExperimentContract,
    OutcomeLikelihood,
    contract_applicable,
    expected_information_gain,
    posterior_for_outcome,
)
from causalrag.reasoning.policy import score_action
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Filter is clogged", probability=0.5)
    world.upsert_hypothesis("H2", "Fan is weak", probability=0.5)
    return world


def _contract():
    return ExperimentContract(
        experiment_id="filter_pressure_test",
        outcome_key="outcome",
        description="Pressure drop distinguishes filter restriction from fan weakness.",
        outcomes=[
            OutcomeLikelihood("high", {"H1": 0.9, "H2": 0.1}),
            OutcomeLikelihood("normal", {"H1": 0.1, "H2": 0.9}),
        ],
    )


def test_contract_computes_exact_posterior_and_expected_information_gain():
    world = _world()
    contract = _contract()
    posterior = posterior_for_outcome(contract, world, "normal")
    eig = expected_information_gain(contract, world)
    assert posterior["H1"] == pytest.approx(0.1)
    assert posterior["H2"] == pytest.approx(0.9)
    assert 0.5 < eig < 0.6


def test_contract_is_not_applicable_until_all_modeled_hypotheses_exist():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Filter is clogged", probability=0.7)
    contract = _contract()
    assert contract_applicable(contract, world) is False
    assert expected_information_gain(contract, world) == 0.0
    assert posterior_for_outcome(contract, world, "normal") == {}

    tools = ToolRegistry([
        ToolSpec(
            name="measure_pressure",
            description="diagnostic pressure test",
            handler=lambda: {"outcome": "normal"},
            experiment_contract=contract,
        )
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="measure_pressure",
        expected_information_gain=0.4,
        tests_hypotheses=["H1"],
        falsification_target="H1",
    )
    score = score_action(action, world_model=world, tools=tools)
    assert score.information_source == "runtime_hypothesis_discrimination"
    assert score.bayesian_information_gain is None


def test_policy_prefers_bayesian_eig_over_model_or_heuristic_information_scores():
    world = _world()
    contract = _contract()
    tools = ToolRegistry([
        ToolSpec(
            name="measure_pressure",
            description="diagnostic pressure test",
            handler=lambda: {"outcome": "normal"},
            experiment_contract=contract,
        )
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="measure_pressure",
        expected_information_gain=0.99,
        tests_hypotheses=["H1", "H2"],
    )
    score = score_action(action, world_model=world, tools=tools)
    assert score.information_source == "runtime_bayesian_eig"
    assert score.bayesian_information_gain == pytest.approx(expected_information_gain(contract, world))
    assert score.information_gain == score.bayesian_information_gain


class ContractReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="measure_pressure",
                    tests_hypotheses=["H1", "H2"],
                    expected_information_gain=0.01,
                )
            ]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

    def uncertainty(self, state, world_model):
        return "which fault explains low airflow"


class FailingHypothesisUpdater:
    def __call__(self, *args, **kwargs):
        raise AssertionError("LLM hypothesis updater must not run after a resolved experiment contract")


def test_agent_loop_applies_posterior_and_skips_llm_double_counting():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(
            name="measure_pressure",
            description="diagnostic pressure test",
            handler=lambda: {"outcome": "normal", "pressure_pa": 12},
            experiment_contract=_contract(),
        )
    ])
    loop = CausalAgentLoop(
        reasoner=ContractReasoner(),
        tools=tools,
        world_model=world,
        hypothesis_updater=FailingHypothesisUpdater(),
    )
    state = loop.run("diagnose low airflow", max_steps=3)
    assert state.done
    assert world.get_hypothesis("H1").probability == pytest.approx(0.1)
    assert world.get_hypothesis("H2").probability == pytest.approx(0.9)
    transition = world.transitions[0]
    assert transition.expected_effects["information_source"] == "runtime_bayesian_eig"
    assert transition.expected_effects["experiment_id"] == "filter_pressure_test"
    assert transition.expected_effects["observed_outcome"] == "normal"
    assert transition.expected_effects["posterior"]["H2"] == pytest.approx(0.9)
