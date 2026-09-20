import pytest

from branchpoint.agent import ActionKind, CandidateAction
from branchpoint.benchmarks import build_hvac_hidden_world, run_hidden_world
from branchpoint.reasoning.policy import score_action
from branchpoint.tools import ToolRegistry, ToolSpec


@pytest.mark.parametrize("hidden", ["H1", "H2", "H3"])
def test_hidden_world_identifies_mechanism_and_intervenes_successfully(hidden):
    scenario = build_hvac_hidden_world(hidden_hypothesis=hidden)

    metrics, result = run_hidden_world(scenario)

    assert metrics.success is True
    assert metrics.identification_correct is True
    assert metrics.selected_hypothesis == hidden
    assert metrics.true_hypothesis_posterior >= 0.8
    assert metrics.probes >= 1
    assert metrics.interventions == 1
    assert metrics.total_cost > 0
    assert metrics.causal_regret >= 0
    assert result.answer == "HiddenWorld intervention succeeded."


def test_hidden_world_trace_uses_runtime_bayesian_eig():
    metrics, result = run_hidden_world(build_hvac_hidden_world("H2"))

    first = result.state.decisions[0]
    selected_score = next(
        score for score in first.action_scores if score.action_name == first.selected.name
    )

    assert selected_score.information_source == "runtime_bayesian_eig"
    assert selected_score.bayesian_information_gain is not None
    assert selected_score.bayesian_information_gain > 0
    assert first.selected.name in {
        "measure_filter_pressure",
        "measure_fan_rpm",
        "measure_duct_pressure",
    }
    assert metrics.success


def test_hidden_world_regret_charges_probe_cost_against_oracle():
    scenario = build_hvac_hidden_world("H2")

    metrics, _result = run_hidden_world(scenario)

    # Oracle knows H2 and repairs the fan immediately for 0.25 cost.
    # The benchmark agent must pay at least one diagnostic probe before repair,
    # so its utility is lower by at least that probe cost.
    assert metrics.causal_regret >= min(scenario.experiment_costs.values())


def test_runtime_policy_enforces_tool_cost_for_custom_reasoners():
    tools = ToolRegistry(
        [
            ToolSpec(
                name="expensive_probe",
                description="real capability cost is runtime-owned",
                handler=lambda: {"ok": True},
                cost=0.7,
                risk=0.2,
                reversible=False,
            )
        ]
    )
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="expensive_probe",
        expected_information_gain=0.9,
        cost=0.0,
        risk=0.0,
        irreversibility=0.0,
    )

    score = score_action(action, tools=tools)

    assert score.cost == pytest.approx(0.7)
    assert score.risk == pytest.approx(0.2)
    assert score.irreversibility == pytest.approx(1.0)
    assert score.total_utility == pytest.approx(0.9 - 0.7 - 0.2 - 1.0)
