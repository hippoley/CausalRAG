from causalrag import CausalWorldModel, ToolSpec
from causalrag.agent.temporal import PendingEffect, TemporalObservationPoint
from causalrag.experiments import (
    ExperimentContract,
    InterventionContract,
    OutcomeLikelihood,
    TemporalDecisionPreferences,
    best_temporal_observation_value,
    temporal_observation_values,
)
from causalrag.tools.base import ToolRegistry


def _experiment(experiment_id, accuracy):
    error = 1.0 - accuracy
    return ExperimentContract(
        experiment_id=experiment_id,
        outcome_key="signal",
        outcomes=[
            OutcomeLikelihood(
                outcome="h1",
                likelihoods={"H1": accuracy, "H2": error},
            ),
            OutcomeLikelihood(
                outcome="h2",
                likelihoods={"H1": error, "H2": accuracy},
            ),
        ],
    )


def _setup():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Mechanism H1 is active.", probability=0.5)
    world.upsert_hypothesis("H2", "Mechanism H2 is active.", probability=0.5)

    tools = ToolRegistry(
        [
            ToolSpec(
                name="fix_h1",
                description="Act for H1.",
                handler=lambda: None,
                intervention_contract=InterventionContract(
                    "fix_h1", {"H1": 1.0, "H2": -1.0}
                ),
            ),
            ToolSpec(
                name="fix_h2",
                description="Act for H2.",
                handler=lambda: None,
                intervention_contract=InterventionContract(
                    "fix_h2", {"H1": -1.0, "H2": 1.0}
                ),
            ),
        ]
    )

    early = _experiment("observe_at_5s", 0.80)
    late = _experiment("observe_at_8s", 0.90)
    effect = PendingEffect(
        effect_id="delayed_signal",
        intervention="stimulate",
        observe_with="read_signal",
        observation_key="signal",
        started_at=0.0,
        ready_at=5.0,
        expires_at=12.0,
        expected_outcomes={"H1": "h1", "H2": "h2"},
        observation_points=(
            TemporalObservationPoint(5.0, early),
            TemporalObservationPoint(8.0, late),
        ),
    )
    return world, tools, effect


def test_low_wait_cost_prefers_more_discriminative_later_observation():
    world, tools, effect = _setup()
    values = temporal_observation_values(
        effect,
        world,
        tools,
        now=0.0,
        preferences=TemporalDecisionPreferences(wait_cost_per_second=0.02),
    )
    by_time = {value.offset_seconds: value for value in values}

    assert by_time[8.0].evsi > by_time[5.0].evsi
    assert by_time[8.0].wait_cost > by_time[5.0].wait_cost
    assert by_time[8.0].net_value > by_time[5.0].net_value

    selected = best_temporal_observation_value(
        effect,
        world,
        tools,
        now=0.0,
        preferences=TemporalDecisionPreferences(wait_cost_per_second=0.02),
    )
    assert selected.offset_seconds == 8.0
    assert selected.experiment_id == "observe_at_8s"


def test_high_wait_cost_prefers_earlier_weaker_observation():
    world, tools, effect = _setup()
    selected = best_temporal_observation_value(
        effect,
        world,
        tools,
        now=0.0,
        preferences=TemporalDecisionPreferences(wait_cost_per_second=0.08),
    )

    assert selected.offset_seconds == 5.0
    assert selected.experiment_id == "observe_at_5s"


def test_past_observation_points_are_not_reconsidered():
    world, tools, effect = _setup()
    selected = best_temporal_observation_value(
        effect,
        world,
        tools,
        now=6.0,
        preferences=TemporalDecisionPreferences(wait_cost_per_second=0.0),
    )

    assert selected.offset_seconds == 8.0
