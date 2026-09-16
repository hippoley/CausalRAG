import pytest

from causalrag.benchmarks import (
    build_hvac_hidden_world,
    run_hidden_world,
    run_hidden_world_suite,
)


def test_seeded_stochastic_episode_is_reproducible():
    scenario = build_hvac_hidden_world("H2")

    metrics_a, result_a = run_hidden_world(
        scenario,
        outcome_mode="stochastic",
        seed=17,
    )
    metrics_b, result_b = run_hidden_world(
        scenario,
        outcome_mode="stochastic",
        seed=17,
    )

    assert metrics_a.to_dict() == metrics_b.to_dict()
    assert [obs.result for obs in result_a.state.observations] == [
        obs.result for obs in result_b.state.observations
    ]


def test_stochastic_metrics_include_calibration_signal():
    metrics, result = run_hidden_world(
        build_hvac_hidden_world("H1"),
        outcome_mode="stochastic",
        seed=3,
    )

    probabilities = [hypothesis.probability for hypothesis in result.world_model.hypotheses()]
    expected_brier = sum(
        (probability - (1.0 if index == 0 else 0.0)) ** 2
        for index, probability in enumerate(probabilities)
    )

    assert metrics.outcome_mode == "stochastic"
    assert metrics.seed == 3
    assert metrics.brier_score == pytest.approx(expected_brier)
    assert 0.0 <= metrics.true_hypothesis_posterior <= 1.0
    assert metrics.brier_score >= 0.0


def test_hidden_world_suite_aggregates_seeded_episodes_without_hiding_failures():
    seeds = [0, 1, 2, 3]
    report, episodes = run_hidden_world_suite(seeds=seeds)

    assert report.episodes == 12
    assert len(episodes) == 12
    assert {row.hidden_hypothesis for row in episodes} == {"H1", "H2", "H3"}
    assert all(row.outcome_mode == "stochastic" for row in episodes)
    assert {row.seed for row in episodes} == set(seeds)

    assert 0.0 <= report.success_rate <= 1.0
    assert 0.0 <= report.identification_accuracy <= 1.0
    assert 0.0 <= report.mean_true_hypothesis_posterior <= 1.0
    assert report.mean_brier_score >= 0.0
    assert report.mean_probes >= 0.0
    assert report.mean_total_cost >= 0.0
    assert report.mean_causal_regret >= 0.0

    for hidden in ("H1", "H2", "H3"):
        row = report.per_hidden[hidden]
        assert row["episodes"] == 4.0
        assert 0.0 <= row["success_rate"] <= 1.0
        assert 0.0 <= row["identification_accuracy"] <= 1.0
        assert row["mean_brier_score"] >= 0.0
