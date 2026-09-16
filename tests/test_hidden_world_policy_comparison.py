from causalrag.benchmarks import (
    CheapestProbePolicy,
    ConservativeEIGPolicy,
    DecisionValuePolicy,
    GreedyEIGPolicy,
    RandomProbePolicy,
    RiskSensitiveDecisionValuePolicy,
    compare_hidden_world_policies,
    run_policy_episode,
    build_hvac_hidden_world,
)


def test_policy_comparison_uses_same_episode_grid_for_every_policy():
    seeds = [0, 1, 2]
    policies = [
        GreedyEIGPolicy,
        ConservativeEIGPolicy,
        CheapestProbePolicy,
        RandomProbePolicy,
    ]

    report, metrics = compare_hidden_world_policies(
        seeds=seeds,
        policy_types=policies,
    )

    assert set(report.reports) == {
        "greedy_eig",
        "conservative_eig",
        "cheapest_probe",
        "random_probe",
    }
    for policy in policies:
        rows = metrics[policy.policy_id]
        assert len(rows) == 9
        assert {(row.hidden_hypothesis, row.seed) for row in rows} == {
            (hidden, seed)
            for hidden in ("H1", "H2", "H3")
            for seed in seeds
        }
        assert report.reports[policy.policy_id].episodes == 9


def test_random_policy_is_seed_reproducible_and_independent_of_environment_rng():
    scenario = build_hvac_hidden_world("H2")

    metrics_a, result_a = run_policy_episode(RandomProbePolicy, scenario, seed=5)
    metrics_b, result_b = run_policy_episode(RandomProbePolicy, scenario, seed=5)

    assert metrics_a.to_dict() == metrics_b.to_dict()
    assert [decision.selected.name for decision in result_a.state.decisions] == [
        decision.selected.name for decision in result_b.state.decisions
    ]


def test_conservative_policy_does_not_intervene_before_minimum_evidence():
    scenario = build_hvac_hidden_world("H1")

    metrics, result = run_policy_episode(
        ConservativeEIGPolicy,
        scenario,
        seed=0,
        max_steps=7,
    )

    intervention_index = next(
        index
        for index, decision in enumerate(result.state.decisions)
        if decision.selected.kind.value == "intervene"
    )
    probes_before_intervention = sum(
        1
        for decision in result.state.decisions[:intervention_index]
        if decision.selected.kind.value == "observe"
    )
    assert probes_before_intervention >= 2
    assert metrics.probes >= 2


def test_risk_sensitive_policy_uses_runtime_decision_value_and_never_samples_less_than_neutral_on_same_episode():
    scenario = build_hvac_hidden_world("H3")

    neutral_metrics, _neutral_result = run_policy_episode(
        DecisionValuePolicy,
        scenario,
        seed=0,
        max_steps=7,
    )
    risk_metrics, risk_result = run_policy_episode(
        RiskSensitiveDecisionValuePolicy,
        scenario,
        seed=0,
        max_steps=7,
    )

    first = risk_result.state.decisions[0]
    selected_score = next(
        score for score in first.action_scores if score.action_name == first.selected.name
    )
    assert first.selected.kind.value == "observe"
    assert selected_score.decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert risk_metrics.probes >= neutral_metrics.probes


def test_risk_sensitive_policy_is_in_default_tournament():
    report, metrics = compare_hidden_world_policies(seeds=[0, 1])

    assert "decision_value" in report.reports
    assert "risk_sensitive_decision_value" in report.reports
    assert len(metrics["risk_sensitive_decision_value"]) == 6


def test_comparison_reports_metric_ranges_without_declaring_a_winner():
    report, _metrics = compare_hidden_world_policies(seeds=[0, 1])

    for policy_report in report.reports.values():
        assert 0.0 <= policy_report.success_rate <= 1.0
        assert 0.0 <= policy_report.identification_accuracy <= 1.0
        assert 0.0 <= policy_report.mean_true_hypothesis_posterior <= 1.0
        assert policy_report.mean_brier_score >= 0.0
        assert policy_report.mean_probes >= 0.0
        assert policy_report.mean_total_cost >= 0.0
        assert policy_report.mean_causal_regret >= 0.0
