from causalrag.probe import ProbeRunConfig, capability_ladder_profiles, run_probe_ladder


def test_capability_ladder_is_cumulative_and_runtime_real():
    profiles = capability_ladder_profiles()
    ids = [row["id"] for row in profiles]
    assert ids == [
        "vanilla",
        "runtime_eig",
        "bayesian_learning",
        "decision_value",
        "temporal_attribution",
        "open_world",
        "full",
    ]

    by_id = {row["id"]: row["capabilities"].to_dict() for row in profiles}
    assert by_id["vanilla"]["causal_selection"] is False
    assert by_id["runtime_eig"]["causal_selection"] is True
    assert by_id["runtime_eig"]["eig"] is True
    assert by_id["runtime_eig"]["bayesian_updates"] is False
    assert by_id["bayesian_learning"]["bayesian_updates"] is True
    assert by_id["decision_value"]["evsi"] is True
    assert by_id["temporal_attribution"]["temporal_attribution"] is True
    assert by_id["temporal_attribution"]["open_world"] is False
    assert by_id["open_world"]["open_world"] is True
    assert by_id["full"]["retrieval"] is True


def test_h2_ladder_shows_information_selection_is_not_learning():
    report = run_probe_ladder(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )
    arms = {row["id"]: row for row in report["arms"]}

    assert arms["vanilla"]["episode"]["metrics"]["success"] is False
    assert arms["runtime_eig"]["episode"]["metrics"]["success"] is False
    assert arms["bayesian_learning"]["episode"]["metrics"]["success"] is True

    eig_metrics = arms["runtime_eig"]["episode"]["metrics"]
    bayes_metrics = arms["bayesian_learning"]["episode"]["metrics"]
    assert bayes_metrics["true_hypothesis_posterior"] > eig_metrics["true_hypothesis_posterior"]
    assert arms["bayesian_learning"]["marginal_delta_from_previous"]["success"] == 1.0
    assert arms["bayesian_learning"]["first_action_difference_from_previous"] is not None


def test_ladder_reports_zero_marginal_layers_when_scenario_does_not_need_them():
    report = run_probe_ladder(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )
    arms = {row["id"]: row for row in report["arms"]}
    temporal_delta = arms["temporal_attribution"]["marginal_delta_from_previous"]
    open_world_delta = arms["open_world"]["marginal_delta_from_previous"]

    assert temporal_delta is not None
    assert temporal_delta["success"] == 0.0
    assert open_world_delta["success"] == 0.0
    assert report["comparison"]["paired_randomness"] == "identical_deterministic_outcomes"


def test_stochastic_ladder_uses_action_indexed_common_random_numbers():
    report = run_probe_ladder(
        ProbeRunConfig(
            hidden_hypothesis="H3",
            outcome_mode="stochastic",
            seed=11,
            proposer_family="deterministic",
        )
    )
    assert report["comparison"]["paired_randomness"] == "action_indexed_common_random_numbers"
