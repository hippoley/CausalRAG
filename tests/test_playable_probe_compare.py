from causalrag.probe import ProbeRunConfig, run_probe_comparison


def test_same_world_comparison_is_real_and_isolates_causal_control_plane():
    report = run_probe_comparison(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )

    meta = report["comparison"]
    assert meta["same_world_inputs"] is True
    assert meta["same_proposer_configuration"] is True
    assert meta["paired_randomness"] == "identical_deterministic_outcomes"

    vanilla = report["vanilla"]
    causal = report["causal"]
    assert vanilla["config"]["hidden_hypothesis"] == causal["config"]["hidden_hypothesis"] == "H2"
    assert vanilla["config"]["seed"] == causal["config"]["seed"] == 0
    assert vanilla["config"]["proposer_family"] == causal["config"]["proposer_family"] == "deterministic"

    assert vanilla["runtime_capabilities"]["causal_selection"] is False
    assert causal["runtime_capabilities"]["causal_selection"] is True
    assert vanilla["metrics"]["success"] is False
    assert causal["metrics"]["success"] is True
    assert report["success_delta"] == 1
    assert report["metric_deltas_causal_minus_vanilla"]["causal_regret"] < 0
    assert report["first_divergence"] is not None


def test_stochastic_comparison_discloses_pairing_limitation():
    report = run_probe_comparison(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="stochastic",
            seed=3,
            proposer_family="deterministic",
        )
    )
    assert report["comparison"]["paired_randomness"] == "same_seed_action_sequence_dependent"
