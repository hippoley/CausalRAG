import time

from causalrag.probe.session import ProbeSession
from causalrag.agent import RuntimeCapabilities
from causalrag.probe import ProbeRunConfig, available_probe_config, run_probe_comparison, run_probe_episode, run_probe_ladder


def test_probe_catalog_exposes_three_executable_scenarios():
    config = available_probe_config()
    by_id = {row["id"]: row for row in config["scenarios"]}
    assert set(by_id) == {
        "hvac_hidden_world",
        "temporal_delayed_effect",
        "open_world_mismatch",
    }
    assert by_id["temporal_delayed_effect"]["hidden_hypotheses"] == ["H1", "H2"]
    assert by_id["temporal_delayed_effect"]["outcome_modes"] == ["deterministic"]
    assert by_id["open_world_mismatch"]["hidden_hypotheses"] == ["H4"]
    assert "open-world" in by_id["open_world_mismatch"]["recommended_test"].lower()


def test_temporal_probe_fails_without_temporal_attribution_and_succeeds_with_it():
    full = run_probe_episode(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H1",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    vanilla = run_probe_episode(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H1",
            outcome_mode="deterministic",
            proposer_family="deterministic",
            capabilities=RuntimeCapabilities.vanilla_tool_loop().to_dict(),
        )
    )

    assert vanilla["metrics"]["success"] is False
    assert vanilla["metrics"]["premature_reads"] == 1
    assert vanilla["metrics"]["wait_actions"] == 0

    assert full["metrics"]["success"] is True
    assert full["metrics"]["premature_reads"] == 0
    assert full["metrics"]["wait_actions"] == 1
    assert full["metrics"]["virtual_time_seconds"] == 5.0


def test_temporal_same_world_ab_shows_runtime_guard_changes_outcome():
    report = run_probe_comparison(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    assert report["vanilla"]["metrics"]["success"] is False
    assert report["causal"]["metrics"]["success"] is True
    assert report["metric_deltas_causal_minus_vanilla"]["premature_reads"] == -1.0
    assert report["metric_deltas_causal_minus_vanilla"]["wait_actions"] == 1.0
    assert report["first_divergence"] is not None


def test_open_world_probe_requires_model_mismatch_discovery_to_validate_h4():
    full = run_probe_episode(
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    closed = run_probe_episode(
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="deterministic",
            proposer_family="deterministic",
            capabilities=RuntimeCapabilities(
                causal_selection=True,
                causal_updates=True,
                bayesian_updates=True,
                eig=True,
                evsi=True,
                temporal_attribution=True,
                open_world=False,
                retrieval=False,
            ).to_dict(),
        )
    )

    assert closed["metrics"]["success"] is False
    assert closed["metrics"]["discovered_hypothesis"] is False

    assert full["metrics"]["success"] is True
    assert full["metrics"]["discovered_hypothesis"] is True
    assert full["metrics"]["discovered_validated"] is True
    assert full["metrics"]["true_hypothesis_posterior"] > 0.5


def test_scenario_ladder_attributes_temporal_and_open_world_capabilities():
    temporal = run_probe_ladder(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H1",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    t = {row["id"]: row for row in temporal["arms"]}
    assert t["decision_value"]["episode"]["metrics"]["success"] is False
    assert t["temporal_open_world"]["episode"]["metrics"]["success"] is True
    assert t["temporal_open_world"]["marginal_delta_from_previous"]["premature_reads"] == -1.0

    open_world = run_probe_ladder(
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    o = {row["id"]: row for row in open_world["arms"]}
    assert o["decision_value"]["episode"]["metrics"]["success"] is False
    assert o["temporal_open_world"]["episode"]["metrics"]["success"] is True
    assert o["temporal_open_world"]["marginal_delta_from_previous"]["discovered_hypothesis"] == 1.0


def test_scenario_validation_rejects_impossible_mode_or_hidden_hypothesis():
    try:
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H3",
            outcome_mode="deterministic",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("temporal scenario must reject H3")

    try:
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="stochastic",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("open-world scenario must reject stochastic mode")


def _drive_session(session, timeout=8.0):
    seen = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = session.snapshot()
        if snap["status"] == "waiting_for_human":
            pending = snap["pending_decision"]
            seen.append(pending["runtime_selected"]["name"])
            session.resolve_decision("approve")
        elif snap["status"] == "completed":
            return snap, seen
        elif snap["status"] == "failed":
            raise AssertionError(snap["error"])
        time.sleep(0.01)
    raise AssertionError(f"session did not finish: {session.snapshot()}")


def test_temporal_scenario_hitl_exposes_runtime_inserted_wait_before_execution():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H1",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    final, selected = _drive_session(session)

    assert "wait_for_effect_window" in selected
    assert final["result"]["metrics"]["success"] is True
    assert final["result"]["metrics"]["wait_actions"] == 1
    assert final["result"]["metrics"]["premature_reads"] == 0


def test_open_world_scenario_hitl_discovers_h4_during_live_session():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    final, _selected = _drive_session(session)

    assert final["result"]["metrics"]["success"] is True
    assert any(
        (row.get("id") or row.get("hypothesis_id")) == "H4"
        for row in final["hypotheses"]
    )
