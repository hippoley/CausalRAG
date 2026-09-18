from causalrag.agent import RuntimeCapabilities
from causalrag.probe import (
    InteractiveProbeSession,
    ProbeRunConfig,
    available_probe_config,
    run_probe_episode,
)


def test_probe_config_exposes_research_controls_without_credentials():
    config = available_probe_config()
    assert config["scenarios"][0]["id"] == "hvac_hidden_world"
    assert set(config["proposer_families"]) == {"deterministic", "small", "frontier"}
    assert "eig" in config["capabilities"]
    assert "evsi" in config["capabilities"]
    assert "open_world" in config["capabilities"]


def test_probe_full_runtime_executes_real_hidden_world_episode_and_returns_trace():
    result = run_probe_episode(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=7,
            proposer_family="deterministic",
        )
    )
    assert result["metrics"]["success"] is True
    assert result["metrics"]["selected_hypothesis"] == "H2"
    assert result["trace_id"]
    assert result["causal_trace"]
    assert result["decisions"]
    assert result["observations"]
    assert result["runtime_capabilities"]["causal_selection"] is True


def test_probe_vanilla_arm_changes_execution_not_only_metadata():
    full = run_probe_episode(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )
    vanilla = run_probe_episode(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
            capabilities=RuntimeCapabilities.vanilla_tool_loop().to_dict(),
        )
    )
    assert full["metrics"]["success"] is True
    assert vanilla["runtime_capabilities"]["causal_selection"] is False
    assert full["decisions"] != vanilla["decisions"]
    assert full["metrics"] != vanilla["metrics"]


def test_interactive_session_previews_without_executing_and_records_human_choice():
    session = InteractiveProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )
    try:
        started = session.start()
        preview = started["preview"]
        assert preview["candidates"]
        assert preview["ranking"]
        assert session.environment.probes == 0
        assert session.environment.interventions == 0
        assert session.state.observations == []
        assert started["trace_id"]

        snapshot = session.commit(
            "runtime",
            human_note="I accept the runtime diagnostic ranking.",
        )
        assert snapshot["step"] == 1
        assert session.environment.probes == 1
        assert snapshot["observations"]
        assert snapshot["human_events"][0]["selection"] == "runtime"
        assert any(
            row["name"] == "causalrag.probe.human_choice"
            for row in snapshot["trace"]
        )
    finally:
        session.close()


def test_interactive_session_can_override_runtime_with_another_candidate():
    session = InteractiveProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
        )
    )
    try:
        preview = session.start()["preview"]
        runtime_name = preview["runtime_preference"]
        alternative = next(
            row["name"]
            for row in preview["candidates"]
            if row["name"] != runtime_name
        )

        snapshot = session.commit(
            alternative,
            human_note="Deliberate tester override.",
        )
        human = snapshot["human_events"][0]
        assert human["selected_action"] == alternative
        assert human["overrode_runtime"] is True
        assert snapshot["observations"][0]["action_name"] == alternative
        assert any(
            row["name"] == "causalrag.probe.human_choice"
            for row in snapshot["trace"]
        )
    finally:
        session.close()


def test_probe_config_reports_backend_readiness_without_api_keys():
    config = available_probe_config()
    assert "backend_status" in config
    assert set(config["backend_status"]) == {"openai", "anthropic", "local"}
    assert "api_key" not in str(config["backend_status"]).lower()
