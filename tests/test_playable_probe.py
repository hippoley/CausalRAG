from causalrag.agent import RuntimeCapabilities
from causalrag.probe import ProbeRunConfig, available_probe_config, run_probe_episode


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
