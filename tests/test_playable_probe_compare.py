from causalrag.probe import ProbeRunConfig, run_probe_comparison
from causalrag.probe.compare import MemoizedLLM, SharedPromptMemo
from causalrag.benchmarks.hidden_world import HiddenWorldEnvironment, build_hvac_hidden_world


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
    assert report["comparison"]["paired_randomness"] == "action_indexed_common_random_numbers"


def test_action_indexed_noise_pairs_same_experiment_across_different_action_orders():
    scenario = build_hvac_hidden_world("H2")
    a = HiddenWorldEnvironment(
        scenario,
        outcome_mode="stochastic",
        seed=17,
        outcome_coupling="action_indexed",
    )
    b = HiddenWorldEnvironment(
        scenario,
        outcome_mode="stochastic",
        seed=17,
        outcome_coupling="action_indexed",
    )

    fan = scenario.experiments["measure_fan_rpm"]
    filt = scenario.experiments["measure_filter_pressure"]

    a_fan = a.experiment_tool("measure_fan_rpm", fan).handler()
    a_filter = a.experiment_tool("measure_filter_pressure", filt).handler()

    b_filter = b.experiment_tool("measure_filter_pressure", filt).handler()
    b_fan = b.experiment_tool("measure_fan_rpm", fan).handler()

    assert a_fan[fan.outcome_key] == b_fan[fan.outcome_key]
    assert a_filter[filt.outcome_key] == b_filter[filt.outcome_key]


class _FakeLLM:
    def __init__(self, label):
        self.label = label
        self.model = "fake-model"
        self.provider = "fake"
        self.telemetry = None
        self.last_usage = {}
        self.calls = []

    def generate(self, prompt, temperature=0.3, max_tokens=800, stream=False, json_mode=False):
        self.calls.append((prompt, temperature, max_tokens, stream, json_mode))
        return f"{self.label}:{prompt}"


def test_identical_external_model_prompts_are_replayed_across_arms():
    memo = SharedPromptMemo()
    left_inner = _FakeLLM("left")
    right_inner = _FakeLLM("right")
    left = MemoizedLLM(left_inner, memo, "vanilla")
    right = MemoizedLLM(right_inner, memo, "causal")

    first = left.generate("same prompt", temperature=0.1, max_tokens=100, json_mode=True)
    second = right.generate("same prompt", temperature=0.1, max_tokens=100, json_mode=True)

    assert first == second == "left:same prompt"
    assert len(left_inner.calls) == 1
    assert len(right_inner.calls) == 0
    assert memo.provider_calls == 1
    assert memo.replays == 1


def test_external_model_prompt_cache_stops_sharing_after_state_prompt_diverges():
    memo = SharedPromptMemo()
    left_inner = _FakeLLM("left")
    right_inner = _FakeLLM("right")
    left = MemoizedLLM(left_inner, memo, "vanilla")
    right = MemoizedLLM(right_inner, memo, "causal")

    left.generate("state A", temperature=0.1, max_tokens=100, json_mode=True)
    right.generate("state B", temperature=0.1, max_tokens=100, json_mode=True)

    assert len(left_inner.calls) == 1
    assert len(right_inner.calls) == 1
    assert memo.provider_calls == 2
    assert memo.replays == 0
