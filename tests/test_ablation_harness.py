import pytest

from causalrag.benchmarks.ablation import (
    AblationVariant,
    ExperimentProtocol,
    ModelSpec,
    RuntimeFeatures,
    canonical_model_ablation,
    causal_runtime_ablation,
    run_ablation,
)


def test_model_tier_ablation_uses_identical_seeded_protocol_and_aggregates_metrics():
    protocol = ExperimentProtocol(
        benchmark="hidden-world",
        scenario="hvac-faults",
        seeds=[3, 7, 11],
        max_steps=8,
        observation_budget=4,
        intervention_budget=1,
        tool_surface_id="hvac-tools-v1",
        observation_surface_id="hvac-observations-v1",
    )
    variants = canonical_model_ablation(
        deterministic_model="fixed",
        local_model="qwen-local",
        frontier_model="frontier",
    )
    calls = []

    def runner(variant, received_protocol, seed):
        calls.append((variant.name, seed, received_protocol.fingerprint()))
        bonus = {
            "deterministic+causal": 0.0,
            "local-small+causal": 1.0,
            "frontier+causal": 2.0,
        }[variant.name]
        return {
            "metrics": {
                "utility": seed + bonus,
                "wrong_interventions": 1.0 if variant.name == "deterministic+causal" else 0.0,
            }
        }

    report = run_ablation(variants, protocol, runner)

    assert len(report["episodes"]) == 9
    assert {fingerprint for _, _, fingerprint in calls} == {protocol.fingerprint()}
    for seed in protocol.seeds:
        assert [name for name, seen_seed, _ in calls if seen_seed == seed] == [variant.name for variant in variants]

    by_name = {summary["variant"]: summary for summary in report["summaries"]}
    assert by_name["frontier+causal"]["metrics"]["utility"]["mean"] == pytest.approx((5 + 9 + 13) / 3)
    assert by_name["local-small+causal"]["model_tier"] == "local_small"
    assert by_name["deterministic+causal"]["metrics"]["wrong_interventions"]["mean"] == 1.0
    assert by_name["frontier+causal"]["metrics"]["utility"]["ci95_high"] > by_name["frontier+causal"]["metrics"]["utility"]["mean"]


def test_architecture_ablation_holds_model_identity_fixed_and_changes_only_runtime_features():
    model = ModelSpec("frontier", "openai", "same-model")
    variants = causal_runtime_ablation(model)
    assert len(variants) == 6
    assert {variant.model for variant in variants} == {model}
    full = next(variant for variant in variants if variant.name == "causal-full")
    model_only = next(variant for variant in variants if variant.name == "model-only-loop")
    assert full.features == RuntimeFeatures()
    assert model_only.features.causal_runtime is False
    assert model_only.features.eig is False
    assert model_only.features.open_world is False


def test_protocol_rejects_duplicate_seeds_and_runner_rejects_non_numeric_metrics():
    with pytest.raises(ValueError, match="seeds must be unique"):
        ExperimentProtocol("x", "y", [1, 1], 4)

    protocol = ExperimentProtocol("x", "y", [1], 4)
    variants = [
        AblationVariant("a", ModelSpec("deterministic", "builtin", "a")),
        AblationVariant("b", ModelSpec("frontier", "openai", "b")),
    ]
    with pytest.raises(ValueError, match="not numeric"):
        run_ablation(variants, protocol, lambda *_: {"metrics": {"utility": "bad"}})
