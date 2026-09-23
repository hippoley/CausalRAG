import pytest

from branchpoint.benchmarks.boptest_comparison import (
    BOPTESTComparisonError,
    BOPTESTComparisonReport,
    BOPTESTPairedEpisode,
    bootstrap_paired_kpi_intervals,
    expand_seeded_manifests,
)
from branchpoint.benchmarks.boptest_protocol import (
    BOPTESTEpisodeResult,
    BOPTESTScenarioManifest,
)


def _episode(manifest, controller_id, value):
    return BOPTESTEpisodeResult(
        manifest=manifest,
        controller_id=controller_id,
        service_version={"version": "1.0"},
        testcase_name={"name": "Fake"},
        input_metadata={},
        measurement_metadata={},
        initial_observation={"time": 0.0},
        trajectory=(),
        kpis={"metric": float(value)},
    )


def _report(deltas):
    pairs = []
    for index, delta in enumerate(deltas):
        manifest = BOPTESTScenarioManifest(
            testcase="bestest_air",
            step_seconds=300,
            horizon_seconds=300,
            seed=index,
            required_kpis=("metric",),
        )
        pairs.append(
            BOPTESTPairedEpisode(
                manifest=manifest,
                episodes={
                    "baseline": _episode(manifest, "baseline", 10.0),
                    "candidate": _episode(manifest, "candidate", 10.0 + float(delta)),
                },
                reference_controller_id="baseline",
            )
        )
    return BOPTESTComparisonReport(
        reference_controller_id="baseline",
        paired_episodes=tuple(pairs),
    )


def test_bootstrap_interval_is_reproducible_from_master_seed():
    report = _report([1.0, 2.0, 3.0, 4.0])
    first = bootstrap_paired_kpi_intervals(
        report, confidence=0.90, resamples=1000, seed=17
    )
    second = bootstrap_paired_kpi_intervals(
        report, confidence=0.90, resamples=1000, seed=17
    )

    assert first.to_dict() == second.to_dict()
    interval = first.intervals["candidate"]["metric"]
    assert interval.status == "ok"
    assert interval.count == 4
    assert interval.mean_delta == pytest.approx(2.5)
    assert interval.lower <= interval.mean_delta <= interval.upper
    assert interval.confidence == pytest.approx(0.90)


def test_zero_paired_deltas_have_zero_bootstrap_interval():
    summary = bootstrap_paired_kpi_intervals(
        _report([0.0, 0.0, 0.0]), resamples=500, seed=9
    )
    interval = summary.intervals["candidate"]["metric"]
    assert interval.lower == pytest.approx(0.0)
    assert interval.upper == pytest.approx(0.0)
    assert interval.mean_delta == pytest.approx(0.0)


def test_insufficient_pairs_are_explicit_instead_of_fake_interval():
    summary = bootstrap_paired_kpi_intervals(
        _report([1.0]), resamples=500, seed=1, min_pairs=2
    )
    interval = summary.intervals["candidate"]["metric"]
    assert interval.status == "insufficient_pairs"
    assert interval.count == 1
    assert interval.lower is None
    assert interval.upper is None


def test_expand_seeded_manifests_changes_only_seeded_identity():
    base = BOPTESTScenarioManifest(
        testcase="bestest_air",
        step_seconds=300,
        horizon_seconds=600,
        controlled_inputs=("oveHea_u",),
    )
    manifests = expand_seeded_manifests(base, [11, 12, 13])
    assert [manifest.seed for manifest in manifests] == [11, 12, 13]
    assert len({manifest.manifest_hash for manifest in manifests}) == 3
    assert all(manifest.testcase == base.testcase for manifest in manifests)
    assert all(manifest.horizon_seconds == base.horizon_seconds for manifest in manifests)


def test_duplicate_seed_list_is_rejected():
    base = BOPTESTScenarioManifest(
        testcase="bestest_air", step_seconds=300, horizon_seconds=300
    )
    with pytest.raises(BOPTESTComparisonError, match="seed list must be unique"):
        expand_seeded_manifests(base, [7, 7])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"confidence": 1.0}, "confidence"),
        ({"confidence": 0.0}, "confidence"),
        ({"resamples": 99}, "resamples"),
        ({"min_pairs": 1}, "min_pairs"),
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_paired_kpi_intervals(_report([1.0, 2.0]), **kwargs)
