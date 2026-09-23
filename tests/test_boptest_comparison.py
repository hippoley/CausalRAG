import pytest

from branchpoint.benchmarks.boptest_comparison import (
    BOPTESTComparisonError,
    BOPTESTControllerSpec,
    constant_controller,
    run_boptest_comparison,
)
from branchpoint.benchmarks.boptest_protocol import BOPTESTScenarioManifest, no_op_controller


class ScoredFakeClient:
    def __init__(self, service_version="1.0"):
        self.testid = None
        self.service_version = service_version
        self.step_seconds = 300.0
        self.clock = 0.0
        self.control_total = 0.0

    def version(self):
        return {"version": self.service_version}

    def select_testcase(self, _testcase):
        self.testid = "fake"
        return self.testid

    def status(self):
        return {"status": "Running"}

    def name(self):
        return {"name": "Fake"}

    def inputs(self):
        return {"oveHea_u": {"Unit": "1"}}

    def measurements(self):
        return {"TRooAir_y": {"Unit": "K"}}

    def set_step(self, seconds):
        self.step_seconds = float(seconds)
        return self.step_seconds

    def set_scenario(self, **settings):
        return dict(settings)

    def initialize(self, start_time, warmup_period):
        self.clock = float(start_time)
        return {"time": self.clock, "TRooAir_y": 294.0}

    def advance(self, controls=None):
        controls = dict(controls or {})
        self.control_total += float(controls.get("oveHea_u", 0.0))
        self.clock += self.step_seconds
        return {"time": self.clock, "TRooAir_y": 294.0}

    def kpi(self):
        return {
            "ener_tot": 10.0 + self.control_total,
            "tdis_tot": 2.0 - 0.5 * self.control_total,
        }

    def stop(self):
        self.testid = None
        return {"stopped": True}


def _manifest(seed):
    return BOPTESTScenarioManifest(
        testcase="bestest_air",
        step_seconds=300,
        horizon_seconds=600,
        seed=seed,
        controlled_inputs=("oveHea_u",),
        required_kpis=("ener_tot", "tdis_tot"),
    )


def test_paired_comparison_uses_same_manifest_and_reports_raw_deltas():
    report = run_boptest_comparison(
        [_manifest(1), _manifest(2)],
        [
            BOPTESTControllerSpec(
                "embedded",
                no_op_controller,
                metadata={"kind": "reference"},
            ),
            BOPTESTControllerSpec(
                "constant-heat",
                constant_controller({"oveHea_u": 0.25, "oveHea_activate": 1}),
                metadata={"kind": "fixed"},
            ),
        ],
        reference_controller_id="embedded",
        client_factory=ScoredFakeClient,
        sleep=lambda _seconds: None,
    )

    assert len(report.paired_episodes) == 2
    for pair in report.paired_episodes:
        assert set(pair.episodes) == {"embedded", "constant-heat"}
        assert all(
            episode.manifest.manifest_hash == pair.manifest.manifest_hash
            for episode in pair.episodes.values()
        )
        delta = pair.kpi_deltas()["constant-heat"]
        assert delta["ener_tot"] == pytest.approx(0.5)
        assert delta["tdis_tot"] == pytest.approx(-0.25)

    aggregate = report.aggregate_paired_deltas()["constant-heat"]
    assert aggregate["ener_tot"]["count"] == 2
    assert aggregate["ener_tot"]["mean_delta"] == pytest.approx(0.5)
    assert aggregate["tdis_tot"]["mean_delta"] == pytest.approx(-0.25)


def test_duplicate_manifest_hashes_are_rejected_before_running():
    manifest = _manifest(1)
    with pytest.raises(BOPTESTComparisonError, match="unique manifest hashes"):
        run_boptest_comparison(
            [manifest, manifest],
            [BOPTESTControllerSpec("embedded", no_op_controller)],
            reference_controller_id="embedded",
            client_factory=ScoredFakeClient,
        )


def test_duplicate_controller_ids_are_rejected():
    with pytest.raises(BOPTESTComparisonError, match="controller ids must be unique"):
        run_boptest_comparison(
            [_manifest(1)],
            [
                BOPTESTControllerSpec("same", no_op_controller),
                BOPTESTControllerSpec("same", no_op_controller),
            ],
            reference_controller_id="same",
            client_factory=ScoredFakeClient,
        )


def test_reference_controller_must_exist():
    with pytest.raises(BOPTESTComparisonError, match="reference_controller_id"):
        run_boptest_comparison(
            [_manifest(1)],
            [BOPTESTControllerSpec("embedded", no_op_controller)],
            reference_controller_id="missing",
            client_factory=ScoredFakeClient,
        )


def test_service_version_change_invalidates_pair():
    versions = iter(["1.0", "2.0"])

    def client_factory():
        return ScoredFakeClient(next(versions))

    with pytest.raises(BOPTESTComparisonError, match="service version changed"):
        run_boptest_comparison(
            [_manifest(1)],
            [
                BOPTESTControllerSpec("embedded", no_op_controller),
                BOPTESTControllerSpec(
                    "constant-heat",
                    constant_controller({"oveHea_u": 0.25}),
                ),
            ],
            reference_controller_id="embedded",
            client_factory=client_factory,
            sleep=lambda _seconds: None,
        )


def test_constant_controller_returns_fresh_control_mapping():
    controller = constant_controller({"oveHea_u": 0.5})
    first = controller(None)
    first["oveHea_u"] = 0.1
    second = controller(None)
    assert second == {"oveHea_u": 0.5}
