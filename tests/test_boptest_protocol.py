import pytest

from branchpoint.benchmarks.boptest_protocol import (
    BOPTESTControlDecision,
    BOPTESTControlError,
    BOPTESTExperimentError,
    BOPTESTManifestError,
    BOPTESTScenarioManifest,
    run_boptest_episode,
)


class FakeBOPTESTClient:
    def __init__(self):
        self.testid = None
        self.calls = []
        self.step_seconds = None
        self.clock = 0.0
        self.statuses = [{"status": "Queued"}, {"status": "Running"}]

    def version(self):
        self.calls.append(("version",))
        return {"version": "test"}

    def select_testcase(self, testcase):
        self.calls.append(("select", testcase))
        self.testid = "test-1"
        return self.testid

    def status(self):
        self.calls.append(("status",))
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def name(self):
        self.calls.append(("name",))
        return {"name": "Fake BESTEST"}

    def inputs(self):
        self.calls.append(("inputs",))
        return {"oveHea_u": {"Unit": "1"}, "oveCoo_u": {"Unit": "1"}}

    def measurements(self):
        self.calls.append(("measurements",))
        return {"TRooAir_y": {"Unit": "K"}, "PHea_y": {"Unit": "W"}}

    def set_step(self, seconds):
        self.calls.append(("set_step", float(seconds)))
        self.step_seconds = float(seconds)
        return self.step_seconds

    def set_scenario(self, **settings):
        self.calls.append(("scenario", dict(settings)))
        return dict(settings)

    def initialize(self, start_time, warmup_period):
        self.calls.append(("initialize", float(start_time), float(warmup_period)))
        self.clock = float(start_time)
        return {"time": self.clock, "TRooAir_y": 294.0, "PHea_y": 100.0}

    def advance(self, controls=None):
        controls = dict(controls or {})
        self.calls.append(("advance", controls))
        self.clock += float(self.step_seconds)
        return {
            "time": self.clock,
            "TRooAir_y": 294.0 + self.clock / 10000.0,
            "PHea_y": float(controls.get("oveHea_u", 0.0)) * 1000.0,
        }

    def kpi(self):
        self.calls.append(("kpi",))
        return {"ener_tot": 1.25, "tdis_tot": 0.4, "cost_tot": 0.3}

    def stop(self):
        self.calls.append(("stop", self.testid))
        self.testid = None
        return {"stopped": True}


def _manifest(**overrides):
    values = {
        "testcase": "bestest_air",
        "start_time": 0,
        "warmup_period": 86400,
        "step_seconds": 300,
        "horizon_seconds": 900,
        "seed": 7,
        "controlled_inputs": ("oveHea_u",),
        "measurement_points": ("TRooAir_y",),
        "required_kpis": ("ener_tot", "tdis_tot"),
    }
    values.update(overrides)
    return BOPTESTScenarioManifest(**values)


def test_manifest_hash_is_stable_and_protocol_is_explicit():
    first = _manifest()
    second = _manifest()
    assert first.manifest_hash == second.manifest_hash
    assert first.steps == 3
    assert first.to_dict()["scenario"] == {"seed": 7}
    assert first.to_dict()["protocol_version"] == "branchpoint.boptest.v1"


def test_manifest_round_trip_preserves_experiment_identity():
    original = _manifest(
        electricity_price="dynamic",
        temperature_uncertainty="medium",
    )
    restored = BOPTESTScenarioManifest.from_dict(original.to_dict())
    assert restored == original
    assert restored.manifest_hash == original.manifest_hash


def test_manifest_rejects_fractional_control_horizon():
    with pytest.raises(BOPTESTManifestError, match="integer multiple"):
        _manifest(horizon_seconds=1000)


def test_episode_freezes_surface_and_records_exact_trajectory():
    client = FakeBOPTESTClient()
    seen = []

    def controller(context):
        seen.append((context.step_index, context.elapsed_seconds, context.observation["TRooAir_y"]))
        return {"oveHea_u": 0.25, "oveHea_activate": 1}

    result = run_boptest_episode(
        _manifest(),
        controller,
        controller_id="constant-heat-025",
        client=client,
        sleep=lambda _seconds: None,
    )

    assert result.controller_id == "constant-heat-025"
    assert len(result.trajectory) == 3
    assert [row.elapsed_seconds for row in result.trajectory] == [300.0, 600.0, 900.0]
    assert all(set(row.observation) == {"time", "TRooAir_y"} for row in result.trajectory)
    assert result.kpis["ener_tot"] == pytest.approx(1.25)
    assert result.scenario_state == {"seed": 7}
    assert seen[0][:2] == (0, 0.0)
    assert client.testid is None
    assert client.calls[-1][0] == "stop"


def test_controller_cannot_escape_manifest_control_surface():
    client = FakeBOPTESTClient()

    def controller(_context):
        return {"oveCoo_u": 0.5}

    with pytest.raises(BOPTESTControlError, match="escaped"):
        run_boptest_episode(
            _manifest(),
            controller,
            client=client,
            sleep=lambda _seconds: None,
        )
    assert client.testid is None
    assert client.calls[-1][0] == "stop"


def test_unknown_manifest_actuator_fails_before_first_advance():
    client = FakeBOPTESTClient()
    with pytest.raises(BOPTESTManifestError, match="not exposed"):
        run_boptest_episode(
            _manifest(controlled_inputs=("not_real_u",)),
            client=client,
            sleep=lambda _seconds: None,
        )
    assert not any(call[0] == "advance" for call in client.calls)
    assert client.calls[-1][0] == "stop"


def test_missing_required_kpi_invalidates_episode():
    client = FakeBOPTESTClient()
    with pytest.raises(BOPTESTExperimentError, match="required KPI"):
        run_boptest_episode(
            _manifest(required_kpis=("not_returned",)),
            client=client,
            sleep=lambda _seconds: None,
        )
    assert client.calls[-1][0] == "stop"


def test_controller_failure_still_releases_boptest_worker():
    client = FakeBOPTESTClient()

    def broken(_context):
        raise RuntimeError("controller exploded")

    with pytest.raises(RuntimeError, match="controller exploded"):
        run_boptest_episode(
            _manifest(),
            broken,
            client=client,
            sleep=lambda _seconds: None,
        )
    assert client.testid is None
    assert client.calls[-1][0] == "stop"


def test_structured_controller_decision_is_recorded_in_episode_artifact():
    client = FakeBOPTESTClient()

    def controller(context):
        return BOPTESTControlDecision(
            controls={"oveHea_u": 0.25, "oveHea_activate": 1},
            metadata={
                "mode": "branchpoint",
                "step_index": context.step_index,
                "selected": "heat-quarter",
            },
        )

    result = run_boptest_episode(
        _manifest(),
        controller,
        controller_id="branchpoint",
        client=client,
        sleep=lambda _seconds: None,
    )

    assert len(result.trajectory) == 3
    assert result.trajectory[0].controller_metadata == {
        "mode": "branchpoint",
        "step_index": 0,
        "selected": "heat-quarter",
    }
    assert result.to_dict()["trajectory"][0]["controller_metadata"]["selected"] == "heat-quarter"


def test_structured_controller_metadata_must_be_json_serializable():
    with pytest.raises(BOPTESTControlError, match="JSON-serializable"):
        BOPTESTControlDecision(
            controls={},
            metadata={"bad": object()},
        )
