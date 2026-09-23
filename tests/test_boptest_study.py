import pytest

from branchpoint.benchmarks import (
    BOPTESTArbitrationStudyPlan,
    BOPTESTScenarioManifest,
    BOPTESTStudyPeriod,
    BOPTESTStudyPlanError,
    TemperatureBandProposalConfig,
    run_boptest_arbitration_study,
)


class StudyFakeClient:
    def __init__(self):
        self.testid = None
        self.step_seconds = 900.0
        self.clock = 0.0
        self.override_steps = 0

    def version(self):
        return {"version": "study-fake-1"}

    def select_testcase(self, testcase):
        assert testcase == "bestest_air"
        self.testid = "fake"
        return self.testid

    def status(self):
        return {"status": "Running"}

    def name(self):
        return {"name": "BESTEST Air Fake"}

    def inputs(self):
        return {
            "con_oveTSetHea_u": {"Unit": "K"},
            "con_oveTSetCoo_u": {"Unit": "K"},
        }

    def measurements(self):
        return {
            "zon_reaTRooAir_y": {"Unit": "K"},
        }

    def set_step(self, seconds):
        self.step_seconds = float(seconds)
        return self.step_seconds

    def set_scenario(self, **settings):
        return dict(settings)

    def initialize(self, start_time, warmup_period):
        self.clock = float(start_time)
        # Deliberately outside the configured band so both arms receive the
        # exact same two-candidate proposal at the first step.
        return {
            "time": self.clock,
            "zon_reaTRooAir_y": 300.0,
        }

    def advance(self, controls=None):
        controls = dict(controls or {})
        if controls.get("con_oveTSetCoo_activate") == 1:
            self.override_steps += 1
        self.clock += self.step_seconds
        return {
            "time": self.clock,
            "zon_reaTRooAir_y": 300.0,
        }

    def kpi(self):
        return {
            "ener_tot": 10.0 + 0.01 * self.override_steps,
            "cost_tot": 2.0 + 0.005 * self.override_steps,
            "tdis_tot": 5.0 - 0.02 * self.override_steps,
            "idis_tot": 0.0,
            "time_rat": 1.0,
        }

    def stop(self):
        self.testid = None
        return {"stopped": True}


def _base_manifest(**overrides):
    values = {
        "testcase": "bestest_air",
        "start_time": 0,
        "warmup_period": 604800,
        "step_seconds": 900,
        "horizon_seconds": 1800,
        "electricity_price": "constant",
        "controlled_inputs": (
            "con_oveTSetHea_u",
            "con_oveTSetCoo_u",
        ),
        "measurement_points": ("zon_reaTRooAir_y",),
        "required_kpis": (
            "ener_tot",
            "cost_tot",
            "tdis_tot",
            "idis_tot",
            "time_rat",
        ),
    }
    values.update(overrides)
    return BOPTESTScenarioManifest(**values)


def _plan(**overrides):
    values = {
        "base_manifest": _base_manifest(),
        "periods": (
            BOPTESTStudyPeriod("peak_heat_center", 341),
            BOPTESTStudyPeriod("peak_cool_center", 289),
            BOPTESTStudyPeriod("mix_center", 21),
        ),
        "proposal_config": TemperatureBandProposalConfig(
            lower_kelvin=294.15,
            upper_kelvin=297.15,
            intervention_goal_gain=0.70,
            embedded_goal_gain=0.20,
            intervention_risk=0.90,
            intervention_cost=0.02,
        ),
        "reference_controller_id": "proposal-order",
        "confidence": 0.95,
        "resamples": 500,
        "bootstrap_seed": 20260923,
    }
    values.update(overrides)
    return BOPTESTArbitrationStudyPlan(**values)


def test_study_plan_round_trip_preserves_hash_and_period_identity():
    original = _plan()
    restored = BOPTESTArbitrationStudyPlan.from_dict(original.to_dict())

    assert restored == original
    assert restored.study_hash == original.study_hash
    assert [period.day_index for period in restored.periods] == [341, 289, 21]
    assert [period.start_time for period in restored.periods] == [
        341 * 86400.0,
        289 * 86400.0,
        21 * 86400.0,
    ]
    assert len({m.manifest_hash for m in restored.manifests()}) == 3


def test_non_forecast_study_rejects_forecast_uncertainty_seed_design():
    with pytest.raises(BOPTESTStudyPlanError, match="forecast uncertainty"):
        _plan(
            base_manifest=_base_manifest(
                temperature_uncertainty="medium",
            )
        )

    with pytest.raises(BOPTESTStudyPlanError, match="seed must be null"):
        _plan(
            base_manifest=_base_manifest(seed=7)
        )


def test_study_requires_proposer_measurement_and_control_surface():
    with pytest.raises(BOPTESTStudyPlanError, match="temperature measurement"):
        _plan(
            base_manifest=_base_manifest(measurement_points=()),
        )

    with pytest.raises(BOPTESTStudyPlanError, match="missing proposal control"):
        _plan(
            base_manifest=_base_manifest(
                controlled_inputs=("con_oveTSetHea_u",),
            )
        )


def test_study_runner_binds_plan_periods_comparison_and_uncertainty():
    plan = _plan()
    result = run_boptest_arbitration_study(
        plan,
        client_factory=StudyFakeClient,
    )
    payload = result.to_dict()

    assert payload["study_hash"] == plan.study_hash
    assert set(payload["period_manifest_hashes"]) == {
        "peak_heat_center",
        "peak_cool_center",
        "mix_center",
    }
    assert len(payload["comparison"]["paired_episodes"]) == 3

    aggregate = payload["comparison"]["aggregate_paired_kpi_deltas"]["branchpoint"]
    # Proposal-order executes the high-risk override while Branchpoint rejects
    # it in the fake world, so paired deltas are non-zero and deterministic.
    assert aggregate["ener_tot"]["mean_delta"] < 0
    assert aggregate["tdis_tot"]["mean_delta"] > 0

    interval = payload["uncertainty"]["intervals"]["branchpoint"]["ener_tot"]
    assert interval["status"] == "ok"
    assert interval["count"] == 3
    assert payload["uncertainty"]["interpretation"].startswith(
        "descriptive robustness interval"
    )


def test_period_validation_rejects_duplicate_or_out_of_range_days():
    with pytest.raises(BOPTESTStudyPlanError, match="day_index"):
        BOPTESTStudyPeriod("bad", 366)

    with pytest.raises(BOPTESTStudyPlanError, match="period ids"):
        _plan(
            periods=(
                BOPTESTStudyPeriod("same", 10),
                BOPTESTStudyPeriod("same", 20),
            )
        )

    with pytest.raises(BOPTESTStudyPlanError, match="start times"):
        _plan(
            periods=(
                BOPTESTStudyPeriod("one", 10),
                BOPTESTStudyPeriod("two", 10),
            )
        )
