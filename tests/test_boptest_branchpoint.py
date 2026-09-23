import pytest

from branchpoint.benchmarks.boptest_branchpoint import (
    BOPTESTDecisionAdapterError,
    BranchpointBOPTESTController,
    ProposalOrderBOPTESTController,
    TemperatureBandProposalConfig,
    temperature_band_proposal_provider,
)
from branchpoint.benchmarks.boptest_protocol import (
    BOPTESTControlContext,
    BOPTESTScenarioManifest,
)


def _context(temperature):
    manifest = BOPTESTScenarioManifest(
        testcase="bestest_air",
        step_seconds=300,
        horizon_seconds=300,
        controlled_inputs=("con_oveTSetHea_u", "con_oveTSetCoo_u"),
        measurement_points=("zon_reaTRooAir_y",),
    )
    return BOPTESTControlContext(
        step_index=0,
        elapsed_seconds=0.0,
        observation={"time": 0.0, "zon_reaTRooAir_y": float(temperature)},
        input_metadata={
            "con_oveTSetHea_u": {"Unit": "K"},
            "con_oveTSetCoo_u": {"Unit": "K"},
        },
        measurement_metadata={"zon_reaTRooAir_y": {"Unit": "K"}},
        manifest=manifest,
    )


def test_same_proposer_first_action_is_executed_by_proposal_order_baseline():
    config = TemperatureBandProposalConfig(
        lower_kelvin=294.15,
        upper_kelvin=297.15,
        intervention_risk=0.90,
    )
    provider = temperature_band_proposal_provider(config)
    baseline = ProposalOrderBOPTESTController(provider)

    controls = baseline(_context(300.0))

    assert controls == {
        "con_oveTSetHea_u": 294.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetCoo_u": 297.15,
        "con_oveTSetCoo_activate": 1,
    }
    assert baseline.decisions[0].selected == "apply_temperature_band"
    assert baseline.decisions[0].changed_proposer_order is False


def test_branchpoint_can_override_same_proposal_from_canonical_tool_risk():
    config = TemperatureBandProposalConfig(
        lower_kelvin=294.15,
        upper_kelvin=297.15,
        intervention_goal_gain=0.70,
        embedded_goal_gain=0.20,
        intervention_risk=0.90,
        intervention_cost=0.02,
    )
    provider = temperature_band_proposal_provider(config)
    controller = BranchpointBOPTESTController(provider)

    controls = controller(_context(300.0))

    assert controls == {
        "con_oveTSetHea_activate": 0,
        "con_oveTSetCoo_activate": 0,
    }
    decision = controller.decisions[0]
    assert decision.proposer_first == "apply_temperature_band"
    assert decision.selected == "embedded_control"
    assert decision.changed_proposer_order is True
    scores = {score["action_name"]: score for score in decision.scores}
    assert scores["apply_temperature_band"]["risk"] == pytest.approx(0.90)


def test_branchpoint_keeps_intervention_when_canonical_risk_is_low():
    config = TemperatureBandProposalConfig(
        lower_kelvin=294.15,
        upper_kelvin=297.15,
        intervention_goal_gain=0.70,
        embedded_goal_gain=0.20,
        intervention_risk=0.05,
        intervention_cost=0.02,
    )
    provider = temperature_band_proposal_provider(config)
    controller = BranchpointBOPTESTController(provider)

    controls = controller(_context(300.0))

    assert controls["con_oveTSetCoo_activate"] == 1
    assert controller.decisions[0].selected == "apply_temperature_band"
    assert controller.decisions[0].changed_proposer_order is False


def test_in_band_temperature_leaves_embedded_controller_in_charge():
    config = TemperatureBandProposalConfig(lower_kelvin=294.15, upper_kelvin=297.15)
    provider = temperature_band_proposal_provider(config)

    plans = tuple(provider(_context(295.0)))

    assert len(plans) == 1
    assert plans[0].candidate.name == "embedded_control"
    assert plans[0].controls == {
        "con_oveTSetHea_activate": 0,
        "con_oveTSetCoo_activate": 0,
    }


def test_missing_temperature_measurement_fails_closed():
    config = TemperatureBandProposalConfig(lower_kelvin=294.15, upper_kelvin=297.15)
    provider = temperature_band_proposal_provider(config)
    context = _context(300.0)
    context = BOPTESTControlContext(
        step_index=context.step_index,
        elapsed_seconds=context.elapsed_seconds,
        observation={"time": 0.0},
        input_metadata=context.input_metadata,
        measurement_metadata=context.measurement_metadata,
        manifest=context.manifest,
    )

    with pytest.raises(BOPTESTDecisionAdapterError, match="missing temperature"):
        provider(context)


def test_invalid_temperature_band_is_rejected():
    with pytest.raises(ValueError, match="lower < upper"):
        TemperatureBandProposalConfig(lower_kelvin=300.0, upper_kelvin=295.0)
