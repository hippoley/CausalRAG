from __future__ import annotations

import argparse
import json
from pathlib import Path

from branchpoint.benchmarks import (
    BOPTESTScenarioManifest,
    TemperatureBandProposalConfig,
    run_boptest_comparison,
    temperature_band_controller_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare proposer-order execution with Branchpoint arbitration "
            "using the same transparent BESTEST Air proposer."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boptest_branchpoint_vs_proposer.json"),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--hours", type=float, default=6.0)
    args = parser.parse_args()

    manifest = BOPTESTScenarioManifest(
        testcase="bestest_air",
        start_time=0,
        warmup_period=86400,
        step_seconds=300,
        horizon_seconds=float(args.hours) * 3600.0,
        seed=int(args.seed),
        controlled_inputs=(
            "con_oveTSetHea_u",
            "con_oveTSetCoo_u",
        ),
        measurement_points=("zon_reaTRooAir_y",),
        required_kpis=(
            "ener_tot",
            "cost_tot",
            "tdis_tot",
            "idis_tot",
            "time_rat",
        ),
    )
    config = TemperatureBandProposalConfig(
        lower_kelvin=294.15,
        upper_kelvin=297.15,
        intervention_goal_gain=0.70,
        embedded_goal_gain=0.20,
        intervention_risk=0.10,
        intervention_cost=0.02,
    )
    controllers = temperature_band_controller_specs(config)
    report = run_boptest_comparison(
        [manifest],
        controllers,
        reference_controller_id="proposal-order",
    )
    payload = report.to_dict()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"manifest_hash={manifest.manifest_hash}")
    print(
        json.dumps(
            payload["aggregate_paired_kpi_deltas"],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
