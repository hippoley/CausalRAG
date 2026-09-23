from __future__ import annotations

import argparse
import json
from pathlib import Path

from branchpoint.benchmarks import (
    BOPTESTArbitrationStudyPlan,
    run_boptest_arbitration_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a preregistered Branchpoint BOPTEST arbitration study."
    )
    parser.add_argument(
        "plan",
        type=Path,
        nargs="?",
        default=Path("examples/boptest_bestest_air_arbitration_study.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boptest_arbitration_study_result.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    plan = BOPTESTArbitrationStudyPlan.from_dict(payload)
    result = run_boptest_arbitration_study(plan)
    artifact = result.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"study_hash={plan.study_hash}")
    print(
        json.dumps(
            artifact["comparison"]["aggregate_paired_kpi_deltas"],
            indent=2,
            sort_keys=True,
        )
    )
    print(
        json.dumps(
            artifact["uncertainty"]["intervals"],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
