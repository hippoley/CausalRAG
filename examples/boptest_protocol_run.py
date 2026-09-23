from __future__ import annotations

import argparse
import json
from pathlib import Path

from branchpoint.benchmarks import (
    BOPTESTScenarioManifest,
    no_op_controller,
    run_boptest_episode,
)
from branchpoint.environments import BOPTESTClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen BOPTEST baseline episode.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("boptest_episode_result.json"))
    parser.add_argument("--base-url", default="https://api.boptest.net")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = BOPTESTScenarioManifest.from_dict(payload)
    result = run_boptest_episode(
        manifest,
        no_op_controller,
        controller_id="embedded-baseline",
        client=BOPTESTClient(base_url=args.base_url, timeout=60.0),
    )
    args.output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest_hash={manifest.manifest_hash}")
    print(f"steps={len(result.trajectory)}")
    print("kpis=" + json.dumps(result.kpis, sort_keys=True))
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
