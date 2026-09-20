"""Build the static GitHub Pages demo from the real no-key runtime examples."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from causalrag.probe import ProbeRunConfig, run_probe_comparison, run_probe_episode


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DIST = ROOT / "dist"


def run_example(path: str) -> str:
    completed = subprocess.run(
        [sys.executable, path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SITE, DIST)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "bayesian_experiment": run_example("examples/bayesian_experiment_demo.py"),
        "hidden_world_suite": run_example("examples/hidden_world_suite_demo.py"),
        "policy_comparison": run_example("examples/hidden_world_policy_comparison.py"),
        "playable_ab": run_probe_comparison(
            ProbeRunConfig(
                hidden_hypothesis="H2",
                outcome_mode="deterministic",
                seed=0,
                proposer_family="deterministic",
            )
        ),
        "examples": {
            "hvac": run_probe_episode(
                ProbeRunConfig(
                    scenario="hvac_hidden_world",
                    hidden_hypothesis="H2",
                    outcome_mode="deterministic",
                    seed=0,
                    proposer_family="deterministic",
                )
            ),
            "temporal": run_probe_episode(
                ProbeRunConfig(
                    scenario="temporal_delayed_effect",
                    hidden_hypothesis="H1",
                    outcome_mode="deterministic",
                    seed=0,
                    proposer_family="deterministic",
                )
            ),
            "open_world": run_probe_episode(
                ProbeRunConfig(
                    scenario="open_world_mismatch",
                    hidden_hypothesis="H4",
                    outcome_mode="deterministic",
                    seed=0,
                    proposer_family="deterministic",
                )
            ),
        },
    }
    data_dir = DIST / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "runtime.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DIST / ".nojekyll").touch()


if __name__ == "__main__":
    main()
