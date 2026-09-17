"""Build the static GitHub Pages demo from the real no-key runtime examples."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    }
    data_dir = DIST / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "runtime.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DIST / ".nojekyll").touch()


if __name__ == "__main__":
    main()
