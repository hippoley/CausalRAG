import subprocess
import sys

import causalrag


def test_package_version_is_v030():
    assert causalrag.__version__ == "0.3.0"


def test_cli_reports_v030():
    completed = subprocess.run(
        [sys.executable, "-m", "causalrag.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "CausalRAG version 0.3.0"
