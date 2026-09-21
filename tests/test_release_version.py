import subprocess
import sys

import branchpoint


def test_package_version_is_v030():
    assert branchpoint.__version__ == "0.3.0"


def test_cli_reports_v030():
    completed = subprocess.run(
        [sys.executable, "-m", "branchpoint.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "Branchpoint version 0.3.0"
