from __future__ import annotations

import json
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "branchpoint.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_doctor_passes_without_model_key():
    completed = _run("doctor")

    assert completed.returncode == 0, completed.stderr
    assert "Branchpoint doctor" in completed.stdout
    assert "PASS" in completed.stdout
    assert "decision_runtime" in completed.stdout
    assert "authorization" in completed.stdout
    assert "durable_receipt" in completed.stdout


def test_doctor_json_exposes_required_and_optional_checks():
    completed = _run("doctor", "--json")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "branchpoint.doctor.v1"
    assert payload["ok"] is True

    by_name = {row["name"]: row for row in payload["checks"]}
    assert by_name["decision_runtime"]["status"] == "ok"
    assert by_name["decision_runtime"]["changed_proposer_order"] is True
    assert by_name["authorization"]["status"] == "ok"
    assert by_name["durable_receipt"]["status"] == "ok"
    assert by_name["durable_receipt"]["handler_calls"] == 1
    assert by_name["durable_receipt"]["receipt_status"] == "succeeded"

    for name in ("openai_agents", "postgres_driver", "observability_sdk"):
        assert by_name[name]["status"] in {"available", "optional"}
        assert by_name[name]["required"] is False
