from pathlib import Path
import runpy


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "production_execution_boundary.py"
)
run_demo = runpy.run_path(str(EXAMPLE_PATH))["run_demo"]


def test_production_boundary_reference_composes_all_runtime_guards(tmp_path):
    result = run_demo(tmp_path / "effects.sqlite3")

    assert result["proposer_first"] == "submit_form"
    assert result["runtime_first"] == "inspect_submission_state"
    assert result["inspection"]["session_status"] == "expired"
    assert result["reauthentication"]["session_status"] == "active"
    assert result["submission"]["submitted"] is True
    assert result["replayed_submission"] == result["submission"]
    assert result["external_submit_calls"] == 1
    assert result["guest_denied_before_effect"] is True
    assert result["submit_receipt_status"] == "succeeded"
