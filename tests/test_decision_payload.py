import io
import json
import sys

from branchpoint import cli
from branchpoint.decision_io import DecisionPayloadError, arbitrate_payload


BROWSER_PAYLOAD = {
    "candidates": [
        {
            "kind": "intervene",
            "name": "submit_form",
            "expected_goal_gain": 0.95,
            "risk": 0.0,
        },
        {
            "kind": "observe",
            "name": "inspect_submission_state",
            "expected_goal_gain": 0.15,
            "expected_information_gain": 0.70,
        },
    ],
    "tools": [
        {
            "name": "submit_form",
            "cost": 0.12,
            "risk": 0.72,
            "reversible": False,
        },
        {
            "name": "inspect_submission_state",
            "cost": 0.02,
            "risk": 0.0,
            "reversible": True,
        },
    ],
}


def test_portable_decision_payload_uses_same_canonical_runtime_scoring():
    result = arbitrate_payload(BROWSER_PAYLOAD)

    assert result["proposer_first"]["name"] == "submit_form"
    assert result["selected"]["name"] == "inspect_submission_state"
    assert result["changed_proposer_order"] is True
    submit = next(
        row for row in result["ranking"]
        if row["candidate"]["name"] == "submit_form"
    )
    assert submit["canonical_overrides"] == [
        "cost",
        "risk",
        "irreversibility",
    ]
    assert result["truthfulness"]["executes_tools"] is False
    assert result["truthfulness"]["simulates_outcomes"] is False


def test_portable_decision_payload_normalizes_positive_hypothesis_weights():
    result = arbitrate_payload(
        {
            "candidates": [
                {
                    "kind": "observe",
                    "name": "narrow_probe",
                    "tests_hypotheses": ["H1"],
                    "expected_information_gain": 0.9,
                },
                {
                    "kind": "observe",
                    "name": "broad_probe",
                    "tests_hypotheses": ["H1", "H2"],
                    "expected_information_gain": 0.01,
                },
            ],
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": "release regression",
                    "probability": 40,
                },
                {
                    "hypothesis_id": "H2",
                    "statement": "database saturation",
                    "probability": 60,
                },
            ],
        }
    )

    assert result["selected"]["name"] == "broad_probe"
    assert abs(sum(row["probability"] for row in result["hypotheses"]) - 1.0) < 1e-9
    assert result["ranking"][0]["score"]["information_source"] == (
        "runtime_hypothesis_discrimination"
    )


def test_portable_decision_payload_rejects_duplicate_candidates():
    try:
        arbitrate_payload(
            {
                "candidates": [
                    {"kind": "observe", "name": "duplicate"},
                    {"kind": "intervene", "name": "duplicate"},
                ]
            }
        )
    except DecisionPayloadError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate candidates should be rejected")


def test_decide_cli_reads_json_file_and_prints_branch_split(monkeypatch, tmp_path, capsys):
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(BROWSER_PAYLOAD), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["branchpoint", "decide", str(path)])

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "proposer: submit_form" in output
    assert "runtime:  inspect_submission_state" in output
    assert "branch:   changed" in output
    assert "canonical tool policy: cost, risk, irreversibility" in output


def test_decide_cli_supports_stdin_and_json_output(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(BROWSER_PAYLOAD)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["branchpoint", "decide", "-", "--json"],
    )

    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected"]["name"] == "inspect_submission_state"
    assert output["truthfulness"]["executes_tools"] is False
