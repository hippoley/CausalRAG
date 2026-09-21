import json
import sys

from branchpoint import cli
from branchpoint.probe import unregister_probe_scenario


def test_packs_cli_lists_registered_builtin_capability_packs(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["branchpoint", "packs"])

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "browser_action_guard" in output
    assert "tool_routing" in output
    assert "incident_triage" in output
    assert "temporal_delayed_effect" in output
    assert "open_world_mismatch" in output


def test_packs_cli_json_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["branchpoint", "packs", "--json"])

    assert cli.main() == 0
    rows = json.loads(capsys.readouterr().out)
    by_id = {row["id"]: row for row in rows}
    assert by_id["browser_action_guard"]["default_hidden_hypothesis"] == "H2"
    assert by_id["tool_routing"]["default_outcome_mode"] == "deterministic"


def test_packs_cli_can_load_application_pack_module(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "branchpoint",
            "packs",
            "--load-pack",
            "examples/custom_capability_pack.py:register_pack",
            "--json",
        ],
    )
    try:
        assert cli.main() == 0
        rows = json.loads(capsys.readouterr().out)
        assert "queue_incident_demo" in {row["id"] for row in rows}
    finally:
        unregister_probe_scenario("queue_incident_demo")


def test_packs_cli_reports_missing_loader_as_user_error(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "branchpoint",
            "packs",
            "--load-pack",
            "examples/custom_capability_pack.py:missing_loader",
        ],
    )

    assert cli.main() == 2
