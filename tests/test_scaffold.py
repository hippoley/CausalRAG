import importlib.util
import sys

from branchpoint import cli
from branchpoint.probe import available_probe_config, unregister_probe_scenario
from branchpoint.scaffold import scaffold_capability_pack


PAYLOAD = {
    "candidates": [
        {
            "kind": "observe",
            "name": "diagnose",
            "expected_information_gain": 0.4,
            "tests_hypotheses": ["H1", "H2"],
        },
        {
            "kind": "intervene",
            "name": "repair",
            "expected_goal_gain": 0.8,
        },
    ],
    "tools": [
        {"name": "diagnose", "cost": 0.02, "risk": 0.0, "reversible": True},
        {"name": "repair", "cost": 0.2, "risk": 0.1, "reversible": False},
    ],
    "hypotheses": [
        {"hypothesis_id": "H1", "statement": "cause one", "probability": 0.7},
        {"hypothesis_id": "H2", "statement": "cause two", "probability": 0.3},
    ],
}


def _load_generated(path):
    spec = importlib.util.spec_from_file_location("_generated_pack_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scaffold_source_is_importable_and_registers_pack(tmp_path):
    source = scaffold_capability_pack(
        PAYLOAD,
        pack_id="Generated Demo",
        label="Generated demo",
    )
    path = tmp_path / "generated_pack.py"
    path.write_text(source, encoding="utf-8")
    module = _load_generated(path)

    try:
        module.register_pack()
        by_id = {row["id"]: row for row in available_probe_config()["scenarios"]}
        assert "generated_demo" in by_id
        assert by_id["generated_demo"]["label"] == "Generated demo"
        runtime = module.build_pack(
            type("Config", (), {"hidden_hypothesis": "H1"})()
        )
        assert [tool.name for tool in runtime.tools] == ["diagnose", "repair"]
        proposed = runtime.default_reasoner.propose(None, runtime.world_model)
        assert [row.name for row in proposed] == ["diagnose", "repair"]
    finally:
        unregister_probe_scenario("generated_demo")


def test_scaffold_cli_writes_python_file(monkeypatch, tmp_path, capsys):
    import json

    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    output = tmp_path / "my_pack.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "branchpoint",
            "scaffold",
            str(decision),
            "--pack-id",
            "incident_pack",
            "--label",
            "Incident pack",
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    assert output.exists()
    assert "PACK_ID = 'incident_pack'" in output.read_text(encoding="utf-8")
    assert "Wrote capability-pack starter" in capsys.readouterr().out


def test_scaffold_cli_rejects_invalid_decision(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"candidates":[]}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["branchpoint", "scaffold", str(bad)],
    )

    assert cli.main() == 2


def test_scaffold_preserves_experiment_and_intervention_contracts(tmp_path):
    import json
    from pathlib import Path

    source_payload = json.loads(
        Path("examples/value_of_information.json").read_text(encoding="utf-8")
    )
    source = scaffold_capability_pack(
        source_payload,
        pack_id="evsi_scaffold_test",
        label="EVSI scaffold test",
    )
    path = tmp_path / "evsi_pack.py"
    path.write_text(source, encoding="utf-8")
    module = _load_generated(path)

    runtime = module.build_pack(
        type("Config", (), {"hidden_hypothesis": "H1"})()
    )
    tools = {tool.name: tool for tool in runtime.tools}
    assert tools["diagnose"].experiment_contract is not None
    assert tools["diagnose"].experiment_contract.experiment_id == "diagnostic"
    assert tools["fix_h1"].intervention_contract is not None
    assert tools["fix_h1"].intervention_contract.utilities == {
        "H1": 1.0,
        "H2": -1.0,
    }
