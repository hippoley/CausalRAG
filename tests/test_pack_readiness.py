import importlib.util
import sys

from branchpoint.pack_readiness import inspect_capability_pack
from branchpoint.probe import unregister_probe_scenario
from branchpoint.scaffold import scaffold_capability_pack


PAYLOAD = {
    "candidates": [
        {"kind": "observe", "name": "inspect", "expected_information_gain": 0.5},
        {"kind": "intervene", "name": "repair", "expected_goal_gain": 0.8},
    ],
    "tools": [
        {"name": "inspect", "cost": 0.01, "risk": 0.0, "reversible": True},
        {"name": "repair", "cost": 0.1, "risk": 0.1, "reversible": False},
    ],
    "hypotheses": [
        {"hypothesis_id": "H1", "statement": "cause one", "probability": 0.6},
        {"hypothesis_id": "H2", "statement": "cause two", "probability": 0.4},
    ],
}


def _load(path):
    spec = importlib.util.spec_from_file_location("_readiness_pack", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_scaffold_is_truthfully_not_ready_and_never_executes_handlers(tmp_path):
    source = scaffold_capability_pack(PAYLOAD, pack_id="readiness_generated")
    path = tmp_path / "pack.py"
    path.write_text(source, encoding="utf-8")
    module = _load(path)
    module.register_pack()

    try:
        report = inspect_capability_pack("readiness_generated")
        assert report["ready"] is False
        assert report["truthfulness"] == {
            "tool_handlers_executed": False,
            "external_effects_created": False,
        }
        assert any("real domain goal" in row for row in report["blockers"])
        assert any("inspect" in row and "repair" in row for row in report["blockers"])
    finally:
        unregister_probe_scenario("readiness_generated")


def test_builtin_pack_is_ready_without_running_episode():
    report = inspect_capability_pack("incident_triage")
    assert report["ready"] is True
    assert report["blockers"] == []
    assert report["truthfulness"]["tool_handlers_executed"] is False
