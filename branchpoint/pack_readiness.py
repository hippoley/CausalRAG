"""Capability-pack readiness inspection without executing external effects."""

from __future__ import annotations

from typing import Any, Dict, List

from branchpoint.probe import get_probe_scenario


_PLACEHOLDER_GOALS = {
    "replace with the real domain goal.",
}


def inspect_capability_pack(scenario_id: str) -> Dict[str, Any]:
    """Inspect whether a registered pack is ready for a real Workbench run.

    This deliberately never invokes tool handlers. A readiness check must not
    create external effects merely to discover whether a pack is wired.
    """

    spec = get_probe_scenario(scenario_id)
    checks: List[Dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    goal = str(spec.default_goal or "").strip()
    if not goal or goal.lower() in _PLACEHOLDER_GOALS:
        add("goal", "blocked", "replace the generated placeholder with a real domain goal")
    else:
        add("goal", "ok", goal)

    try:
        from branchpoint.probe.runtime import ProbeRunConfig
        from branchpoint.probe.scenarios import build_scenario_runtime

        config = ProbeRunConfig(
            scenario=spec.scenario_id,
            hidden_hypothesis=spec.default_hidden_hypothesis,
            outcome_mode=spec.default_outcome_mode,
        )
        runtime = build_scenario_runtime(config)
        add("builder", "ok", "scenario builder constructed without executing tools")
    except Exception as exc:
        add("builder", "blocked", f"{type(exc).__name__}: {exc}")
        return {
            "schema_version": "branchpoint.pack-readiness.v1",
            "scenario_id": scenario_id,
            "ready": False,
            "checks": checks,
            "blockers": [row["detail"] for row in checks if row["status"] == "blocked"],
            "warnings": [],
        }

    hypotheses = list(runtime.world_model.hypotheses())
    if hypotheses:
        add("hypotheses", "ok", f"{len(hypotheses)} hypotheses available")
    else:
        add("hypotheses", "blocked", "world model has no hypotheses")

    tools = list(runtime.tools)
    if tools:
        add("tools", "ok", f"{len(tools)} tools registered")
    else:
        add("tools", "blocked", "pack exposes no tools")

    placeholder_tools = [
        tool.name
        for tool in tools
        if getattr(tool.handler, "__name__", "") == "_not_connected"
    ]
    if placeholder_tools:
        add(
            "handlers",
            "blocked",
            "connect real domain I/O for: " + ", ".join(sorted(placeholder_tools)),
        )
    else:
        add("handlers", "ok", "no generated placeholder handlers detected")

    uncontracted = [
        tool.name
        for tool in tools
        if getattr(tool, "experiment_contract", None) is None
        and getattr(tool, "intervention_contract", None) is None
    ]
    if uncontracted:
        add(
            "causal_contracts",
            "warning",
            "tools without Experiment/Intervention contracts: "
            + ", ".join(sorted(uncontracted)),
        )
    else:
        add("causal_contracts", "ok", "all tools declare a causal contract")

    if "connect real handlers" in str(spec.recommended_test or "").lower():
        add(
            "recommended_test",
            "warning",
            "generated-pack recommendation still present; replace it with a domain regression test",
        )
    else:
        add("recommended_test", "ok", str(spec.recommended_test or "configured"))

    blockers = [row["detail"] for row in checks if row["status"] == "blocked"]
    warnings = [row["detail"] for row in checks if row["status"] == "warning"]
    return {
        "schema_version": "branchpoint.pack-readiness.v1",
        "scenario_id": scenario_id,
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "truthfulness": {
            "tool_handlers_executed": False,
            "external_effects_created": False,
        },
    }
