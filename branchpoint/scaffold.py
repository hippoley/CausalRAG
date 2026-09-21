"""Generate an application capability-pack starter from a portable decision."""

from __future__ import annotations

import json
import keyword
import re
from typing import Any, Mapping

from branchpoint.decision_io import DecisionPayloadError, arbitrate_payload


def _identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip()).strip("_").lower()
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"pack_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned += "_pack"
    return cleaned


def _literal(value: Any) -> str:
    return repr(value)


def scaffold_capability_pack(
    payload: Mapping[str, Any],
    *,
    pack_id: str = "my_capability_pack",
    label: str = "My capability pack",
) -> str:
    """Return a runnable starter pack while preserving the validated decision contract."""

    result = arbitrate_payload(payload)
    normalized_id = _identifier(pack_id, "my_capability_pack")
    normalized_hypotheses = list(result.get("hypotheses") or [])
    candidate_rows = list(payload.get("candidates") or [])
    tool_rows = list(payload.get("tools") or [])
    input_hypotheses = list(payload.get("hypotheses") or [])

    if input_hypotheses:
        source_hypotheses = input_hypotheses
    elif normalized_hypotheses:
        source_hypotheses = [
            {
                "hypothesis_id": row["id"],
                "statement": row["statement"],
                "probability": row["probability"],
            }
            for row in normalized_hypotheses
        ]
    else:
        source_hypotheses = [
            {
                "hypothesis_id": "H1",
                "statement": "Replace with a domain hypothesis.",
                "probability": 1.0,
            }
        ]

    source_payload = {
        "candidates": candidate_rows,
        "tools": tool_rows,
        "hypotheses": source_hypotheses,
    }
    payload_json = json.dumps(source_payload, ensure_ascii=False, indent=2)

    hypothesis_ids = tuple(
        str(row["hypothesis_id"]) for row in source_hypotheses
    )
    default_hypothesis = max(
        source_hypotheses,
        key=lambda row: float(row.get("probability", 0.0)),
    )["hypothesis_id"]

    return f'''"""Branchpoint capability-pack starter generated from a validated decision.

The decision contract below is preserved exactly. Replace the placeholder
environment/tool handlers with real domain I/O before treating this as a live pack.
"""

import json

from branchpoint import ActionKind, CandidateAction, ToolSpec
from branchpoint.experiments import ExperimentContract, InterventionContract, OutcomeLikelihood
from branchpoint.probe import ProbeScenarioRuntime, ProbeScenarioSpec, register_probe_scenario
from branchpoint.world_model import CausalWorldModel

PACK_ID = {_literal(normalized_id)}
SOURCE_DECISION = json.loads({_literal(payload_json)})


def _not_connected(**kwargs):
    raise NotImplementedError(
        "Generated capability-pack tool handler is not connected to real domain I/O."
    )


class GeneratedEnvironment:
    def __init__(self):
        self.world = CausalWorldModel()
        for row in SOURCE_DECISION.get("hypotheses", []):
            self.world.upsert_hypothesis(
                row["hypothesis_id"],
                row["statement"],
                probability=float(row["probability"]),
            )

    def world_model(self):
        return self.world

    def tools(self):
        # TODO: replace no-op handlers with real read/write boundaries.
        tools = []
        for row in SOURCE_DECISION.get("tools", []):
            experiment = row.get("experiment_contract")
            experiment_contract = None
            if experiment:
                experiment_contract = ExperimentContract(
                    experiment_id=experiment["experiment_id"],
                    outcomes=[
                        OutcomeLikelihood(
                            outcome=outcome["outcome"],
                            likelihoods=dict(outcome["likelihoods"]),
                        )
                        for outcome in experiment.get("outcomes", [])
                    ],
                    outcome_key=experiment.get("outcome_key", "outcome"),
                    description=experiment.get("description", ""),
                )

            intervention = row.get("intervention_contract")
            intervention_contract = None
            if intervention:
                intervention_contract = InterventionContract(
                    intervention_id=intervention["intervention_id"],
                    utilities=dict(intervention["utilities"]),
                    description=intervention.get("description", ""),
                )

            tools.append(
                ToolSpec(
                    name=row["name"],
                    description=row.get("description") or row["name"],
                    handler=lambda **kwargs: {{"status": "not_connected", "arguments": kwargs}},
                    cost=float(row.get("cost", 0.0)),
                    risk=float(row.get("risk", 0.0)),
                    reversible=bool(row.get("reversible", True)),
                    experiment_contract=experiment_contract,
                    intervention_contract=intervention_contract,
                )
            )
        return tools

    def metrics(self, result):
        # TODO: replace with domain success/regret metrics.
        return {{"success": False, "scaffold": True}}


class GeneratedReasoner:
    def propose(self, state, world_model):
        rows = []
        for row in SOURCE_DECISION["candidates"]:
            rows.append(
                CandidateAction(
                    kind=ActionKind(row["kind"]),
                    name=row["name"],
                    expected_goal_gain=float(row.get("expected_goal_gain", 0.0)),
                    expected_information_gain=float(row.get("expected_information_gain", 0.0)),
                    cost=float(row.get("cost", 0.0)),
                    risk=float(row.get("risk", 0.0)),
                    irreversibility=float(row.get("irreversibility", 0.0)),
                    rationale=row.get("rationale") or "",
                    tests_hypotheses=list(row.get("tests_hypotheses") or []),
                )
            )
        return rows

    def uncertainty(self, state, world_model):
        return None


def build_pack(config):
    environment = GeneratedEnvironment()
    return ProbeScenarioRuntime(
        scenario_id=PACK_ID,
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=GeneratedReasoner(),
        goal={_literal("Replace with the real domain goal.")},
    )


def register_pack():
    return register_probe_scenario(
        ProbeScenarioSpec(
            scenario_id=PACK_ID,
            label={_literal(label)},
            description="Generated from a validated portable Branchpoint decision.",
            hidden_hypotheses={_literal(hypothesis_ids)},
            outcome_modes=("deterministic",),
            recommended_test="Connect real handlers, then add domain regression tests.",
            default_hidden_hypothesis={_literal(str(default_hypothesis))},
            default_outcome_mode="deterministic",
            default_goal="Replace with the real domain goal.",
            builder=build_pack,
        ),
        replace=True,
    )
'''
