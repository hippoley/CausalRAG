from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from branchpoint.agent.actions import ActionKind, CandidateAction
from branchpoint.decision import decide
from branchpoint.tools.base import ToolSpec
from branchpoint.world_model.models import CausalWorldModel


class DecisionPayloadError(ValueError):
    """Raised when a portable one-shot decision payload is invalid."""


def _bounded_float(
    value: Any,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionPayloadError(f"{name} must be numeric") from exc
    if result < minimum:
        raise DecisionPayloadError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise DecisionPayloadError(f"{name} must be <= {maximum}")
    return result


def _rows(value: Any, *, name: str, maximum: int = 20) -> List[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DecisionPayloadError(f"{name} must be a list")
    if len(value) > maximum:
        raise DecisionPayloadError(f"{name} supports at most {maximum} rows")
    rows: List[Mapping[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise DecisionPayloadError(f"{name}[{index}] must be an object")
        rows.append(row)
    return rows


def _non_empty(value: Any, *, name: str, maximum: int = 1000) -> str:
    result = str(value or "").strip()
    if not result:
        raise DecisionPayloadError(f"{name} must be non-empty")
    if len(result) > maximum:
        raise DecisionPayloadError(f"{name} is too long")
    return result


def _candidate_from_row(row: Mapping[str, Any], index: int) -> CandidateAction:
    name = _non_empty(row.get("name"), name=f"candidates[{index}].name", maximum=120)
    kind_raw = str(row.get("kind") or "").strip()
    try:
        kind = ActionKind(kind_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ActionKind)
        raise DecisionPayloadError(
            f"candidates[{index}].kind must be one of: {allowed}"
        ) from exc

    tests_raw = row.get("tests_hypotheses") or []
    if not isinstance(tests_raw, list):
        raise DecisionPayloadError(
            f"candidates[{index}].tests_hypotheses must be a list"
        )
    tests = [str(value).strip() for value in tests_raw if str(value).strip()]

    return CandidateAction(
        kind=kind,
        name=name,
        expected_goal_gain=_bounded_float(
            row.get("expected_goal_gain", 0.0),
            name=f"candidates[{index}].expected_goal_gain",
            maximum=1.0,
        ),
        expected_information_gain=_bounded_float(
            row.get("expected_information_gain", 0.0),
            name=f"candidates[{index}].expected_information_gain",
            maximum=1.0,
        ),
        cost=_bounded_float(
            row.get("cost", 0.0),
            name=f"candidates[{index}].cost",
        ),
        risk=_bounded_float(
            row.get("risk", 0.0),
            name=f"candidates[{index}].risk",
        ),
        irreversibility=_bounded_float(
            row.get("irreversibility", 0.0),
            name=f"candidates[{index}].irreversibility",
        ),
        rationale=str(row.get("rationale") or "")[:1000],
        tests_hypotheses=tests,
    )


def _tool_from_row(row: Mapping[str, Any], index: int) -> ToolSpec:
    name = _non_empty(row.get("name"), name=f"tools[{index}].name", maximum=120)
    reversible = row.get("reversible", True)
    if not isinstance(reversible, bool):
        raise DecisionPayloadError(f"tools[{index}].reversible must be boolean")
    return ToolSpec(
        name=name,
        description=str(row.get("description") or name)[:1000],
        handler=lambda **_kwargs: None,
        cost=_bounded_float(row.get("cost", 0.0), name=f"tools[{index}].cost"),
        risk=_bounded_float(row.get("risk", 0.0), name=f"tools[{index}].risk"),
        reversible=reversible,
    )


def _candidate_payload(candidate: CandidateAction, index: int) -> Dict[str, Any]:
    return {
        "index": int(index),
        "kind": candidate.kind.value,
        "name": candidate.name,
        "expected_goal_gain": float(candidate.expected_goal_gain),
        "expected_information_gain": float(candidate.expected_information_gain),
        "cost": float(candidate.cost),
        "risk": float(candidate.risk),
        "irreversibility": float(candidate.irreversibility),
        "rationale": candidate.rationale,
        "tests_hypotheses": list(candidate.tests_hypotheses),
    }


def arbitrate_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the one-shot decision runtime from a portable JSON-like payload.

    This adapter is deliberately dependency-light so the CLI and HTTP surface
    share the exact same arbitration path without importing FastAPI/Pydantic.
    It never executes supplied tools and never simulates outcomes.
    """

    candidate_rows = _rows(payload.get("candidates"), name="candidates")
    if not candidate_rows:
        raise DecisionPayloadError("candidates must contain at least one row")
    tool_rows = _rows(payload.get("tools"), name="tools")
    hypothesis_rows = _rows(payload.get("hypotheses"), name="hypotheses")

    candidate_names = [
        _non_empty(row.get("name"), name=f"candidates[{index}].name", maximum=120)
        for index, row in enumerate(candidate_rows)
    ]
    if len(candidate_names) != len(set(candidate_names)):
        raise DecisionPayloadError("candidate names must be unique")

    tool_names = [
        _non_empty(row.get("name"), name=f"tools[{index}].name", maximum=120)
        for index, row in enumerate(tool_rows)
    ]
    if len(tool_names) != len(set(tool_names)):
        raise DecisionPayloadError("tool names must be unique")

    candidates = [
        _candidate_from_row(row, index) for index, row in enumerate(candidate_rows)
    ]
    tools = [_tool_from_row(row, index) for index, row in enumerate(tool_rows)]

    world = None
    normalized_hypotheses: List[Dict[str, Any]] = []
    if hypothesis_rows:
        hypothesis_ids = [
            _non_empty(
                row.get("hypothesis_id"),
                name=f"hypotheses[{index}].hypothesis_id",
                maximum=80,
            )
            for index, row in enumerate(hypothesis_rows)
        ]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise DecisionPayloadError("hypothesis ids must be unique")
        weights = [
            _bounded_float(
                row.get("probability"),
                name=f"hypotheses[{index}].probability",
                minimum=1e-12,
                maximum=1_000_000.0,
            )
            for index, row in enumerate(hypothesis_rows)
        ]
        total = sum(weights)
        world = CausalWorldModel()
        for index, (row, hypothesis_id, weight) in enumerate(
            zip(hypothesis_rows, hypothesis_ids, weights)
        ):
            statement = _non_empty(
                row.get("statement"),
                name=f"hypotheses[{index}].statement",
                maximum=1000,
            )
            probability = weight / total
            world.upsert_hypothesis(
                hypothesis_id,
                statement,
                probability=probability,
            )
            normalized_hypotheses.append(
                {
                    "id": hypothesis_id,
                    "statement": statement,
                    "probability": probability,
                }
            )

    result = decide(candidates, tools=tools or None, world_model=world)
    tool_by_name = {row.name: row for row in tools}
    candidate_input_by_name = {
        candidate.name: candidate for candidate in candidates
    }

    ranking: List[Dict[str, Any]] = []
    for rank, (action, score) in enumerate(
        zip(result.ranked_actions, result.scores),
        start=1,
    ):
        submitted = candidate_input_by_name[action.name]
        registered = tool_by_name.get(action.name)
        canonical_overrides: List[str] = []
        if registered is not None:
            if float(registered.cost) > float(submitted.cost):
                canonical_overrides.append("cost")
            if float(registered.risk) > float(submitted.risk):
                canonical_overrides.append("risk")
            if not registered.reversible and float(submitted.irreversibility) < 1.0:
                canonical_overrides.append("irreversibility")

        ranking.append(
            {
                "rank": rank,
                "candidate": _candidate_payload(action, score.candidate_index),
                "score": {
                    "total_utility": score.total_utility,
                    "goal_gain": score.goal_gain,
                    "information_gain": score.information_gain,
                    "information_source": score.information_source,
                    "model_information_gain": score.model_information_gain,
                    "discrimination_score": score.discrimination_score,
                    "bayesian_information_gain": score.bayesian_information_gain,
                    "cost": score.cost,
                    "risk": score.risk,
                    "irreversibility": score.irreversibility,
                    "decision_value": score.decision_value,
                    "decision_value_source": score.decision_value_source,
                    "expected_value_of_sample_information": (
                        score.expected_value_of_sample_information
                    ),
                    "net_value_of_sampling": score.net_value_of_sampling,
                },
                "registered_tool": (
                    {
                        "cost": registered.cost,
                        "risk": registered.risk,
                        "reversible": registered.reversible,
                    }
                    if registered is not None
                    else None
                ),
                "canonical_overrides": canonical_overrides,
            }
        )

    proposer_row = next(
        row for row in ranking if row["candidate"]["index"] == 0
    )
    selected_row = ranking[0]
    selected_index = int(selected_row["candidate"]["index"])

    reasons: List[Dict[str, Any]] = []
    if result.changed_proposer_order:
        reasons.append(
            {
                "code": "higher_runtime_utility",
                "delta": (
                    selected_row["score"]["total_utility"]
                    - proposer_row["score"]["total_utility"]
                ),
            }
        )
    if proposer_row["canonical_overrides"]:
        reasons.append(
            {
                "code": "canonical_tool_policy",
                "fields": proposer_row["canonical_overrides"],
            }
        )
    if selected_row["score"]["information_source"] not in {
        "model_estimate",
        "unanchored_model_estimate",
    }:
        reasons.append(
            {
                "code": "runtime_information_source",
                "source": selected_row["score"]["information_source"],
            }
        )

    return {
        "truthfulness": {
            "executes_tools": False,
            "simulates_outcomes": False,
            "uses_runtime_scoring": True,
            "canonical_tool_policy_applied": bool(tools),
            "next_step_for_real_outcomes": (
                "Run a capability pack or Live Workbench session."
            ),
        },
        "proposer_first": _candidate_payload(result.proposer_first, 0),
        "selected": _candidate_payload(result.selected, selected_index),
        "changed_proposer_order": result.changed_proposer_order,
        "ranking": ranking,
        "reasons": reasons,
        "hypotheses": normalized_hypotheses,
    }
