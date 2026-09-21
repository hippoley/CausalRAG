import pytest

from branchpoint import ActionKind, CandidateAction, DecisionResult, ToolSpec, decide


def _candidates():
    return [
        CandidateAction(
            kind=ActionKind.INTERVENE,
            name="submit_form",
            expected_goal_gain=0.95,
            rationale="Submit immediately.",
        ),
        CandidateAction(
            kind=ActionKind.OBSERVE,
            name="inspect_submission_state",
            expected_goal_gain=0.15,
            expected_information_gain=0.70,
            rationale="Verify before creating an external side effect.",
        ),
    ]


def test_decide_uses_canonical_tool_risk_and_reversibility():
    result = decide(
        _candidates(),
        tools=[
            ToolSpec(
                name="submit_form",
                description="Irreversible external submission.",
                handler=lambda: {"submitted": True},
                cost=0.12,
                risk=0.72,
                reversible=False,
                metadata={"kind": "intervene"},
            ),
            ToolSpec(
                name="inspect_submission_state",
                description="Read-only page-state check.",
                handler=lambda: {"ready": False},
                cost=0.02,
                risk=0.0,
                reversible=True,
                metadata={"kind": "observe"},
            ),
        ],
    )

    assert result.proposer_first.name == "submit_form"
    assert result.selected.name == "inspect_submission_state"
    assert result.changed_proposer_order is True

    submit_score = next(score for score in result.scores if score.action_name == "submit_form")
    assert submit_score.risk == pytest.approx(0.72)
    assert submit_score.irreversibility == pytest.approx(1.0)


def test_decide_without_registered_tools_keeps_candidate_level_scoring():
    result = decide(_candidates())

    assert result.selected.name == "submit_form"
    assert result.changed_proposer_order is False


def test_decide_empty_candidates_returns_explicit_stop():
    result = decide([])

    assert result.selected.kind == ActionKind.STOP
    assert result.selected.name == "stop"
    assert result.changed_proposer_order is False


def test_decision_result_distinguishes_equal_but_distinct_candidates():
    proposer = CandidateAction(kind=ActionKind.OBSERVE, name="same")
    selected = CandidateAction(kind=ActionKind.OBSERVE, name="same")
    assert proposer == selected
    result = DecisionResult(
        selected=selected,
        proposer_first=proposer,
        ranked_actions=(selected, proposer),
        scores=(),
    )

    assert result.changed_proposer_order is True
