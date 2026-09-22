import json
from io import BytesIO

import pytest

from branchpoint import ActionKind, CandidateAction, ToolSpec
from branchpoint.jev import (
    JevError,
    JevProposal,
    jev_then_branchpoint,
    propose_actions_with_jev,
    reorder_candidates_from_jev,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def _candidates():
    return [
        CandidateAction(
            ActionKind.INTERVENE,
            "submit_form",
            expected_goal_gain=0.95,
            rationale="Submit the form now.",
        ),
        CandidateAction(
            ActionKind.OBSERVE,
            "inspect_submission_state",
            expected_goal_gain=0.15,
            expected_information_gain=0.70,
            rationale="Check whether the session is still valid before submitting.",
        ),
    ]


def test_jev_proposal_is_typed_and_does_not_execute(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "next_action": {
                        "type": "choice",
                        "choice": "submit_form",
                        "probabilities": {
                            "submit_form": 0.81,
                            "inspect_submission_state": 0.19,
                        },
                        "confidence": 0.72,
                    }
                },
                "usage": {"input_tokens": 123},
            }
        )

    monkeypatch.setattr("branchpoint.jev.urlopen", fake_urlopen)
    result = propose_actions_with_jev(
        {"session_status": "unknown"},
        _candidates(),
        api_key="test-key",
    )

    assert result.selected_name == "submit_form"
    assert result.probabilities["submit_form"] == pytest.approx(0.81)
    assert result.confidence == pytest.approx(0.72)
    assert captured["body"]["questions"]["next_action"]["type"] == "choice"
    assert set(captured["body"]["questions"]["next_action"]["criteria"]) == {
        "submit_form",
        "inspect_submission_state",
    }


def test_reorder_preserves_candidate_objects():
    candidates = _candidates()
    proposal = JevProposal(
        selected_name="submit_form",
        probabilities={"submit_form": 0.8, "inspect_submission_state": 0.2},
        confidence=0.9,
        model="jev-latest",
        usage={},
        raw_response={},
    )
    ordered = reorder_candidates_from_jev(candidates, proposal)
    assert ordered[0] is candidates[0]
    assert ordered[1] is candidates[1]


def test_branchpoint_can_override_jev_proposal(monkeypatch):
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "next_action": {
                        "type": "choice",
                        "choice": "submit_form",
                        "probabilities": {
                            "submit_form": 0.90,
                            "inspect_submission_state": 0.10,
                        },
                        "confidence": 0.85,
                    }
                },
            }
        )

    monkeypatch.setattr("branchpoint.jev.urlopen", fake_urlopen)

    proposal, decision = jev_then_branchpoint(
        {"session_status": "unknown"},
        _candidates(),
        api_key="test-key",
        tools=[
            ToolSpec(
                "submit_form",
                "External submit",
                lambda: None,
                risk=0.72,
                reversible=False,
            ),
            ToolSpec(
                "inspect_submission_state",
                "Read-only state check",
                lambda: None,
                cost=0.02,
            ),
        ],
    )

    assert proposal.selected_name == "submit_form"
    assert decision.proposer_first.name == "submit_form"
    assert decision.selected.name == "inspect_submission_state"
    assert decision.changed_proposer_order is True


def test_missing_api_key_is_explicit(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        propose_actions_with_jev("state", _candidates())
