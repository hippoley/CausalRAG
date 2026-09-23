from branchpoint.probe.replay import build_semantic_replay


def test_semantic_replay_keeps_discarded_preview_separate_from_execution():
    previews = [
        {
            "gate_id": "g1",
            "step": 0,
            "status": "discarded_before_execution",
            "runtime_selected": {"kind": "observe", "name": "scan", "arguments": {}},
            "human_events": [
                {"action": "operator_message", "message": "consider drift", "replan": True},
                {"action": "add_hypothesis", "hypothesis_id": "H4"},
            ],
            "operator_messages": [{"step": 0, "message": "consider drift"}],
            "human_hypothesis_events": [{"step": 0, "hypothesis_id": "H4"}],
            "world_before": {"hypotheses": [{"id": "H1", "probability": 1.0}]},
            "world_after_gate": {
                "hypotheses": [
                    {"id": "H1", "probability": 0.8},
                    {"id": "H4", "probability": 0.2},
                ]
            },
        },
        {
            "gate_id": "g2",
            "step": 0,
            "status": "released_for_execution",
            "runtime_selected": {"kind": "observe", "name": "targeted_scan", "arguments": {}},
        },
    ]
    ledger = [
        {
            "step": 0,
            "selected": {"kind": "observe", "name": "targeted_scan", "arguments": {}},
            "observation": {"action_name": "targeted_scan", "result": {"signal": "drift"}},
            "transition": {"action": "targeted_scan"},
            "world_after": {"hypotheses": [{"id": "H4", "probability": 0.8}]},
            "posterior": [{"id": "H4", "probability": 0.8}],
            "posterior_delta": {"H4": 0.6},
            "hypothesis_changes": {},
        }
    ]

    replay = build_semantic_replay(previews, ledger)

    assert replay["schema_version"] == "branchpoint.semantic-replay.v1"
    assert replay["frame_count"] == 2
    assert replay["executed_count"] == 1
    assert replay["discarded_count"] == 1
    discarded, executed = replay["frames"]
    assert discarded["gate_id"] == "g1"
    assert discarded["outcome"] == "discarded_before_execution"
    assert discarded["executed"] is False
    assert discarded["observation"] is None
    assert discarded["human_hypothesis_events"][0]["hypothesis_id"] == "H4"
    assert discarded["world_after_gate"]["hypotheses"][1]["id"] == "H4"
    assert executed["gate_id"] == "g2"
    assert executed["outcome"] == "executed"
    assert executed["actual_selected"]["name"] == "targeted_scan"
    assert executed["observation"]["result"]["signal"] == "drift"


def test_released_timeout_is_not_guessed_as_execution_without_canonical_evidence():
    replay = build_semantic_replay(
        [
            {
                "gate_id": "timeout",
                "step": 3,
                "status": "released_by_timeout",
                "runtime_selected": {"kind": "observe", "name": "scan", "arguments": {}},
            }
        ],
        [],
    )

    frame = replay["frames"][0]
    assert frame["executed"] is False
    assert frame["outcome"] == "released_without_canonical_outcome"


def test_terminal_stop_is_visible_without_inventing_tool_execution():
    replay = build_semantic_replay(
        [
            {
                "gate_id": "stop",
                "step": 2,
                "status": "released_for_execution",
                "runtime_selected": {"kind": "stop", "name": "stop", "arguments": {"answer": "done"}},
            }
        ],
        [],
        final_decisions=[
            {
                "step": 2,
                "selected": {"kind": "stop", "name": "stop", "arguments": {"answer": "done"}},
            }
        ],
    )

    frame = replay["frames"][0]
    assert frame["outcome"] == "terminal_without_tool_execution"
    assert frame["terminal_decision"] is True
    assert frame["executed"] is False
    assert frame["actual_selected"]["name"] == "stop"
