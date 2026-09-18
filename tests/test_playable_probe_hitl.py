import time

from causalrag.probe import ProbeRunConfig
from causalrag.probe.session import ProbeSession


def _wait_until(session, target, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = session.status()
        if status == target:
            return
        if status in {"completed", "failed"} and status != target:
            raise AssertionError(f"session ended early with {status}: {session.snapshot()}")
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {target}: {session.snapshot()}")


def _finish_by_approving(session):
    deadline = time.time() + 10.0
    while time.time() < deadline:
        status = session.status()
        if status == "waiting_for_human":
            session.resolve_decision("approve")
        elif status == "completed":
            return session.snapshot()
        elif status == "failed":
            raise AssertionError(session.snapshot()["error"])
        time.sleep(0.01)
    raise AssertionError(f"session did not finish: {session.snapshot()}")


def test_human_gate_pauses_before_first_tool_and_can_approve_to_completion():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    first = session.snapshot()
    assert first["result"] is None
    assert first["pending_decision"]["candidates"]
    assert first["pending_decision"]["runtime_selected"]["name"]
    assert first["trace_count"] > 0

    final = _finish_by_approving(session)
    assert final["status"] == "completed"
    assert final["result"]["metrics"]["success"] is True
    names = [row["name"] for row in final["result"]["causal_trace"]]
    assert "causalrag.human_gate.waiting" in names
    assert "causalrag.human_gate.approved" in names


def test_human_can_override_runtime_candidate_and_trace_matches_override():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    pending = session.snapshot()["pending_decision"]
    runtime_name = pending["runtime_selected"]["name"]
    alternative = next(
        row for row in pending["candidates"] if row["name"] != runtime_name
    )
    session.resolve_decision("choose", alternative["index"])

    final = _finish_by_approving(session)
    first_decision = final["result"]["decisions"][0]
    assert first_decision["selected"]["name"] == alternative["name"]
    override_events = [
        row for row in final["result"]["causal_trace"]
        if row["name"] == "causalrag.human_gate.override"
    ]
    assert override_events
    assert override_events[0]["attributes"]["causalrag.action.name"] == alternative["name"]


def test_human_can_add_provisional_hypothesis_while_paused():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    added = session.add_hypothesis(
        "H4",
        "The airflow sensor is drifting.",
        probability=0.25,
    )
    assert added["origin"] == "human"
    assert added["validated"] is False
    assert added["probability"] == 0.25
    assert any(row["hypothesis_id"] == "H4" for row in session.snapshot()["hypotheses"])
    session.resolve_decision("approve")
    session.gate.close()


def test_human_world_model_edit_can_force_replan_before_execution():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    first_gate = session.snapshot()["pending_decision"]["gate_id"]
    assert session.environment.probes == 0

    session.add_hypothesis("H4", "The sensor itself may be drifting.", probability=0.2)
    session.resolve_decision("replan")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = session.snapshot()
        pending = snap["pending_decision"]
        if snap["status"] == "waiting_for_human" and pending and pending["gate_id"] != first_gate:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("replanned decision gate did not appear")

    assert session.environment.probes == 0
    assert any(row["hypothesis_id"] == "H4" for row in snap["hypotheses"])
    session.resolve_decision("approve")
    final = _finish_by_approving(session)
    names = [row["name"] for row in final["result"]["causal_trace"]]
    assert "causalrag.human_gate.replan" in names
