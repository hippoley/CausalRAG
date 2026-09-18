import time

from causalrag.probe import ProbeRunConfig
from causalrag.probe.session import ProbeSession


def _wait_gate(session, *, different_from=None, timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = session.snapshot()
        if snap["status"] == "waiting_for_human" and snap["pending_decision"]:
            gate = snap["pending_decision"]
            if different_from is None or gate["gate_id"] != different_from:
                return gate
        if snap["status"] == "failed":
            raise AssertionError(snap["error"])
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for gate: {session.snapshot()}")


def _finish(session, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = session.snapshot()
        if snap["status"] == "waiting_for_human":
            session.resolve_decision("approve")
        elif snap["status"] == "completed":
            return snap["result"]
        elif snap["status"] == "failed":
            raise AssertionError(snap["error"])
        time.sleep(0.01)
    raise AssertionError(f"session did not finish: {session.snapshot()}")


def test_timeline_replays_bayesian_world_update_semantically():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="hvac_hidden_world",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    result = _finish(session)
    timeline = result["episode_timeline"]

    assert timeline
    executed = [row for row in timeline if row["executed"]]
    assert executed
    first_observed = next(
        row
        for row in executed
        if row["actual"]["observation"] is not None
        and row["causal_effects"].get("posterior")
    )
    assert first_observed["model"]["candidates"]
    assert first_observed["runtime"]["selected_before_human"]["name"]
    assert first_observed["actual"]["selected"]["name"]
    assert first_observed["actual"]["observation"]["result"]
    assert any(
        change["delta"] not in (None, 0.0)
        for change in first_observed["hypothesis_delta"].values()
    )


def test_timeline_preserves_runtime_choice_human_override_and_actual_action():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="hvac_hidden_world",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    gate = _wait_gate(session)
    runtime_name = gate["runtime_selected"]["name"]
    alternative = next(
        row
        for row in gate["candidates"]
        if row["runtime_valid"] and row["name"] != runtime_name
    )
    session.resolve_decision("choose", alternative["index"])
    result = _finish(session)
    frame = result["episode_timeline"][0]

    assert frame["status"] == "released_for_execution"
    assert frame["runtime"]["selected_before_human"]["name"] == runtime_name
    assert frame["human"]["action"] == "choose"
    assert frame["human"]["candidate"]["name"] == alternative["name"]
    assert frame["actual"]["selected"]["name"] == alternative["name"]


def test_operator_message_replan_remains_as_discarded_nonexecuted_frame():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="hvac_hidden_world",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    first = _wait_gate(session)
    session.add_operator_message(
        "You ignored sensor drift. Reconsider before acting.",
        replan=True,
    )
    _wait_gate(session, different_from=first["gate_id"])
    result = _finish(session)
    timeline = result["episode_timeline"]

    discarded = timeline[0]
    assert discarded["status"] == "discarded_before_execution"
    assert discarded["executed"] is False
    assert discarded["actual"]["selected"] is None
    assert discarded["actual"]["observation"] is None
    assert discarded["human"]["action"] == "operator_message"
    assert "sensor drift" in discarded["human"]["message"]
    assert timeline[1]["step"] == discarded["step"]


def test_manual_hypothesis_addition_is_visible_as_human_event_and_world_delta():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="hvac_hidden_world",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    first = _wait_gate(session)
    session.add_hypothesis("H4", "A human-proposed sensor drift mechanism.", probability=0.2)
    session.resolve_decision("replan")
    _wait_gate(session, different_from=first["gate_id"])
    result = _finish(session)
    frame = result["episode_timeline"][0]

    assert any(event["action"] == "add_hypothesis" for event in frame["human_events"])
    assert frame["hypothesis_delta"]["H4"]["created"] is True
    assert frame["hypothesis_delta"]["H4"]["after"] == 0.2


def test_temporal_timeline_shows_runtime_rewriting_immediate_read_into_wait():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="temporal_delayed_effect",
            hidden_hypothesis="H1",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    result = _finish(session)
    timeline = result["episode_timeline"]

    guarded = next(
        row
        for row in timeline
        if row["runtime"]["selected_before_human"].get("name") == "wait_for_effect_window"
    )
    assert any(
        candidate["name"] == "read_flow"
        for candidate in guarded["model"]["candidates"]
    )
    assert guarded["actual"]["selected"]["kind"] == "wait"
    assert guarded["actual"]["selected"]["name"] == "wait_for_effect_window"
    assert guarded["actual"]["observation"]["result"]["waited"] == 5.0


def test_open_world_timeline_shows_h4_created_after_mismatch_frame():
    session = ProbeSession(
        ProbeRunConfig(
            scenario="open_world_mismatch",
            hidden_hypothesis="H4",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    result = _finish(session)
    timeline = result["episode_timeline"]

    discovery = next(
        row
        for row in timeline
        if row["hypothesis_delta"].get("H4", {}).get("created")
    )
    assert discovery["executed"] is True
    assert discovery["causal_effects"]["model_mismatch"]["escalate"] is True
    assert discovery["causal_effects"]["model_mismatch"]["discovered_hypotheses"] == ["H4"]
    assert discovery["hypothesis_delta"]["H4"]["after"] > 0.0
