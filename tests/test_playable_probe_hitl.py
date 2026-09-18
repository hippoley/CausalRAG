from causalrag.agent.actions import ActionKind, ActionScore, CandidateAction, DecisionRecord
from causalrag.agent.state import AgentState
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry
from causalrag.world_model import CausalWorldModel
import time

from causalrag.probe import ProbeRunConfig
from causalrag.probe.session import ProbeSession, _decision_inspector


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
    assert any((row.get("hypothesis_id") or row.get("id")) == "H4" for row in session.snapshot()["hypotheses"])
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
    assert any((row.get("hypothesis_id") or row.get("id")) == "H4" for row in snap["hypotheses"])
    session.resolve_decision("approve")
    final = _finish_by_approving(session)
    names = [row["name"] for row in final["result"]["causal_trace"]]
    assert "causalrag.human_gate.replan" in names


def test_human_gate_response_is_single_assignment_and_cannot_be_overwritten():
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
        row
        for row in pending["candidates"]
        if row["runtime_valid"] and row["name"] != runtime_name
    )
    session.resolve_decision("choose", alternative["index"])
    assert session.snapshot()["pending_decision"] is None

    try:
        session.resolve_decision("approve")
    except RuntimeError:
        pass
    else:
        raise AssertionError("second human response must not overwrite first response")

    final = _finish_by_approving(session)
    assert final["result"]["decisions"][0]["selected"]["name"] == alternative["name"]


def test_operator_message_is_part_of_next_llm_proposer_prompt():
    state = AgentState(goal="Diagnose the system", max_steps=4)
    state.scratch["operator_messages"] = [
        {"step": 0, "message": "You ignored sensor drift. Reconsider it."}
    ]
    reasoner = LLMCausalReasoner(llm=None, tools=ToolRegistry())
    prompt = reasoner._build_prompt(state, CausalWorldModel())
    assert "You ignored sensor drift. Reconsider it." in prompt
    assert "operator_messages" in prompt


def test_operator_message_replans_before_old_action_executes():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    first = session.snapshot()["pending_decision"]["gate_id"]
    assert session.environment.probes == 0

    row = session.add_operator_message(
        "You ignored sensor drift. Reconsider before acting.",
        replan=True,
    )
    assert row["message"].startswith("You ignored")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = session.snapshot()
        pending = snap["pending_decision"]
        if snap["status"] == "waiting_for_human" and pending and pending["gate_id"] != first:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("operator message did not trigger a new gate")

    assert session.environment.probes == 0
    session.resolve_decision("approve")
    final = _finish_by_approving(session)
    names = [record["name"] for record in final["result"]["causal_trace"]]
    assert "causalrag.human.operator_message" in names
    assert "causalrag.human_gate.replan" in names


def test_episode_ledger_appears_only_after_execution_and_tracks_posterior():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    _wait_until(session, "waiting_for_human")
    first = session.snapshot()["pending_decision"]
    assert first["episode_ledger"] == []

    session.resolve_decision("approve")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = session.snapshot()
        pending = snap["pending_decision"]
        if snap["status"] == "waiting_for_human" and pending and pending["episode_ledger"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("completed step did not appear in episode ledger")

    row = pending["episode_ledger"][0]
    assert row["selected"]["name"]
    assert row["observation"]["action_name"] == row["selected"]["name"]
    assert row["prior"]
    assert row["posterior"]
    assert row["posterior_delta"]
    assert row["human"]["action"] == "approve"
    session.resolve_decision("approve")
    _finish_by_approving(session)


def test_episode_ledger_records_human_override_as_executed_action():
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
        row
        for row in pending["candidates"]
        if row["runtime_valid"] and row["name"] != runtime_name
    )
    session.resolve_decision("choose", alternative["index"])

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = session.snapshot()
        next_gate = snap["pending_decision"]
        if snap["status"] == "waiting_for_human" and next_gate and next_gate["episode_ledger"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("override execution did not reach next gate")

    row = next_gate["episode_ledger"][0]
    assert row["selected"]["name"] == alternative["name"]
    assert row["observation"]["action_name"] == alternative["name"]
    assert row["human"]["action"] == "choose"
    assert row["human"]["runtime_original"] == runtime_name
    session.resolve_decision("approve")
    _finish_by_approving(session)


def test_replan_does_not_create_ghost_episode_ledger_step():
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
    session.add_operator_message("Reconsider before executing.", replan=True)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = session.snapshot()
        pending = snap["pending_decision"]
        if snap["status"] == "waiting_for_human" and pending and pending["gate_id"] != first_gate:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("replan did not reach a replacement gate")

    assert pending["episode_ledger"] == []
    assert session.environment.probes == 0
    session.resolve_decision("approve")
    _finish_by_approving(session)


def test_decision_inspector_explains_runtime_reordering_from_structured_scores_only():
    proposer_first = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="read_temperature",
        expected_information_gain=0.95,
        rationale="Model prefers this observation.",
    )
    runtime_choice = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="measure_pressure",
        expected_information_gain=0.05,
        rationale="Runtime experiment contract makes this diagnostic.",
    )
    decision = DecisionRecord(
        step=0,
        uncertainty="H1 vs H2",
        candidates=[proposer_first, runtime_choice],
        selected=runtime_choice,
        beliefs_before={"hypotheses": []},
        action_scores=[
            ActionScore(
                candidate_index=1,
                action_name="measure_pressure",
                action_kind=ActionKind.OBSERVE,
                total_utility=0.70,
                goal_gain=0.10,
                information_gain=0.65,
                information_source="runtime_bayesian_eig",
                model_information_gain=0.05,
                discrimination_score=None,
                bayesian_information_gain=0.65,
                cost=0.05,
                risk=0.0,
                irreversibility=0.0,
            ),
            ActionScore(
                candidate_index=0,
                action_name="read_temperature",
                action_kind=ActionKind.OBSERVE,
                total_utility=0.05,
                goal_gain=0.10,
                information_gain=0.0,
                information_source="unanchored_model_estimate",
                model_information_gain=0.95,
                discrimination_score=None,
                bayesian_information_gain=None,
                cost=0.05,
                risk=0.0,
                irreversibility=0.0,
            ),
        ],
    )

    inspector = _decision_inspector(decision)
    assert inspector["diverged"] is True
    assert inspector["divergence_kind"] == "runtime_reordered"
    assert inspector["proposer_first"]["name"] == "read_temperature"
    assert inspector["runtime_selected"]["name"] == "measure_pressure"
    codes = {row["code"] for row in inspector["reasons"]}
    assert "runtime_information_source" in codes
    assert "higher_runtime_utility" in codes
    assert inspector["runtime_selected_score"]["information_source"] == "runtime_bayesian_eig"


def test_session_export_is_self_contained_and_replayable_from_ledger():
    session = ProbeSession(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    session.start()
    final = _finish_by_approving(session)
    artifact = session.export_payload()

    assert artifact["schema_version"] == "causalrag.playable_probe.session.v1"
    assert artifact["session_id"] == session.session_id
    assert artifact["status"] == "completed"
    assert artifact["result"]["metrics"]["success"] is True
    assert artifact["episode_ledger"] == final["result"]["episode_ledger"]
    assert len(artifact["episode_ledger"]) == len(final["result"]["observations"])
    assert artifact["trace"]
    assert artifact["world_model"]["hypotheses"]
    assert artifact["interactions"]["human_gate_history"]
    assert all("decision_inspector" in row for row in artifact["episode_ledger"])
