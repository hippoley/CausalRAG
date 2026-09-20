import time

from fastapi.testclient import TestClient

from causalrag.interface.probe_api import app


client = TestClient(app)


def test_probe_health_and_config():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["surface"] == "playable_probe"

    config = client.get("/api/config")
    assert config.status_code == 200
    body = config.json()
    assert "small" in body["proposer_families"]
    assert "frontier" in body["proposer_families"]


def test_probe_surfaces_are_separated():
    landing = client.get("/")
    assert landing.status_code == 200
    assert "CAUSALRAG · TWO WAYS IN" in landing.text
    assert 'href="/demo"' in landing.text
    assert 'href="/workbench"' in landing.text
    assert 'href="/research"' in landing.text

    demo = client.get("/demo")
    assert demo.status_code == 200
    assert "THREE THINGS ORDINARY AGENTS GET WRONG" in demo.text
    assert "Bayesian learning / EIG / EVSI" in demo.text
    assert "temporal_delayed_effect" in demo.text
    assert "open_world_mismatch" in demo.text
    assert "fetch('/api/run'" in demo.text

    workbench = client.get("/workbench")
    assert workbench.status_code == 200
    assert "CausalRAG · Live Workbench" in workbench.text
    assert "AGENT HISTORY" in workbench.text
    assert "Start live agent" in workbench.text
    assert "Safe Auto evidence" in workbench.text
    assert "WORLD MODEL CHANGED" in workbench.text
    assert "applyLaunchParams" in workbench.text
    assert "Test live model" in workbench.text
    assert "OWNER ACCESS · external models are locked" in workbench.text
    assert "Unlock external models" in workbench.text
    assert "fetch('/api/access'" in workbench.text
    assert "proposerLabel" in workbench.text
    assert "HUMAN GATE · BEFORE EXECUTION" in workbench.text
    assert "Add provisional hypothesis" in workbench.text
    assert "fetch('/api/sessions'" in workbench.text
    assert 'href="/research"' in workbench.text

    research = client.get("/research")
    assert research.status_code == 200
    assert "CausalRAG · Playable Causal Probe" in research.text
    assert "Runtime capabilities" in research.text
    assert "Start interactive episode" in research.text
    assert "CAUSAL AGENT · COMPLETE LIVE FLOW" in research.text
    assert "LIVE WORKBENCH · REAL AGENT CONTROL PLANE" in research.text
    assert "TASK COMPOSER" in research.text
    assert "EVIDENCE & DEBUG" in research.text
    assert "STEP INSPECTOR" in research.text
    assert "candidateHTML" in research.text
    assert "POSTERIOR UPDATE BASIS" in research.text
    assert "SCORE PROVENANCE" in research.text
    assert "WORLD SNAPSHOT · BEFORE" in research.text
    assert "HUMAN / OPERATOR CONTEXT" in research.text
    assert "RELATED TELEMETRY" in research.text
    assert "COUNTERFACTUAL" in research.text
    assert "Truthful full-trajectory fork available" in research.text
    assert "runCounterfactual" in research.text
    assert "EIG = (H(prior)" in research.text
    assert "EVSI = expected best value after" in research.text
    assert "inspectStep" in research.text
    assert "InteractiveDecisionGate" in research.text
    assert "Challenge scenario" in research.text
    assert 'id="scenario"' in research.text
    assert "scenario:$('scenario').value" in research.text
    assert "Test live model" in research.text
    assert "OWNER ACCESS · external models locked" in research.text
    assert "wbUnlock" in research.text
    assert "fetch('/api/access'" in research.text
    assert "PROPOSER AUDIT · FORMAL MODEL SUBMISSIONS" in research.text
    assert "formal structured submission" in research.text
    assert "MODEL PROPOSER FAILED" in research.text

def test_probe_step_context_api_exposes_full_frozen_debug_state():
    created = client.post(
        "/api/sessions",
        json={
            "scenario": "temporal_delayed_effect",
            "hidden_hypothesis": "H1",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            step = snap["pending_decision"]["step"]
            ctx = client.get(f"/api/sessions/{session_id}/steps/{step}")
            assert ctx.status_code == 200
            body = ctx.json()
            assert body["status"] == "pending"
            assert "world_before" in body
            assert "candidates" in body
            assert "action_scores" in body
            assert "score_provenance" in body
            assert body["proposer"]["kind"] == "deterministic"
            assert body["proposer_attempts"]
            assert "operator_context" in body
            assert "telemetry" in body
            assert body["counterfactual"]["available"] is False
            assert body["counterfactual"]["whole_run_ab_available"] is True
            break
        time.sleep(0.01)
    else:
        raise AssertionError("session did not reach a human gate")

    client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})


def test_probe_completed_step_counterfactual_fork_is_real_and_not_frontend_simulation():
    created = client.post(
        "/api/sessions",
        json={
            "scenario": "hvac_hidden_world",
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    first_completed_step = None
    first_candidates = None
    deadline = time.time() + 8.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            pending = snap["pending_decision"]
            ledger = pending.get("episode_ledger") or []
            if ledger and first_completed_step is None:
                first_completed_step = ledger[0]["step"]
                first_candidates = ledger[0]["candidates"]
                break
            approved = client.post(
                f"/api/sessions/{session_id}/decision",
                json={"action": "approve"},
            )
            assert approved.status_code == 200
        elif snap["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert first_completed_step is not None
    ctx = client.get(f"/api/sessions/{session_id}/steps/{first_completed_step}")
    assert ctx.status_code == 200
    body = ctx.json()
    assert body["status"] == "completed"
    assert body["counterfactual"]["available"] is True

    actual = body["selected"]
    alternative = next(
        row for row in first_candidates
        if (row["name"], row["kind"]) != (actual["name"], actual["kind"])
    )
    forked = client.post(
        f"/api/sessions/{session_id}/steps/{first_completed_step}/counterfactual",
        json={"candidate_index": alternative["index"]},
    )
    assert forked.status_code == 200
    fork = forked.json()
    assert fork["available"] is True
    assert fork["truthfulness"]["environment_rebuilt_from_same_config"] is True
    assert fork["truthfulness"]["runtime_guards_reapplied"] is True
    assert fork["truthfulness"]["front_end_simulation"] is False
    assert fork["counterfactual"]["requested_candidate"]["name"] == alternative["name"]
    assert fork["truthfulness"]["canonical_reasoner_resumed_after_branch"] is True
    assert fork["truthfulness"]["full_branch_ran_to_stop_or_budget"] is True
    assert fork["branch_trajectory_decision_count"] >= first_completed_step + 1
    assert "decisions" in fork["counterfactual"]
    assert "observations" in fork["counterfactual"]
    assert "first_future_divergence" in fork
    assert "metric_deltas_counterfactual_minus_actual" in fork

    # Release the live session if it is still waiting at a later gate.
    snap = client.get(f"/api/sessions/{session_id}").json()
    if snap["status"] == "waiting_for_human":
        client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})


def test_probe_pending_step_prefers_live_override_over_counterfactual():
    created = client.post(
        "/api/sessions",
        json={
            "scenario": "temporal_delayed_effect",
            "hidden_hypothesis": "H1",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    session_id = created.json()["session_id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            step = snap["pending_decision"]["step"]
            body = client.get(f"/api/sessions/{session_id}/steps/{step}").json()
            assert body["counterfactual"]["available"] is False
            assert body["counterfactual"]["replay_mode"] == "live_override_preferred"
            client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})
            break
        time.sleep(0.01)
    else:
        raise AssertionError("session did not reach a human gate")


def test_probe_deterministic_no_key_run():
    response = client.post(
        "/api/run",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["success"] is True
    assert body["trace_id"]
    assert body["decisions"]
    assert body["causal_trace"]


def test_probe_interactive_session_api_pauses_and_accepts_human_decisions():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    deadline = time.time() + 8.0
    saw_gate = False
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            saw_gate = True
            pending = snap["pending_decision"]
            assert pending["runtime_selected"]["name"]
            approved = client.post(
                f"/api/sessions/{session_id}/decision",
                json={"action": "approve"},
            )
            assert approved.status_code == 200
        elif snap["status"] == "completed":
            assert saw_gate is True
            assert snap["result"]["metrics"]["success"] is True
            break
        elif snap["status"] == "failed":
            raise AssertionError(snap["error"])
        time.sleep(0.01)
    else:
        raise AssertionError("interactive API session did not complete")


def test_probe_human_hypothesis_api_requires_paused_session():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    session_id = created.json()["session_id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            break
        time.sleep(0.01)
    response = client.post(
        f"/api/sessions/{session_id}/hypotheses",
        json={
            "hypothesis_id": "H4",
            "statement": "The airflow sensor is drifting.",
            "probability": 0.2,
        },
    )
    assert response.status_code == 200
    assert response.json()["hypothesis"]["origin"] == "human"
    client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})


def test_probe_config_reports_model_connection_state_without_secrets():
    body = client.get("/api/config").json()
    assert set(body["model_connections"]) == {"openai", "anthropic", "local"}
    assert "configured" in body["model_connections"]["openai"]
    assert "credential_source" in body["model_connections"]["openai"]
    assert body["access"]["required_for_external_models"] is False
    assert body["access"]["authenticated"] is True
    assert "api_key" not in str(body).lower()


def test_model_test_endpoint_reports_provider_result_without_network(monkeypatch):
    monkeypatch.setattr(
        "causalrag.interface.probe_api.LLMInterface.generate",
        lambda self, prompt, temperature=0.0, max_tokens=32: "CAUSALRAG_MODEL_OK",
    )
    response = client.post(
        "/api/model/test",
        json={"provider": "local", "model": "test-model"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_probe_api_can_replan_after_human_hypothesis_without_executing_old_action():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    session_id = created.json()["session_id"]
    deadline = time.time() + 5.0
    first_gate = None
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            first_gate = snap["pending_decision"]["gate_id"]
            break
        time.sleep(0.01)
    assert first_gate

    added = client.post(
        f"/api/sessions/{session_id}/hypotheses",
        json={"hypothesis_id": "H4X", "statement": "A new mechanism is possible."},
    )
    assert added.status_code == 200
    replanned = client.post(
        f"/api/sessions/{session_id}/decision",
        json={"action": "replan"},
    )
    assert replanned.status_code == 200

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        pending = snap.get("pending_decision")
        if snap["status"] == "waiting_for_human" and pending and pending["gate_id"] != first_gate:
            assert any((row.get("hypothesis_id") or row.get("id")) == "H4X" for row in snap["hypotheses"])
            break
        time.sleep(0.01)
    else:
        raise AssertionError("new decision gate did not appear after replan")

    client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})


def test_probe_operator_message_endpoint_replans_paused_session():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    session_id = created.json()["session_id"]
    deadline = time.time() + 5.0
    first_gate = None
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            first_gate = snap["pending_decision"]["gate_id"]
            break
        time.sleep(0.01)
    assert first_gate

    sent = client.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "message": "You may be missing sensor drift. Reconsider.",
            "replan": True,
        },
    )
    assert sent.status_code == 200

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        pending = snap.get("pending_decision")
        if snap["status"] == "waiting_for_human" and pending and pending["gate_id"] != first_gate:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("operator message did not create a new decision gate")

    client.post(f"/api/sessions/{session_id}/decision", json={"action": "approve"})


def test_probe_compare_api_runs_same_world_causal_and_vanilla_arms():
    response = client.post(
        "/api/compare",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["comparison"]["same_world_inputs"] is True
    assert body["vanilla"]["metrics"]["success"] is False
    assert body["causal"]["metrics"]["success"] is True
    assert body["first_divergence"] is not None


def test_probe_ladder_api_exposes_marginal_capability_effects():
    response = client.post(
        "/api/ladder",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    ids = [row["id"] for row in body["arms"]]
    assert ids == [
        "vanilla",
        "runtime_eig",
        "bayesian_learning",
        "decision_value",
        "temporal_attribution",
        "open_world",
        "full",
    ]
    by_id = {row["id"]: row for row in body["arms"]}
    assert by_id["runtime_eig"]["episode"]["metrics"]["success"] is False
    assert by_id["bayesian_learning"]["episode"]["metrics"]["success"] is True
    assert by_id["bayesian_learning"]["marginal_delta_from_previous"]["success"] == 1.0


def test_probe_config_api_exposes_temporal_and_open_world_scenarios():
    response = client.get("/api/config")
    assert response.status_code == 200
    scenarios = {row["id"]: row for row in response.json()["scenarios"]}
    assert "temporal_delayed_effect" in scenarios
    assert "open_world_mismatch" in scenarios
    assert scenarios["open_world_mismatch"]["default_hidden_hypothesis"] == "H4"


def test_probe_api_runs_temporal_scenario_with_runtime_guard():
    response = client.post(
        "/api/run",
        json={
            "scenario": "temporal_delayed_effect",
            "hidden_hypothesis": "H1",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["success"] is True
    assert body["metrics"]["wait_actions"] == 1
    assert body["metrics"]["premature_reads"] == 0


def test_probe_api_runs_open_world_scenario_and_validates_h4():
    response = client.post(
        "/api/run",
        json={
            "scenario": "open_world_mismatch",
            "hidden_hypothesis": "H4",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["success"] is True
    assert body["metrics"]["discovered_hypothesis"] is True
    assert body["metrics"]["discovered_validated"] is True
    changes = [row.get("hypothesis_changes") or {} for row in body["episode_ledger"]]
    assert any(row.get("structural_change") for row in changes)
    assert any(
        any((h.get("id") or h.get("hypothesis_id")) == "H4" for h in row.get("added", []))
        for row in changes
    )


def test_probe_api_rejects_invalid_scenario_mode_combination():
    response = client.post(
        "/api/run",
        json={
            "scenario": "open_world_mismatch",
            "hidden_hypothesis": "H4",
            "outcome_mode": "stochastic",
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 400
    assert "outcome_mode" in response.json()["detail"]


def test_probe_session_export_api_returns_replay_schema_and_ledger():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    deadline = time.time() + 8.0
    while time.time() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] == "waiting_for_human":
            assert snap["pending_decision"]["decision_inspector"]
            approved = client.post(
                f"/api/sessions/{session_id}/decision",
                json={"action": "approve"},
            )
            assert approved.status_code == 200
        elif snap["status"] == "completed":
            break
        elif snap["status"] == "failed":
            raise AssertionError(snap["error"])
        time.sleep(0.01)
    else:
        raise AssertionError("session did not complete")

    exported = client.get(f"/api/sessions/{session_id}/export")
    assert exported.status_code == 200
    body = exported.json()
    assert body["schema_version"] == "causalrag.playable_probe.session.v1"
    assert body["status"] == "completed"
    assert body["episode_ledger"]
    assert body["trace"]
    assert all("decision_inspector" in row for row in body["episode_ledger"])
    assert all("candidates" in row and "action_scores" in row for row in body["episode_ledger"])
    assert all("world_before" in row and "world_after" in row for row in body["episode_ledger"])
    assert all("score_provenance" in row for row in body["episode_ledger"])
    assert all("proposer" in row and "proposer_attempts" in row for row in body["episode_ledger"])
    assert body["proposer_traces"]
    assert all(row["kind"] == "deterministic" for row in body["proposer_traces"])
