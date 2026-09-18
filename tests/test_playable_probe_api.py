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


def test_probe_root_is_real_research_console():
    response = client.get("/")
    assert response.status_code == 200
    assert "CausalRAG · Playable Causal Probe" in response.text
    assert "Runtime capabilities" in response.text
    assert "Start interactive episode" in response.text
    assert "MODEL PROPOSAL" in response.text
    assert "YOUR MOVE" in response.text


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
