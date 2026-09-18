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
    assert "CausalRAG Playable Probe" in response.text
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
