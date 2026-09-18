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
    assert "Start interactive session" in response.text\n    assert "CAUSAL RUNTIME TESTER" in response.text\n    assert "YOU DECIDE" in response.text


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



def test_probe_stepwise_session_api_does_not_execute_before_commit():
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
    body = created.json()
    assert body["preview"]["candidates"]
    assert body["snapshot"]["step"] == 0
    assert body["snapshot"]["observations"] == []

    session_id = body["session_id"]
    committed = client.post(
        f"/api/sessions/{session_id}/commit",
        json={"selection": "runtime", "human_note": "approve diagnostic"},
    )
    assert committed.status_code == 200
    after = committed.json()["snapshot"]
    assert after["step"] == 1
    assert len(after["observations"]) == 1
    assert after["human_events"][0]["human_note"] == "approve diagnostic"

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 200


def test_probe_config_exposes_backend_readiness_without_secrets():
    config = client.get("/api/config").json()
    assert "backend_status" in config
    serialized = str(config["backend_status"]).lower()
    assert "api_key" not in serialized
