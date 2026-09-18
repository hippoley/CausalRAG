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
    assert "backend_status" in body


def test_probe_root_is_human_in_the_loop_causal_lab():
    response = client.get("/")
    assert response.status_code == 200
    assert "Human-in-the-loop Causal Lab" in response.text
    assert "Runtime tester capabilities" in response.text
    assert "Start interactive session" in response.text
    assert "CAUSAL RUNTIME TESTER" in response.text
    assert "YOU DECIDE" in response.text
    assert "Operator context for the model proposer" in response.text


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


def test_probe_stepwise_session_api_pauses_before_commit_and_advances_one_step():
    created = client.post(
        "/api/sessions",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
            "user_context": "Airflow fell after maintenance.",
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["preview"]["candidates"]
    assert body["snapshot"]["step"] == 0
    assert body["snapshot"]["observations"] == []
    assert body["snapshot"]["running"] is True
    assert body["snapshot"]["trace_id"]

    session_id = body["session_id"]
    committed = client.post(
        f"/api/sessions/{session_id}/commit",
        json={
            "selection": "runtime",
            "human_note": "approve diagnostic",
        },
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
