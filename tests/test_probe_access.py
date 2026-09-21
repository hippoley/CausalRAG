import types

from fastapi.testclient import TestClient

from branchpoint.interface import probe_api


def _client(monkeypatch):
    monkeypatch.setenv("BRANCHPOINT_REQUIRE_PROBE_AUTH", "true")
    monkeypatch.setenv("BRANCHPOINT_PROBE_ACCESS_TOKEN", "owner-secret")
    monkeypatch.setenv("BRANCHPOINT_PROBE_COOKIE_SECURE", "false")
    return TestClient(probe_api.app)


def test_auth_mode_keeps_deterministic_probe_public(monkeypatch):
    client = _client(monkeypatch)

    config = client.get("/api/config")
    assert config.status_code == 200
    assert config.json()["access"] == {
        "required_for_external_models": True,
        "configured": True,
        "authenticated": False,
    }

    response = client.post(
        "/api/run",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "proposer_family": "deterministic",
        },
    )
    assert response.status_code == 200
    assert response.json()["metrics"]["success"] is True


def test_external_model_use_requires_owner_access(monkeypatch):
    client = _client(monkeypatch)

    external = client.post(
        "/api/run",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "proposer_family": "frontier",
            "provider": "openai",
            "model": "test-model",
        },
    )
    assert external.status_code == 401
    assert "Owner access" in external.json()["detail"]

    model_test = client.post(
        "/api/model/test",
        json={"provider": "local", "model": "test-model"},
    )
    assert model_test.status_code == 401


def test_owner_unlock_uses_http_only_derived_cookie_and_allows_model_test(monkeypatch):
    client = _client(monkeypatch)

    wrong = client.post("/api/access", json={"token": "wrong"})
    assert wrong.status_code == 401

    unlocked = client.post("/api/access", json={"token": "owner-secret"})
    assert unlocked.status_code == 200
    set_cookie = unlocked.headers["set-cookie"]
    assert "causalrag_probe_access=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=strict" in set_cookie
    assert "owner-secret" not in set_cookie

    status = client.get("/api/access/status")
    assert status.status_code == 200
    assert status.json()["authenticated"] is True

    class FakeLLM:
        def __init__(self, model, provider):
            self.model = model
            self.provider = provider

        def generate(self, prompt, temperature=0.0, max_tokens=32):
            return "BRANCHPOINT_MODEL_OK"

    monkeypatch.setattr(probe_api, "LLMInterface", FakeLLM)
    model_test = client.post(
        "/api/model/test",
        json={"provider": "local", "model": "test-model"},
    )
    assert model_test.status_code == 200
    assert model_test.json()["ok"] is True

    logged_out = client.post("/api/access/logout")
    assert logged_out.status_code == 200
    assert client.get("/api/access/status").json()["authenticated"] is False


def test_bearer_token_allows_scripted_owner_access(monkeypatch):
    client = _client(monkeypatch)

    class FakeLLM:
        def __init__(self, model, provider):
            self.model = model
            self.provider = provider

        def generate(self, prompt, temperature=0.0, max_tokens=32):
            return "BRANCHPOINT_MODEL_OK"

    monkeypatch.setattr(probe_api, "LLMInterface", FakeLLM)
    response = client.post(
        "/api/model/test",
        headers={"Authorization": "Bearer owner-secret"},
        json={"provider": "local", "model": "script-model"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_required_auth_without_configured_token_fails_closed(monkeypatch):
    monkeypatch.setenv("BRANCHPOINT_REQUIRE_PROBE_AUTH", "true")
    monkeypatch.delenv("BRANCHPOINT_PROBE_ACCESS_TOKEN", raising=False)
    client = TestClient(probe_api.app)

    status = client.get("/api/access/status").json()
    assert status["required_for_external_models"] is True
    assert status["configured"] is False
    assert status["authenticated"] is False

    response = client.post(
        "/api/model/test",
        json={"provider": "local", "model": "test-model"},
    )
    assert response.status_code == 503


def test_external_session_state_is_not_readable_without_owner_access(monkeypatch):
    client = _client(monkeypatch)

    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(proposer_family="frontier"),
        snapshot=lambda: {"session_id": "protected-test", "status": "waiting_for_human"},
    )
    with probe_api.SESSION_MANAGER._lock:
        probe_api.SESSION_MANAGER._sessions["protected-test"] = fake
    try:
        blocked = client.get("/api/sessions/protected-test")
        assert blocked.status_code == 401

        unlocked = client.post("/api/access", json={"token": "owner-secret"})
        assert unlocked.status_code == 200

        allowed = client.get("/api/sessions/protected-test")
        assert allowed.status_code == 200
        assert allowed.json()["session_id"] == "protected-test"
    finally:
        with probe_api.SESSION_MANAGER._lock:
            probe_api.SESSION_MANAGER._sessions.pop("protected-test", None)
