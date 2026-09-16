from causalrag.interface import agent_api


class FakeRunResult:
    def to_dict(self):
        return {"answer": "ok"}


class FakeAgent:
    def run(self, goal, max_steps=8):
        assert goal == "inspect ventilation"
        assert max_steps == 4
        return FakeRunResult()


def test_api_forwards_embedding_configuration(monkeypatch):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(agent_api, "create_agent", fake_create_agent)
    payload = agent_api.AgentRunRequest(
        goal="inspect ventilation",
        max_steps=4,
        documents=["Opening a window increases air exchange."],
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        vector_backend="faiss",
    )

    result = agent_api.run_agent(payload)

    assert result == {"answer": "ok"}
    assert captured["embedding_provider_name"] == "local"
    assert captured["embedding_model"] == "all-MiniLM-L6-v2"
    assert captured["vector_backend"] == "faiss"
