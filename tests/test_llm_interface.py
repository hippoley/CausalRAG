from types import SimpleNamespace

from branchpoint.generator.llm_interface import LLMInterface


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text='{"ok": true}')


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="legacy answer")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def _interface(model):
    interface = LLMInterface.__new__(LLMInterface)
    interface.model = model
    interface.provider = "openai"
    interface.system_message = "system"
    interface.client = FakeClient()
    return interface


def test_gpt56_uses_responses_api():
    interface = _interface("gpt-5.6-terra")

    result = interface.generate("reason about this", json_mode=True)

    assert result == '{"ok": true}'
    assert len(interface.client.responses.calls) == 1
    assert interface.client.responses.calls[0]["model"] == "gpt-5.6-terra"
    assert "Return only valid JSON" in interface.client.responses.calls[0]["input"]
    assert interface.client.chat.completions.calls == []


def test_legacy_model_keeps_chat_completions_path():
    interface = _interface("gpt-4o-mini")

    result = interface.generate("legacy prompt")

    assert result == "legacy answer"
    assert len(interface.client.chat.completions.calls) == 1
    assert interface.client.responses.calls == []
