from branchpoint.agent import CausalAgentLoop
from branchpoint.agent.actions import ActionKind, CandidateAction
from branchpoint.agent.loop import DecisionGateReplan
from branchpoint.reasoning.llm import LLMCausalReasoner
from branchpoint.observability import CausalTelemetry
from branchpoint.tools.base import ToolRegistry, ToolSpec
from branchpoint.world_model import CausalWorldModel


class FakeLLM:
    provider = "openai"
    model = "fake-frontier"

    def __init__(self):
        self.last_usage = {}

    def generate(self, prompt, temperature=0.1, max_tokens=1800, json_mode=True):
        self.last_usage = {"input_tokens": 321, "output_tokens": 87}
        return {
            "uncertainty": "which mechanism is active",
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "filter is blocked",
                    "probability": 0.5,
                    "rationale": "still plausible",
                    "falsifiers": ["normal filter pressure"],
                },
                {
                    "id": "H2",
                    "statement": "fan is weak",
                    "probability": 0.5,
                    "rationale": "still plausible",
                    "falsifiers": ["normal fan rpm"],
                },
            ],
            "candidates": [
                {
                    "kind": "observe",
                    "name": "measure_filter",
                    "arguments": {},
                    "expected_goal_gain": 0.1,
                    "expected_information_gain": 0.8,
                    "tests_hypotheses": ["H1", "H2"],
                    "falsification_target": "H1",
                    "rationale": "discriminates H1 from H2",
                }
            ],
        }


def test_llm_reasoner_exposes_only_formal_structured_submission_metadata():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure_filter",
                description="measure filter pressure",
                handler=lambda: {"outcome": "normal"},
                metadata={"kind": "observe"},
            )
        ]
    )
    llm = FakeLLM()
    reasoner = LLMCausalReasoner(llm=llm, tools=registry)

    from branchpoint.agent.state import AgentState

    world = CausalWorldModel()
    state = AgentState(goal="diagnose", max_steps=1)
    candidates = list(reasoner.propose(state, world))
    metadata = reasoner.proposal_metadata()

    assert candidates[0].name == "measure_filter"
    assert metadata["kind"] == "llm"
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "fake-frontier"
    assert metadata["usage"] == {"input_tokens": 321, "output_tokens": 87}
    assert metadata["structured_payload"]["uncertainty"] == "which mechanism is active"
    assert metadata["structured_payload"]["candidates"][0]["name"] == "measure_filter"
    assert "chain_of_thought" not in metadata


def test_agent_loop_records_provider_model_latency_tokens_and_submission_event():
    telemetry = CausalTelemetry()
    registry = ToolRegistry(
        [
            ToolSpec(
                name="measure_filter",
                description="measure filter pressure",
                handler=lambda: {"outcome": "normal"},
                metadata={"kind": "observe"},
            )
        ],
        telemetry=telemetry,
    )
    reasoner = LLMCausalReasoner(llm=FakeLLM(), tools=registry)
    loop = CausalAgentLoop(
        reasoner=reasoner,
        tools=registry,
        world_model=CausalWorldModel(),
    )

    state = loop.run("diagnose", max_steps=1)
    traces = state.scratch["proposer_traces"]

    assert len(traces) == 1
    trace = traces[0]
    assert trace["step"] == 0
    assert trace["attempt"] == 1
    assert trace["kind"] == "llm"
    assert trace["provider"] == "openai"
    assert trace["model"] == "fake-frontier"
    assert trace["usage"]["input_tokens"] == 321
    assert trace["duration_ms"] >= 0.0
    assert trace["candidates"][0]["name"] == "measure_filter"
    assert trace["hypothesis_proposals"][0]["id"] == "H1"

    events = list(registry.telemetry.records())
    submitted = [row for row in events if row["name"] == "branchpoint.proposer.submitted"]
    assert submitted
    assert submitted[0]["attributes"]["branchpoint.proposer.model"] == "fake-frontier"


class OneActionReasoner:
    def propose(self, state, world_model):
        return [
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="ping",
                rationale="one candidate",
            )
        ]

    def uncertainty(self, state, world_model):
        return "test uncertainty"


def test_replan_preserves_multiple_proposer_attempts_for_same_step():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="ping",
                description="ping",
                handler=lambda: {"ok": True},
                metadata={"kind": "observe"},
            )
        ]
    )
    calls = {"count": 0}

    def gate(state, world_model, decision):
        calls["count"] += 1
        if calls["count"] == 1:
            raise DecisionGateReplan()
        return None

    loop = CausalAgentLoop(
        reasoner=OneActionReasoner(),
        tools=registry,
        decision_gate=gate,
    )
    state = loop.run("test replanning", max_steps=1)

    traces = state.scratch["proposer_traces"]
    assert [(row["step"], row["attempt"]) for row in traces] == [(0, 1), (0, 2)]
    assert all(row["kind"] == "deterministic" for row in traces)
    assert state.observations[0].action_name == "ping"


class FailingLLM:
    provider = "openai"
    model = "broken-model"
    last_usage = {}

    def generate(self, prompt, temperature=0.1, max_tokens=1800, json_mode=True):
        return "Error generating response: provider unavailable"


def test_failed_model_call_is_explicit_in_proposer_metadata():
    registry = ToolRegistry()
    reasoner = LLMCausalReasoner(llm=FailingLLM(), tools=registry)

    from branchpoint.agent.state import AgentState

    state = AgentState(goal="diagnose", max_steps=1)
    candidates = list(reasoner.propose(state, CausalWorldModel()))
    metadata = reasoner.proposal_metadata()

    assert metadata["ok"] is False
    assert "provider unavailable" in metadata["error"]
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "broken-model"
    assert metadata["structured_payload"] == {}
    assert candidates[0].kind == ActionKind.STOP
