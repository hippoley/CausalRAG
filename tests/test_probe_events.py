from causalrag.agent import ActionKind, CandidateAction
from causalrag.observability import InMemoryEventSink, create_observable_agent
from causalrag.tools import ToolSpec


class ProbeReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_sensor",
                    arguments={},
                    expected_goal_gain=0.1,
                    rationale="collect one real observation",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "done"},
                rationale="observation is sufficient",
            )
        ]

    def uncertainty(self, state, world_model):
        return "sensor state"

    def hypothesis_proposals(self, state, world_model):
        return []


def test_observable_agent_emits_stable_replayable_event_order_without_otel():
    sink = InMemoryEventSink()
    agent = create_observable_agent(
        event_sink=sink,
        reasoner=ProbeReasoner(),
        tools=[
            ToolSpec(
                name="read_sensor",
                description="read deterministic sensor",
                handler=lambda: {"temperature": 23.0},
                metadata={"kind": "observe"},
            )
        ],
    )

    result = agent.run("inspect room state", max_steps=3)
    events = sink.snapshot(result.state.scratch["run_id"])
    names = [event["name"] for event in events]

    assert names == [
        "run.started",
        "proposal",
        "decision",
        "tool.started",
        "tool.completed",
        "observation",
        "proposal",
        "decision",
        "run.completed",
    ]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert all(event["schema_version"] == "causalrag.probe.v1" for event in events)
    assert events[3]["payload"]["action_name"] == "read_sensor"
    assert events[4]["payload"]["result"] == {"temperature": 23.0}
    assert events[-1]["payload"]["answer"] == "done"


def test_event_stream_keeps_decision_scores_machine_readable():
    sink = InMemoryEventSink()
    agent = create_observable_agent(
        event_sink=sink,
        reasoner=ProbeReasoner(),
        tools=[ToolSpec(name="read_sensor", description="sensor", handler=lambda: {"temperature": 23.0})],
    )
    result = agent.run("inspect", max_steps=3)
    decision = next(event for event in sink.snapshot() if event["name"] == "decision")
    assert isinstance(decision["payload"]["action_scores"], list)
    assert decision["run_id"] == result.state.scratch["run_id"]
