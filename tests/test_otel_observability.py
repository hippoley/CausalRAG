from causalrag.agent import ActionKind, CandidateAction
from causalrag.observability import InMemoryEventSink, create_observable_agent
from causalrag.observability.otel import OpenTelemetryEventSink
from causalrag.tools import ToolSpec


class OTelReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(kind=ActionKind.OBSERVE, name="probe", rationale="probe once")]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "ok"})]

    def uncertainty(self, state, world_model):
        return "test uncertainty"

    def hypothesis_proposals(self, state, world_model):
        return []


def test_otel_sink_creates_root_and_tool_child_span():
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "causalrag-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = OpenTelemetryEventSink(provider.get_tracer("causalrag.test"))

    agent = create_observable_agent(
        event_sink=sink,
        reasoner=OTelReasoner(),
        tools=[ToolSpec(name="probe", description="probe", handler=lambda: {"value": 1})],
    )
    result = agent.run("otel smoke", max_steps=3)
    provider.force_flush()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert "causalrag.agent.run" in by_name
    assert "causalrag.tool.call" in by_name

    root = by_name["causalrag.agent.run"]
    tool = by_name["causalrag.tool.call"]
    assert tool.parent.span_id == root.context.span_id
    assert root.attributes["causalrag.run.id"] == result.state.scratch["run_id"]
    root_events = [event.name for event in root.events]
    assert "proposal" in root_events
    assert "decision" in root_events
    assert "observation" in root_events
    assert "run.completed" in root_events
    assert [event.name for event in tool.events] == ["tool.started", "tool.completed"]
