import json

from causalrag import ActionKind, CandidateAction, CausalTelemetry, ToolSpec, create_agent, replay_trace


class ProbeReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="sense",
                    tests_hypotheses=[],
                    rationale="Collect one observation.",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "done"},
                rationale="Enough evidence.",
            )
        ]

    def uncertainty(self, state, world_model):
        return "whether the sensor is active"

    def hypothesis_proposals(self, state, world_model):
        return []


def _agent(telemetry):
    return create_agent(
        reasoner=ProbeReasoner(),
        telemetry=telemetry,
        tools=[
            ToolSpec(
                name="sense",
                description="read a test sensor",
                handler=lambda: {"value": 7, "secret": "do-not-export"},
                metadata={"kind": "observe"},
            )
        ],
    )


def test_agent_emits_one_correlated_machine_readable_causal_trace_without_content_by_default(tmp_path):
    telemetry = CausalTelemetry(capture_content=False)
    agent = _agent(telemetry)

    result = agent.run("inspect a private sensor", max_steps=3)
    payload = result.to_dict()
    trace = payload["causal_trace"]

    assert payload["trace_id"]
    assert trace
    assert {record["trace_id"] for record in trace} == {payload["trace_id"]}

    names = [record["name"] for record in trace]
    assert "invoke_agent causalrag" in names
    assert "execute_tool sense" in names
    assert "causalrag.decision" in names
    assert "causalrag.observation" in names
    assert "causalrag.run.completed" in names

    root_start = next(
        record
        for record in trace
        if record["record_type"] == "span.start" and record["name"] == "invoke_agent causalrag"
    )
    tool_start = next(
        record
        for record in trace
        if record["record_type"] == "span.start" and record["name"] == "execute_tool sense"
    )
    assert tool_start["parent_span_id"] == root_start["span_id"]

    serialized = json.dumps(trace, ensure_ascii=False)
    assert "do-not-export" not in serialized
    assert "inspect a private sensor" not in serialized
    assert "argument_names" in serialized

    path = agent.export_last_trace(tmp_path / "trace.jsonl")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == trace
    assert replay_trace(reversed(lines)) == lines


def test_content_capture_is_explicit_opt_in():
    telemetry = CausalTelemetry(capture_content=True)
    result = _agent(telemetry).run("inspect a private sensor", max_steps=3)
    serialized = json.dumps(result.to_dict()["causal_trace"], ensure_ascii=False)
    assert "do-not-export" in serialized
    assert "inspect a private sensor" in serialized


def test_trace_context_is_released_after_run():
    telemetry = CausalTelemetry()
    result = _agent(telemetry).run("sensor check", max_steps=3)
    completed_trace_id = result.to_dict()["trace_id"]

    outside = telemetry.event("outside-run", {"kind": "independent"})
    assert outside.trace_id != completed_trace_id
    assert outside.span_id == ""


def test_nested_local_spans_restore_parent_context_exactly_once():
    telemetry = CausalTelemetry()
    with telemetry.span("root") as root:
        with telemetry.span("child") as child:
            event = telemetry.event("inside-child")
            assert child.trace_id == root.trace_id
            assert child.parent_span_id == root.span_id
            assert event.trace_id == root.trace_id
            assert event.span_id == child.span_id
        after_child = telemetry.event("after-child")
        assert after_child.span_id == root.span_id

    outside = telemetry.event("after-root")
    assert outside.trace_id != root.trace_id
