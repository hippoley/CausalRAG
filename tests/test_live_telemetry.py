from causalrag.observability import CausalTelemetry


def test_live_subscriber_receives_records_in_creation_order():
    telemetry = CausalTelemetry()
    received = []
    subscription = telemetry.subscribe(received.append)

    with telemetry.span("root"):
        telemetry.event("first", {"value": 1})
        telemetry.event("second", {"value": 2})

    assert [record.sequence for record in received] == sorted(
        record.sequence for record in received
    )
    assert [record.name for record in received] == ["root", "first", "second", "root"]
    assert telemetry.unsubscribe(subscription) is True
    assert telemetry.unsubscribe(subscription) is False


def test_replay_existing_then_continue_live_without_duplicate_registration():
    telemetry = CausalTelemetry()
    telemetry.event("before")
    received = []

    telemetry.subscribe(received.append, replay_existing=True)
    telemetry.event("after")

    assert [record.name for record in received] == ["before", "after"]
    assert telemetry.subscriber_count() == 1


def test_broken_subscriber_cannot_break_agent_telemetry():
    telemetry = CausalTelemetry()
    healthy = []

    def broken(_record):
        raise RuntimeError("consumer disconnected")

    telemetry.subscribe(broken)
    telemetry.subscribe(healthy.append)
    record = telemetry.event("runtime-still-alive", {"ok": True})

    assert record.name == "runtime-still-alive"
    assert [item.name for item in healthy] == ["runtime-still-alive"]
    assert telemetry.records()[-1]["name"] == "runtime-still-alive"
