from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from .events import ProbeEvent


def _otel_value(value: Any):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
        return list(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _attributes(event: ProbeEvent) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "causalrag.probe.schema_version": event.schema_version,
        "causalrag.run.id": event.run_id,
        "causalrag.event.sequence": int(event.sequence),
        "causalrag.event.name": event.name,
    }
    if event.step is not None:
        attrs["causalrag.agent.step"] = int(event.step)
    for key, value in event.payload.items():
        if value is None:
            continue
        attrs[f"causalrag.{key}"] = _otel_value(value)
    return attrs


class OpenTelemetryEventSink:
    """Map the backend-neutral ProbeEvent stream onto OpenTelemetry traces.

    ``agent.run`` is the root span. Runtime state changes become span events.
    Operations with explicit start/completion boundaries (tool calls and waits)
    become child spans so latency and failures remain measurable.
    """

    def __init__(self, tracer) -> None:
        self.tracer = tracer
        self._runs: Dict[str, Any] = {}
        self._operations: Dict[Tuple[str, str], Any] = {}

    def _root(self, event: ProbeEvent):
        return self._runs.get(event.run_id)

    def _context_for(self, span):
        from opentelemetry import trace

        return trace.set_span_in_context(span) if span is not None else None

    def emit(self, event: ProbeEvent) -> None:
        attrs = _attributes(event)

        if event.name == "run.started":
            span = self.tracer.start_span("causalrag.agent.run", attributes=attrs)
            self._runs[event.run_id] = span
            span.add_event(event.name, attributes=attrs)
            return

        root = self._root(event)
        if root is None:
            # Accept partial/replayed streams without creating invalid parents.
            root = self.tracer.start_span("causalrag.agent.run", attributes={"causalrag.run.id": event.run_id})
            self._runs[event.run_id] = root

        operation_key = str(event.payload.get("operation_id") or event.payload.get("action_name") or event.payload.get("name") or event.step or event.sequence)

        if event.name in {"tool.started", "wait.started"}:
            span_name = "causalrag.tool.call" if event.name == "tool.started" else "causalrag.agent.wait"
            child = self.tracer.start_span(
                span_name,
                context=self._context_for(root),
                attributes=attrs,
            )
            child.add_event(event.name, attributes=attrs)
            self._operations[(event.run_id, operation_key)] = child
            return

        if event.name in {"tool.completed", "tool.failed", "wait.completed"}:
            child = self._operations.pop((event.run_id, operation_key), None)
            if child is not None:
                child.add_event(event.name, attributes=attrs)
                if event.name == "tool.failed":
                    from opentelemetry.trace import Status, StatusCode

                    child.set_status(Status(StatusCode.ERROR, str(event.payload.get("error") or "tool failed")))
                child.end()
            else:
                root.add_event(event.name, attributes=attrs)
            return

        root.add_event(event.name, attributes=attrs)
        if event.name == "run.completed":
            root.set_attributes(attrs)
            root.end()
            self._runs.pop(event.run_id, None)

    def close(self) -> None:
        for span in list(self._operations.values()):
            span.end()
        self._operations.clear()
        for span in list(self._runs.values()):
            span.end()
        self._runs.clear()


def create_otel_sink(
    *,
    service_name: str = "causalrag",
    endpoint: Optional[str] = None,
    console: bool = False,
    tracer_provider=None,
) -> OpenTelemetryEventSink:
    """Create an OTel sink using OTLP when ``endpoint`` is supplied.

    A caller may inject a configured provider. Otherwise this function creates
    a provider with ``service.name`` and attaches a BatchSpanProcessor for OTLP
    or a SimpleSpanProcessor for console debugging.
    """

    from opentelemetry import trace

    if tracer_provider is None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )

        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        elif console:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

            tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        # Avoid replacing an already configured global provider. The returned
        # provider is still used directly by this sink either way.
        try:
            trace.set_tracer_provider(tracer_provider)
        except Exception:
            pass

    tracer = tracer_provider.get_tracer("causalrag.observability")
    return OpenTelemetryEventSink(tracer)
