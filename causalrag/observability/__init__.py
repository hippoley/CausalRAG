from .events import (
    CompositeEventSink,
    InMemoryEventSink,
    JsonlEventSink,
    NullEventSink,
    ProbeEvent,
    ProbeEventSink,
)

__all__ = [
    "ProbeEvent",
    "ProbeEventSink",
    "NullEventSink",
    "InMemoryEventSink",
    "JsonlEventSink",
    "CompositeEventSink",
]


def __getattr__(name):
    if name in {"OpenTelemetryEventSink", "create_otel_sink"}:
        try:
            from .otel import OpenTelemetryEventSink, create_otel_sink
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "OpenTelemetry support is optional. Install it with: "
                "pip install 'causalrag[observability]'"
            ) from exc
        return {
            "OpenTelemetryEventSink": OpenTelemetryEventSink,
            "create_otel_sink": create_otel_sink,
        }[name]
    raise AttributeError(name)
