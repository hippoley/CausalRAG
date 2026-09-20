from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


CAUSAL_TRACE_SCHEMA_VERSION = "0.2"
_EVENT_LOGGER = logging.getLogger("branchpoint.causal_event")
_SUBSCRIBER_LOGGER = logging.getLogger("branchpoint.telemetry_subscriber")
_LOCAL_STACK: contextvars.ContextVar[tuple[tuple[str, str], ...]] = contextvars.ContextVar(
    "branchpoint_trace_stack", default=()
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any, *, max_string: int = 2048) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "…"
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, max_string=max_string) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, max_string=max_string) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float, bool)):
        return getattr(value, "value")
    return _jsonable(str(value), max_string=max_string)


def _otel_attribute(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        values = [_otel_attribute(item) for item in value]
        if all(isinstance(item, (bool, int, float, str)) for item in values):
            return values
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class CausalTraceRecord:
    sequence: int
    timestamp: str
    record_type: str
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TraceSubscriber = Callable[[CausalTraceRecord], None]


class TelemetrySpan(AbstractContextManager):
    def __init__(self, telemetry: "CausalTelemetry", name: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        self.telemetry = telemetry
        self.name = str(name)
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self._otel_scope = None
        self._otel_span = None
        self._stack_token = None
        self._started = 0.0
        self.trace_id = ""
        self.span_id = ""
        self.parent_span_id: Optional[str] = None

    def __enter__(self) -> "TelemetrySpan":
        self._started = time.perf_counter()
        stack = _LOCAL_STACK.get()
        if stack:
            self.trace_id = stack[-1][0]
            self.parent_span_id = stack[-1][1]
        else:
            self.trace_id = uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]

        tracer = self.telemetry._tracer
        if tracer is not None:
            otel_attributes = {
                key: converted
                for key, value in self.attributes.items()
                if (converted := _otel_attribute(value)) is not None
            }
            self._otel_scope = tracer.start_as_current_span(self.name, attributes=otel_attributes)
            self._otel_span = self._otel_scope.__enter__()
            context = self._otel_span.get_span_context()
            if getattr(context, "is_valid", False):
                self.trace_id = f"{context.trace_id:032x}"
                self.span_id = f"{context.span_id:016x}"

        self._stack_token = _LOCAL_STACK.set(stack + ((self.trace_id, self.span_id),))
        self.telemetry._record(
            "span.start",
            self.name,
            self.attributes,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
        )
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[str(key)] = _jsonable(value)
        if self._otel_span is not None:
            converted = _otel_attribute(value)
            if converted is not None:
                self._otel_span.set_attribute(str(key), converted)

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = max(0.0, (time.perf_counter() - self._started) * 1000.0)
        end_attributes = dict(self.attributes)
        end_attributes["branchpoint.duration_ms"] = duration_ms
        if exc is not None:
            end_attributes["error.type"] = type(exc).__name__
            end_attributes["branchpoint.outcome"] = "error"
            if self._otel_span is not None:
                try:
                    self._otel_span.record_exception(exc)
                except Exception:
                    pass
        else:
            end_attributes["branchpoint.outcome"] = "ok"

        self.telemetry._record(
            "span.end",
            self.name,
            end_attributes,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
        )
        if self._stack_token is not None:
            _LOCAL_STACK.reset(self._stack_token)
        if self._otel_scope is not None:
            self._otel_scope.__exit__(exc_type, exc, tb)
        return False


class CausalTelemetry:
    """Bounded causal trace recorder with optional OpenTelemetry integration.

    The local machine-readable trace is always available. Subscribers receive
    the exact same records synchronously at creation time, which makes SSE,
    WebSocket, notebook, and test adapters possible without a second event
    schema. Subscriber failures are isolated from the agent runtime.
    """

    def __init__(
        self,
        *,
        enable_otel: bool = False,
        capture_content: bool = False,
        max_records: int = 10000,
        instrumentation_name: str = "branchpoint",
    ) -> None:
        self.capture_content = bool(capture_content)
        self.max_records = max(100, int(max_records))
        self._records: list[CausalTraceRecord] = []
        self._sequence = 0
        self._tracer = None
        self._subscribers: Dict[str, TraceSubscriber] = {}
        self._lock = RLock()
        if enable_otel:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(instrumentation_name, schema_url=None)
            except ImportError as exc:
                raise RuntimeError(
                    "OpenTelemetry support requires the observability extra: "
                    "pip install 'branchpoint[observability]'"
                ) from exc

    @classmethod
    def from_environment(cls, *, capture_content: Optional[bool] = None) -> "CausalTelemetry":
        enable = os.getenv("BRANCHPOINT_OTEL", "").strip().lower() in {"1", "true", "yes", "on"}
        if capture_content is None:
            capture_content = os.getenv("BRANCHPOINT_CAPTURE_CONTENT", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
        return cls(enable_otel=enable, capture_content=bool(capture_content))

    def subscribe(self, callback: TraceSubscriber, *, replay_existing: bool = False) -> str:
        if not callable(callback):
            raise TypeError("callback must be callable")
        subscription_id = uuid.uuid4().hex
        with self._lock:
            self._subscribers[subscription_id] = callback
            existing = list(self._records) if replay_existing else []
        for record in existing:
            self._notify_one(callback, record)
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._subscribers.pop(str(subscription_id), None) is not None

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def span(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> TelemetrySpan:
        return TelemetrySpan(self, name, attributes)

    def event(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> CausalTraceRecord:
        stack = _LOCAL_STACK.get()
        trace_id = stack[-1][0] if stack else uuid.uuid4().hex
        span_id = stack[-1][1] if stack else ""
        parent = stack[-2][1] if len(stack) >= 2 else None
        record = self._record(
            "event",
            str(name),
            dict(attributes or {}),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent,
        )
        _EVENT_LOGGER.info(
            str(name),
            extra={
                "event_name": str(name),
                "branchpoint_attributes": json.dumps(record.attributes, ensure_ascii=False, sort_keys=True),
                "branchpoint_trace_id": record.trace_id,
            },
        )
        return record

    def _notify_one(self, callback: TraceSubscriber, record: CausalTraceRecord) -> None:
        try:
            callback(record)
        except Exception:
            _SUBSCRIBER_LOGGER.exception("Causal telemetry subscriber failed")

    def _notify(self, record: CausalTraceRecord) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
        for callback in subscribers:
            self._notify_one(callback, record)

    def _record(
        self,
        record_type: str,
        name: str,
        attributes: Mapping[str, Any],
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str],
    ) -> CausalTraceRecord:
        with self._lock:
            self._sequence += 1
            attrs = {"branchpoint.schema.version": CAUSAL_TRACE_SCHEMA_VERSION}
            attrs.update({str(k): _jsonable(v) for k, v in attributes.items()})
            record = CausalTraceRecord(
                sequence=self._sequence,
                timestamp=_now_iso(),
                record_type=str(record_type),
                name=str(name),
                trace_id=str(trace_id),
                span_id=str(span_id),
                parent_span_id=parent_span_id,
                attributes=attrs,
            )
            self._records.append(record)
            if len(self._records) > self.max_records:
                del self._records[: len(self._records) - self.max_records]
        self._notify(record)
        return record

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def records(self, since: int = 0) -> list[Dict[str, Any]]:
        start = max(0, int(since))
        with self._lock:
            records = list(self._records[start:])
        return [record.to_dict() for record in records]

    def export_jsonl(self, path: str | Path, *, since: int = 0) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in self.records(since=since):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return destination


def configure_otlp_telemetry(
    *,
    service_name: str = "branchpoint",
    endpoint: Optional[str] = None,
    capture_content: bool = False,
) -> CausalTelemetry:
    try:
        from opentelemetry import _logs, trace
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("OTLP export requires: pip install 'branchpoint[observability]'") from exc

    resource = Resource.create({"service.name": str(service_name)})
    tracer_provider = TracerProvider(resource=resource)
    span_kwargs = {}
    log_kwargs = {}
    if endpoint:
        base = str(endpoint).rstrip("/")
        span_kwargs["endpoint"] = base + "/v1/traces"
        log_kwargs["endpoint"] = base + "/v1/logs"
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**span_kwargs)))
    trace.set_tracer_provider(tracer_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(**log_kwargs)))
    _logs.set_logger_provider(logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    if not any(isinstance(existing, LoggingHandler) for existing in _EVENT_LOGGER.handlers):
        _EVENT_LOGGER.addHandler(handler)
    _EVENT_LOGGER.setLevel(logging.INFO)
    _EVENT_LOGGER.propagate = False

    return CausalTelemetry(enable_otel=True, capture_content=capture_content)


def replay_trace(records: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    normalized = [dict(record) for record in records]
    return sorted(normalized, key=lambda record: (int(record.get("sequence", 0)), str(record.get("timestamp", ""))))
