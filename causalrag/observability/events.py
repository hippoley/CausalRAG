from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ProbeEvent:
    """Stable machine-readable event emitted by the causal runtime.

    The schema is deliberately backend-neutral. The same event can be sent to
    OpenTelemetry, persisted as JSONL, streamed to the Playable Probe UI, or
    replayed during a paper artifact evaluation.
    """

    name: str
    run_id: str
    sequence: int
    step: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)
    schema_version: str = "causalrag.probe.v1"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


class ProbeEventSink(Protocol):
    def emit(self, event: ProbeEvent) -> None:
        ...

    def close(self) -> None:
        ...


class NullEventSink:
    def emit(self, event: ProbeEvent) -> None:
        return None

    def close(self) -> None:
        return None


class InMemoryEventSink:
    def __init__(self) -> None:
        self.events: List[ProbeEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: ProbeEvent) -> None:
        with self._lock:
            self.events.append(event)

    def close(self) -> None:
        return None

    def snapshot(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            selected = self.events if run_id is None else [event for event in self.events if event.run_id == run_id]
            return [event.to_dict() for event in selected]


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: ProbeEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def close(self) -> None:
        return None


class CompositeEventSink:
    def __init__(self, sinks: Iterable[ProbeEventSink]) -> None:
        self.sinks = [sink for sink in sinks if sink is not None]

    def emit(self, event: ProbeEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class ProbeEmitter:
    """Per-run sequence allocator used by CausalAgentLoop."""

    def __init__(self, sink: Optional[ProbeEventSink] = None, run_id: Optional[str] = None) -> None:
        self.sink = sink or NullEventSink()
        self.run_id = run_id or uuid.uuid4().hex
        self.sequence = 0

    def emit(self, name: str, *, step: Optional[int] = None, payload: Optional[Dict[str, Any]] = None) -> ProbeEvent:
        event = ProbeEvent(
            name=str(name),
            run_id=self.run_id,
            sequence=self.sequence,
            step=step,
            payload=_jsonable(payload or {}),
        )
        self.sequence += 1
        self.sink.emit(event)
        return event
