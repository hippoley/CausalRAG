from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable


class ExecutionBoundaryError(RuntimeError):
    """Base error for durable execution-boundary failures."""


class EffectIdentityConflict(ExecutionBoundaryError):
    """The same effect id was presented with different semantic content."""


class ExecutionInProgress(ExecutionBoundaryError):
    """An effect is already in flight or its crash outcome is still unknown."""


class PreviousExecutionFailed(ExecutionBoundaryError):
    """An effect previously failed and is not automatically retried."""


class NonCanonicalEffect(ExecutionBoundaryError):
    """The effect cannot be represented by the strict canonical envelope."""


@runtime_checkable
class ExecutionLedger(Protocol):
    """Storage contract for durable execution receipts."""

    def get(self, effect_id: str) -> Optional["ExecutionReceipt"]:
        ...

    def claim(
        self,
        effect_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Tuple["ExecutionReceipt", bool]:
        ...

    def complete(self, effect_id: str, result: Any) -> "ExecutionReceipt":
        ...

    def fail(
        self,
        effect_id: str,
        exc: BaseException,
    ) -> "ExecutionReceipt":
        ...

    def reconcile(
        self,
        effect_id: str,
        *,
        succeeded: bool,
        result: Any = None,
        note: str = "",
    ) -> "ExecutionReceipt":
        ...


@dataclass(frozen=True)
class ExecutionReceipt:
    effect_id: str
    effect_hash: str
    tool_name: str
    status: str
    result: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonCanonicalEffect("NaN and Infinity are not valid effect values.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NonCanonicalEffect("Effect object keys must be strings.")
            normalized[unicodedata.normalize("NFC", key)] = _normalize(item)
        return normalized
    raise NonCanonicalEffect(
        f"Unsupported effect value type: {type(value).__name__}. "
        "Use JSON-compatible values at the execution boundary."
    )


def canonical_effect(tool_name: str, arguments: Mapping[str, Any]) -> bytes:
    payload = {
        "tool": unicodedata.normalize("NFC", str(tool_name)),
        "arguments": _normalize(dict(arguments)),
    }
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NonCanonicalEffect(f"Effect cannot be canonicalized: {exc}") from exc
    return text.encode("utf-8")


def effect_hash(tool_name: str, arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_effect(tool_name, arguments)).hexdigest()


def _json_result(value: Any) -> str:
    try:
        return json.dumps(
            _normalize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, NonCanonicalEffect) as exc:
        raise ExecutionBoundaryError(
            "Durable execution requires a JSON-compatible tool result so a replay "
            "can return the original observation without executing the side effect again."
        ) from exc


class SQLiteExecutionLedger:
    """Durable fail-closed receipt store for effectful tool execution.

    A completed effect id is replayed from its stored observation instead of
    re-executing the handler. The same id cannot be rebound to another effect.

    This does not claim exactly-once delivery across an external side-effect
    crash window. If a process dies after the external effect happened but
    before completion was recorded, the receipt stays in_flight and future
    execution fails closed until an authoritative reconciliation records
    the real outcome.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).expanduser())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS branchpoint_execution_receipts (
                    effect_id TEXT PRIMARY KEY,
                    effect_hash TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _receipt(row) -> ExecutionReceipt:
        result = None if row["result_json"] is None else json.loads(row["result_json"])
        return ExecutionReceipt(
            effect_id=row["effect_id"],
            effect_hash=row["effect_hash"],
            tool_name=row["tool_name"],
            status=row["status"],
            result=result,
            error_type=row["error_type"],
            error_message=row["error_message"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def get(self, effect_id: str) -> Optional[ExecutionReceipt]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM branchpoint_execution_receipts WHERE effect_id = ?",
                (str(effect_id),),
            ).fetchone()
        return None if row is None else self._receipt(row)

    def claim(
        self,
        effect_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Tuple[ExecutionReceipt, bool]:
        effect_id = str(effect_id).strip()
        if not effect_id:
            raise ValueError("effect_id must be non-empty")
        digest = effect_hash(tool_name, arguments)
        now = time.time()

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM branchpoint_execution_receipts WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO branchpoint_execution_receipts
                    (effect_id, effect_hash, tool_name, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'in_flight', ?, ?)
                    """,
                    (effect_id, digest, str(tool_name), now, now),
                )
                connection.commit()
                return (
                    ExecutionReceipt(
                        effect_id=effect_id,
                        effect_hash=digest,
                        tool_name=str(tool_name),
                        status="in_flight",
                        created_at=now,
                        updated_at=now,
                    ),
                    True,
                )

            receipt = self._receipt(row)
            if receipt.effect_hash != digest or receipt.tool_name != str(tool_name):
                connection.rollback()
                raise EffectIdentityConflict(
                    f"effect_id {effect_id!r} is already bound to a different effect"
                )
            connection.commit()
            return receipt, False

    def complete(self, effect_id: str, result: Any) -> ExecutionReceipt:
        result_json = _json_result(result)
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE branchpoint_execution_receipts
                SET status = 'succeeded', result_json = ?, error_type = NULL,
                    error_message = NULL, updated_at = ?
                WHERE effect_id = ? AND status = 'in_flight'
                """,
                (result_json, now, str(effect_id)),
            )
            if cursor.rowcount != 1:
                connection.rollback()
            else:
                connection.commit()
        receipt = self.get(effect_id)
        if receipt is None:
            raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")
        if receipt.status != "succeeded":
            raise ExecutionBoundaryError(
                f"Cannot complete effect {effect_id!r} from status {receipt.status!r}"
            )
        return receipt

    def fail(self, effect_id: str, exc: BaseException) -> ExecutionReceipt:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE branchpoint_execution_receipts
                SET status = 'failed', error_type = ?, error_message = ?, updated_at = ?
                WHERE effect_id = ? AND status = 'in_flight'
                """,
                (type(exc).__name__, str(exc)[:2000], now, str(effect_id)),
            )
            connection.commit()
        receipt = self.get(effect_id)
        if receipt is None:
            raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")
        return receipt

    def reconcile(
        self,
        effect_id: str,
        *,
        succeeded: bool,
        result: Any = None,
        note: str = "",
    ) -> ExecutionReceipt:
        """Record an authoritative post-crash outcome without re-executing the tool."""
        existing = self.get(effect_id)
        if existing is None:
            raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")
        if existing.status == "succeeded":
            return existing

        now = time.time()
        if succeeded:
            result_json = _json_result(result)
            status = "succeeded"
            error_type = None
            error_message = None
        else:
            result_json = None
            status = "failed"
            error_type = "ReconciledFailure"
            error_message = str(note or "Authoritative reconciliation marked the effect failed")[:2000]

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE branchpoint_execution_receipts
                SET status = ?, result_json = ?, error_type = ?, error_message = ?, updated_at = ?
                WHERE effect_id = ?
                """,
                (status, result_json, error_type, error_message, now, str(effect_id)),
            )
            connection.commit()
        receipt = self.get(effect_id)
        assert receipt is not None
        return receipt
