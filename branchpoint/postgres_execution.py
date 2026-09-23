from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional, Tuple

from .execution import (
    EffectIdentityConflict,
    ExecutionBoundaryError,
    ExecutionReceipt,
    _json_result,
    effect_hash,
)


class PostgresExecutionLedger:
    """Distributed durable receipt store backed by PostgreSQL.

    Claims use a primary-key insert with ON CONFLICT DO NOTHING. PostgreSQL
    serializes competing inserts for the same effect_id, so independent
    Branchpoint processes converge on one durable claim without a process-local
    lock.

    Crash-window semantics match SQLiteExecutionLedger: if the external side
    effect happened but completion was not recorded, the receipt remains
    in_flight and future execution fails closed until authoritative
    reconciliation records the real outcome.
    """

    _TABLE = "branchpoint_execution_receipts"

    def __init__(self, dsn: str) -> None:
        self.dsn = str(dsn).strip()
        if not self.dsn:
            raise ValueError("PostgreSQL DSN must be non-empty")
        self._initialize()

    @staticmethod
    def _driver():
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL execution receipts require the optional driver. "
                "Install it with: pip install 'branchpoint[postgres]'"
            ) from exc
        return psycopg, dict_row

    def _connect(self):
        psycopg, dict_row = self._driver()
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    effect_id TEXT PRIMARY KEY,
                    effect_hash TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )

    @staticmethod
    def _receipt(row: Mapping[str, Any]) -> ExecutionReceipt:
        result_json = row.get("result_json")
        result = None if result_json is None else json.loads(result_json)
        return ExecutionReceipt(
            effect_id=str(row["effect_id"]),
            effect_hash=str(row["effect_hash"]),
            tool_name=str(row["tool_name"]),
            status=str(row["status"]),
            result=result,
            error_type=row.get("error_type"),
            error_message=row.get("error_message"),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _select(self, connection, effect_id: str, *, for_update: bool = False):
        suffix = " FOR UPDATE" if for_update else ""
        return connection.execute(
            f"""
            SELECT effect_id, effect_hash, tool_name, status, result_json,
                   error_type, error_message, created_at, updated_at
            FROM {self._TABLE}
            WHERE effect_id = %s{suffix}
            """,
            (str(effect_id),),
        ).fetchone()

    def get(self, effect_id: str) -> Optional[ExecutionReceipt]:
        with self._connect() as connection:
            row = self._select(connection, str(effect_id))
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
        tool_name = str(tool_name)
        digest = effect_hash(tool_name, arguments)
        now = time.time()

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (effect_id, effect_hash, tool_name, status, created_at, updated_at)
                VALUES (%s, %s, %s, 'in_flight', %s, %s)
                ON CONFLICT (effect_id) DO NOTHING
                """,
                (effect_id, digest, tool_name, now, now),
            )
            if cursor.rowcount == 1:
                return (
                    ExecutionReceipt(
                        effect_id=effect_id,
                        effect_hash=digest,
                        tool_name=tool_name,
                        status="in_flight",
                        created_at=now,
                        updated_at=now,
                    ),
                    True,
                )

            row = self._select(connection, effect_id)
            if row is None:
                raise ExecutionBoundaryError(
                    f"PostgreSQL claim conflict produced no receipt for {effect_id!r}"
                )
            receipt = self._receipt(row)
            if receipt.effect_hash != digest or receipt.tool_name != tool_name:
                raise EffectIdentityConflict(
                    f"effect_id {effect_id!r} is already bound to a different effect"
                )
            return receipt, False

    def complete(self, effect_id: str, result: Any) -> ExecutionReceipt:
        effect_id = str(effect_id)
        result_json = _json_result(result)
        now = time.time()

        with self._connect() as connection:
            row = connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET status = 'succeeded',
                    result_json = %s,
                    error_type = NULL,
                    error_message = NULL,
                    updated_at = %s
                WHERE effect_id = %s AND status = 'in_flight'
                RETURNING effect_id, effect_hash, tool_name, status, result_json,
                          error_type, error_message, created_at, updated_at
                """,
                (result_json, now, effect_id),
            ).fetchone()
            if row is not None:
                return self._receipt(row)

            row = self._select(connection, effect_id)
            if row is None:
                raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")
            receipt = self._receipt(row)
            if receipt.status != "succeeded":
                raise ExecutionBoundaryError(
                    f"Cannot complete effect {effect_id!r} "
                    f"from status {receipt.status!r}"
                )
            return receipt

    def fail(self, effect_id: str, exc: BaseException) -> ExecutionReceipt:
        effect_id = str(effect_id)
        now = time.time()

        with self._connect() as connection:
            row = connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET status = 'failed',
                    result_json = NULL,
                    error_type = %s,
                    error_message = %s,
                    updated_at = %s
                WHERE effect_id = %s AND status = 'in_flight'
                RETURNING effect_id, effect_hash, tool_name, status, result_json,
                          error_type, error_message, created_at, updated_at
                """,
                (type(exc).__name__, str(exc)[:2000], now, effect_id),
            ).fetchone()
            if row is not None:
                return self._receipt(row)

            row = self._select(connection, effect_id)
            if row is None:
                raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")
            return self._receipt(row)

    def reconcile(
        self,
        effect_id: str,
        *,
        succeeded: bool,
        result: Any = None,
        note: str = "",
    ) -> ExecutionReceipt:
        effect_id = str(effect_id)

        with self._connect() as connection:
            row = self._select(connection, effect_id, for_update=True)
            if row is None:
                raise ExecutionBoundaryError(f"Unknown effect_id: {effect_id}")

            existing = self._receipt(row)
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
                error_message = str(
                    note
                    or "Authoritative reconciliation marked the effect failed"
                )[:2000]

            updated = connection.execute(
                f"""
                UPDATE {self._TABLE}
                SET status = %s,
                    result_json = %s,
                    error_type = %s,
                    error_message = %s,
                    updated_at = %s
                WHERE effect_id = %s
                RETURNING effect_id, effect_hash, tool_name, status, result_json,
                          error_type, error_message, created_at, updated_at
                """,
                (
                    status,
                    result_json,
                    error_type,
                    error_message,
                    now,
                    effect_id,
                ),
            ).fetchone()
            assert updated is not None
            return self._receipt(updated)
