from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default(value: Any):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class SQLiteSessionArchive:
    """Durable archive for Playable Probe session snapshots.

    The archive is deliberately read-only recovery state: it preserves the
    latest session artifact across process restarts but does not claim that a
    live Python thread or an already-started external side effect can be resumed
    automatically.
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
                CREATE TABLE IF NOT EXISTS branchpoint_probe_sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, payload: Dict[str, Any]) -> None:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session payload requires session_id")
        status = str(payload.get("status") or "unknown")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_default,
        )
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO branchpoint_probe_sessions
                    (session_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, status, encoded, now),
            )
            connection.commit()

    def get(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM branchpoint_probe_sessions WHERE session_id = ?",
                (str(session_id),),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return json.loads(row["payload_json"])

    def list(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        size = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, status, updated_at
                FROM branchpoint_probe_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (size,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "status": row["status"],
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]
