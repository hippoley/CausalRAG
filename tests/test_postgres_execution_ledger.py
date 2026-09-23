from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from branchpoint import (
    EffectIdentityConflict,
    ExecutionInProgress,
    ExecutionLedger,
    PostgresExecutionLedger,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)


psycopg = pytest.importorskip("psycopg")
DSN = os.environ.get("BRANCHPOINT_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not DSN,
    reason="BRANCHPOINT_TEST_POSTGRES_DSN is required",
)


def _ledger():
    return PostgresExecutionLedger(DSN)


def test_postgres_and_sqlite_satisfy_execution_ledger_contract(tmp_path):
    assert isinstance(SQLiteExecutionLedger(tmp_path / "effects.sqlite3"), ExecutionLedger)
    assert isinstance(_ledger(), ExecutionLedger)


def test_two_independent_postgres_ledgers_converge_on_one_claim():
    effect_id = "concurrent-claim-1"
    first = _ledger()
    second = _ledger()
    barrier = threading.Barrier(2)

    def claim(ledger):
        barrier.wait()
        receipt, claimed = ledger.claim(
            effect_id,
            "charge",
            {"amount": 25, "currency": "USD"},
        )
        return receipt, claimed

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, first), pool.submit(claim, second)]
        rows = [future.result(timeout=15) for future in futures]

    assert sorted(claimed for _receipt, claimed in rows) == [False, True]
    receipts = [receipt for receipt, _claimed in rows]
    assert {receipt.status for receipt in receipts} == {"in_flight"}
    assert len({receipt.effect_hash for receipt in receipts}) == 1
    assert len({receipt.tool_name for receipt in receipts}) == 1


def test_completed_effect_replays_across_independent_process_boundaries():
    calls = []

    def charge(amount):
        calls.append(amount)
        return {"charge_id": "ch_postgres_1", "amount": amount}

    effect_id = "distributed-replay-1"
    first_registry = ToolRegistry(
        [ToolSpec("charge", "charge", charge)],
        execution_ledger=_ledger(),
    )
    second_registry = ToolRegistry(
        [ToolSpec("charge", "charge", charge)],
        execution_ledger=_ledger(),
    )

    first = first_registry.execute(
        "charge",
        {"amount": 25},
        effect_id=effect_id,
    )
    second = second_registry.execute(
        "charge",
        {"amount": 25},
        effect_id=effect_id,
    )

    assert first == second == {"charge_id": "ch_postgres_1", "amount": 25}
    assert calls == [25]
    assert _ledger().get(effect_id).status == "succeeded"


def test_postgres_effect_id_cannot_be_rebound():
    ledger = _ledger()
    ledger.claim("postgres-conflict-1", "charge", {"amount": 10})

    with pytest.raises(EffectIdentityConflict):
        _ledger().claim("postgres-conflict-1", "charge", {"amount": 11})


def test_postgres_in_flight_crash_window_fails_closed_until_reconciled():
    effect_id = "postgres-crash-window-1"
    _ledger().claim(effect_id, "door_unlock", {"door": "front"})

    calls = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "door_unlock",
                "unlock",
                lambda **kwargs: calls.append(kwargs),
            )
        ],
        execution_ledger=_ledger(),
    )

    with pytest.raises(ExecutionInProgress):
        registry.execute(
            "door_unlock",
            {"door": "front"},
            effect_id=effect_id,
        )
    assert calls == []

    reconciled = _ledger().reconcile(
        effect_id,
        succeeded=True,
        result={"door": "front", "state": "unlocked"},
    )
    assert reconciled.status == "succeeded"

    replay = registry.execute(
        "door_unlock",
        {"door": "front"},
        effect_id=effect_id,
    )
    assert replay == {"door": "front", "state": "unlocked"}
    assert calls == []


def test_postgres_failure_receipt_is_durable_across_connections():
    effect_id = "postgres-failure-1"
    ledger = _ledger()
    ledger.claim(effect_id, "deploy", {"service": "api"})
    failed = ledger.fail(effect_id, RuntimeError("provider rejected deployment"))

    assert failed.status == "failed"
    assert failed.error_type == "RuntimeError"
    assert "provider rejected deployment" in failed.error_message

    reopened = _ledger().get(effect_id)
    assert reopened.status == "failed"
    assert reopened.error_type == "RuntimeError"


def test_postgres_reconcile_can_record_authoritative_failure():
    effect_id = "postgres-reconcile-failure-1"
    _ledger().claim(effect_id, "payment", {"amount": 50})

    receipt = _ledger().reconcile(
        effect_id,
        succeeded=False,
        note="processor confirms no charge was created",
    )

    assert receipt.status == "failed"
    assert receipt.error_type == "ReconciledFailure"
    assert "no charge" in receipt.error_message
