import asyncio
import math

import pytest

from branchpoint import ActionKind, CandidateAction, CausalAgentLoop, ToolRegistry, ToolSpec
from branchpoint.execution import (
    EffectIdentityConflict,
    ExecutionInProgress,
    NonCanonicalEffect,
    SQLiteExecutionLedger,
)


def test_completed_effect_is_replayed_without_second_side_effect(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "effects.sqlite3")

    def send_email(**kwargs):
        calls.append(dict(kwargs))
        return {"message_id": "m-1", "accepted": True}

    registry = ToolRegistry(
        [ToolSpec("send_email", "send", send_email)],
        execution_ledger=ledger,
    )

    first = registry.execute(
        "send_email",
        {"to": "a@example.com"},
        effect_id="run-1:0:send_email",
    )
    second = registry.execute(
        "send_email",
        {"to": "a@example.com"},
        effect_id="run-1:0:send_email",
    )

    assert first == second == {"message_id": "m-1", "accepted": True}
    assert len(calls) == 1
    assert ledger.get("run-1:0:send_email").status == "succeeded"


def test_effect_id_cannot_be_rebound_to_different_arguments(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "effects.sqlite3")
    registry = ToolRegistry(
        [ToolSpec("charge", "charge", lambda **kwargs: {"ok": True})],
        execution_ledger=ledger,
    )
    registry.execute("charge", {"amount": 10}, effect_id="payment-1")

    with pytest.raises(EffectIdentityConflict):
        registry.execute("charge", {"amount": 11}, effect_id="payment-1")


def test_in_flight_crash_window_fails_closed_until_reconciled(tmp_path):
    path = tmp_path / "effects.sqlite3"
    ledger = SQLiteExecutionLedger(path)
    ledger.claim("effect-1", "door_unlock", {"door": "front"})

    calls = []
    reopened = SQLiteExecutionLedger(path)
    registry = ToolRegistry(
        [ToolSpec("door_unlock", "unlock", lambda **kwargs: calls.append(kwargs))],
        execution_ledger=reopened,
    )

    with pytest.raises(ExecutionInProgress):
        registry.execute(
            "door_unlock",
            {"door": "front"},
            effect_id="effect-1",
        )
    assert calls == []

    reopened.reconcile(
        "effect-1",
        succeeded=True,
        result={"door": "front", "state": "unlocked"},
    )
    replay = registry.execute(
        "door_unlock",
        {"door": "front"},
        effect_id="effect-1",
    )
    assert replay == {"door": "front", "state": "unlocked"}
    assert calls == []


def test_downstream_idempotency_key_is_propagated(tmp_path):
    seen = []
    ledger = SQLiteExecutionLedger(tmp_path / "effects.sqlite3")

    def charge(amount, idempotency_key):
        seen.append((amount, idempotency_key))
        return {"charge_id": "ch_1"}

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "charge card",
                charge,
                reversible=False,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )

    result = registry.execute("charge", {"amount": 25}, effect_id="checkout-9")
    assert result == {"charge_id": "ch_1"}
    assert seen == [(25, "checkout-9")]


def test_durable_tool_requires_ledger_and_effect_id():
    registry = ToolRegistry(
        [
            ToolSpec(
                "deploy",
                "deploy",
                lambda: {"ok": True},
                require_durable_receipt=True,
            )
        ]
    )
    with pytest.raises(Exception, match="durable execution receipt"):
        registry.execute("deploy", {})


def test_non_finite_effect_values_are_rejected(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "effects.sqlite3")
    with pytest.raises(NonCanonicalEffect):
        ledger.claim("effect-1", "tool", {"value": math.nan})


class _OneEffectReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(ActionKind.INTERVENE, "send", expected_goal_gain=1.0)]
        return [CandidateAction(ActionKind.STOP, "stop", rationale="done")]

    def uncertainty(self, state, world_model):
        return None


def test_agent_loop_assigns_a_durable_effect_identity(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "effects.sqlite3")
    registry = ToolRegistry(
        [ToolSpec("send", "send", lambda: {"ok": True})],
        execution_ledger=ledger,
    )
    loop = CausalAgentLoop(reasoner=_OneEffectReasoner(), tools=registry)
    state = loop.run("send once", max_steps=2)

    run_id = state.scratch["execution_run_id"]
    receipt = ledger.get(f"{run_id}:0:send")
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert state.observations[0].result == {"ok": True}


def test_sync_execute_rejects_async_handler_before_receipt_claim(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "async-sync-guard.sqlite3")

    async def send():
        return {"ok": True}

    registry = ToolRegistry(
        [
            ToolSpec(
                "send",
                "send",
                send,
                require_durable_receipt=True,
            )
        ],
        execution_ledger=ledger,
    )

    with pytest.raises(TypeError, match="use execute_async"):
        registry.execute("send", {}, effect_id="async-sync-guard-1")

    assert ledger.get("async-sync-guard-1") is None


def test_async_durable_effect_is_replayed_without_second_side_effect(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "async-effects.sqlite3")

    async def charge(amount, idempotency_key):
        await asyncio.sleep(0)
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_async_1", "amount": amount}

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "charge",
                charge,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )

    async def scenario():
        first = await registry.execute_async(
            "charge",
            {"amount": 25},
            effect_id="async-payment-1",
        )
        second = await registry.execute_async(
            "charge",
            {"amount": 25},
            effect_id="async-payment-1",
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first == second == {"charge_id": "ch_async_1", "amount": 25}
    assert calls == [(25, "async-payment-1")]
    receipt = ledger.get("async-payment-1")
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert receipt.result == first


def test_sync_idempotency_argument_conflict_fails_before_receipt_claim(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "sync-idempotency-conflict.sqlite3")
    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "charge",
                lambda amount, idempotency_key: {"ok": True},
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )

    with pytest.raises(EffectIdentityConflict, match="conflicts with effect_id"):
        registry.execute(
            "charge",
            {"amount": 25, "idempotency_key": "caller-key"},
            effect_id="branchpoint-key",
        )

    assert ledger.get("branchpoint-key") is None


def test_async_idempotency_argument_conflict_fails_before_receipt_claim(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "async-idempotency-conflict.sqlite3")

    async def charge(amount, idempotency_key):
        return {"ok": True}

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "charge",
                charge,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )

    async def scenario():
        with pytest.raises(EffectIdentityConflict, match="conflicts with effect_id"):
            await registry.execute_async(
                "charge",
                {"amount": 25, "idempotency_key": "caller-key"},
                effect_id="branchpoint-key-async",
            )

    asyncio.run(scenario())
    assert ledger.get("branchpoint-key-async") is None


def test_execute_async_moves_sync_handler_off_event_loop(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "async-sync-handler.sqlite3")
    calls = []

    def handler(value):
        calls.append(value)
        return {"value": value}

    registry = ToolRegistry(
        [ToolSpec("sync_tool", "sync", handler)],
        execution_ledger=ledger,
    )

    result = asyncio.run(
        registry.execute_async(
            "sync_tool",
            {"value": "ok"},
            effect_id="async-sync-handler-1",
        )
    )

    assert result == {"value": "ok"}
    assert calls == ["ok"]


class _CompletionFailingLedger(SQLiteExecutionLedger):
    def complete(self, effect_id, result):
        raise RuntimeError("receipt store unavailable")


def test_sync_completion_failure_keeps_receipt_in_flight_for_reconciliation(tmp_path):
    calls = []
    ledger = _CompletionFailingLedger(tmp_path / "completion-failure-sync.sqlite3")

    def external_effect():
        calls.append("executed")
        return {"ok": True}

    registry = ToolRegistry(
        [ToolSpec("external_effect", "effect", external_effect)],
        execution_ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="receipt store unavailable"):
        registry.execute(
            "external_effect",
            {},
            effect_id="completion-failure-sync-1",
        )

    receipt = ledger.get("completion-failure-sync-1")
    assert receipt is not None
    assert receipt.status == "in_flight"
    assert calls == ["executed"]

    with pytest.raises(ExecutionInProgress):
        registry.execute(
            "external_effect",
            {},
            effect_id="completion-failure-sync-1",
        )
    assert calls == ["executed"]


def test_async_completion_failure_keeps_receipt_in_flight_for_reconciliation(tmp_path):
    calls = []
    ledger = _CompletionFailingLedger(tmp_path / "completion-failure-async.sqlite3")

    async def external_effect():
        await asyncio.sleep(0)
        calls.append("executed")
        return {"ok": True}

    registry = ToolRegistry(
        [ToolSpec("external_effect_async", "effect", external_effect)],
        execution_ledger=ledger,
    )

    async def first_attempt():
        with pytest.raises(RuntimeError, match="receipt store unavailable"):
            await registry.execute_async(
                "external_effect_async",
                {},
                effect_id="completion-failure-async-1",
            )

    asyncio.run(first_attempt())

    receipt = ledger.get("completion-failure-async-1")
    assert receipt is not None
    assert receipt.status == "in_flight"
    assert calls == ["executed"]

    async def replay_attempt():
        with pytest.raises(ExecutionInProgress):
            await registry.execute_async(
                "external_effect_async",
                {},
                effect_id="completion-failure-async-1",
            )

    asyncio.run(replay_attempt())
    assert calls == ["executed"]
