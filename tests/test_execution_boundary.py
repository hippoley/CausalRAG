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
