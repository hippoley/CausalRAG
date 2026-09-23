from __future__ import annotations

import asyncio
import inspect

import pytest

from agents import function_tool
from agents.items import ToolApprovalItem
from agents.run_state import RunState
from agents.tool_context import ToolContext

from branchpoint import (
    AuthorizationContext,
    AuthorizationDenied,
    ExecutionBoundaryError,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import ApprovalOutcome, OpenAIAgentsApprovalAdapter


class DummyAgent:
    pass


def test_current_sdk_tool_approval_item_matches_adapter_surface():
    interruption = ToolApprovalItem(
        agent=DummyAgent(),
        raw_item={
            "type": "function_call",
            "name": "lookup_order",
            "arguments": '{"order_id":"O-7"}',
            "call_id": "call-sdk-1",
        },
        tool_name="lookup_order",
    )
    adapter = OpenAIAgentsApprovalAdapter(
        [ToolSpec("lookup_order", "read", lambda order_id: order_id)],
        auto_approve_max_risk=0.0,
    )

    decision = adapter.decide(interruption)

    assert decision.outcome is ApprovalOutcome.ALLOW
    assert decision.tool_name == "lookup_order"
    assert decision.call_id == "call-sdk-1"
    assert decision.arguments == {"order_id": "O-7"}


def test_current_sdk_run_state_rejection_signature_supports_adapter_message():
    parameters = inspect.signature(RunState.reject).parameters
    assert "approval_item" in parameters
    assert "rejection_message" in parameters


def test_current_function_tool_accepts_branchpoint_needs_approval_callback():
    adapter = OpenAIAgentsApprovalAdapter(
        [ToolSpec("lookup_order", "read", lambda order_id: order_id)],
        auto_approve_max_risk=0.0,
    )

    @function_tool(needs_approval=adapter.needs_approval("lookup_order"))
    def lookup_order(order_id: str) -> str:
        """Read an order."""
        return order_id

    assert lookup_order.name == "lookup_order"
    assert callable(lookup_order.needs_approval)
    assert asyncio.run(
        lookup_order.needs_approval(None, {"order_id": "O-9"}, "call-sdk-2")
    ) is False


def test_branchpoint_bound_function_tool_uses_durable_receipt_and_call_id(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "openai-agents.sqlite3")

    def charge_card(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {
            "charge_id": "ch_sdk_1",
            "amount": amount,
            "idempotency_key": idempotency_key,
        }

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge_card",
                "Charge a card exactly once.",
                charge_card,
                risk=0.0,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("payments.charge",),
            )
        ],
        execution_ledger=ledger,
    )
    principal = AuthorizationContext.from_permissions(
        "billing:alice",
        ["payments.charge"],
    )
    adapter = OpenAIAgentsApprovalAdapter(
        registry,
        auto_approve_max_risk=0.0,
        authorization_context=principal,
    )
    tool = adapter.function_tool(
        "charge_card",
        params_json_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
            },
            "required": ["amount"],
            "additionalProperties": False,
        },
    )

    needs_approval = asyncio.run(
        tool.needs_approval(None, {"amount": 25}, "call-durable-1")
    )
    assert needs_approval is False

    context = ToolContext(
        context=None,
        tool_name="charge_card",
        tool_call_id="call-durable-1",
        tool_arguments='{"amount":25}',
    )
    first = asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))
    second = asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))

    effect_id = "openai-agents:charge_card:call-durable-1"
    assert first == second
    assert calls == [(25, effect_id)]
    receipt = ledger.get(effect_id)
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert receipt.result == first


def test_bound_durable_tool_still_rechecks_authorization_at_execution(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "openai-agents-auth.sqlite3")
    registry = ToolRegistry(
        [
            ToolSpec(
                "charge_card",
                "Charge a card.",
                lambda amount, idempotency_key: {"ok": True},
                risk=0.0,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("payments.charge",),
            )
        ],
        execution_ledger=ledger,
    )
    allowed = AuthorizationContext.from_permissions(
        "billing:alice",
        ["payments.charge"],
    )
    denied = AuthorizationContext.from_permissions("billing:alice", [])

    current = {"context": allowed}

    def authority(_run_context, _tool_name, _arguments, _call_id):
        return current["context"]

    adapter = OpenAIAgentsApprovalAdapter(
        registry,
        auto_approve_max_risk=0.0,
        authorization_resolver=authority,
    )
    tool = adapter.function_tool(
        "charge_card",
        params_json_schema={
            "type": "object",
            "properties": {"amount": {"type": "number"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
    )

    assert asyncio.run(
        tool.needs_approval(None, {"amount": 25}, "call-auth-1")
    ) is False

    current["context"] = denied
    context = ToolContext(
        context=None,
        tool_name="charge_card",
        tool_call_id="call-auth-1",
        tool_arguments='{"amount":25}',
    )

    with pytest.raises(AuthorizationDenied, match="lacks permission"):
        asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))

    assert ledger.get("openai-agents:charge_card:call-auth-1") is None


def test_bound_durable_tool_requires_configured_execution_ledger():
    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "charge_card",
                "Charge a card.",
                lambda amount, idempotency_key: {"ok": True},
                risk=0.0,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        auto_approve_max_risk=0.0,
    )

    with pytest.raises(ExecutionBoundaryError, match="durable execution receipt"):
        adapter.function_tool(
            "charge_card",
            params_json_schema={
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
                "additionalProperties": False,
            },
        )


def test_function_tool_rejects_async_branchpoint_handler():
    async def async_handler(value):
        return value

    adapter = OpenAIAgentsApprovalAdapter(
        [ToolSpec("async_tool", "async", async_handler)]
    )

    with pytest.raises(TypeError, match="synchronous ToolSpec handler"):
        adapter.function_tool(
            "async_tool",
            params_json_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
