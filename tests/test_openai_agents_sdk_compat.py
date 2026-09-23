from __future__ import annotations

import asyncio
import inspect

import pytest

from agents import function_tool
from agents.items import ToolApprovalItem, ToolCallOutputItem
from agents.run_state import RunState
from agents.tool_context import ToolContext

from branchpoint import (
    AuthorizationContext,
    AuthorizationDecision,
    AuthorizationDenied,
    CapabilityAuthorizationPolicy,
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
    first_custom_data = dict(context._custom_data)
    second = asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))
    second_custom_data = dict(context._custom_data)

    effect_id = "openai-agents:charge_card:call-durable-1"
    assert first == second
    assert calls == [(25, effect_id)]
    receipt = ledger.get(effect_id)
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert receipt.result == first

    first_trace = first_custom_data["branchpoint"]
    second_trace = second_custom_data["branchpoint"]
    assert first_trace["schema_version"] == "branchpoint.openai-agents.execution.v1"
    assert first_trace["execution_boundary"] == "branchpoint"
    assert first_trace["effect_id"] == effect_id
    assert first_trace["receipt_status"] == "succeeded"
    assert first_trace["replayed"] is False
    assert first_trace["effect_hash"] == receipt.effect_hash
    assert first_trace["authorization_rechecked"] is True
    assert second_trace["replayed"] is True


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


def test_bound_durable_tool_bypasses_only_receipt_bypass_guard_not_risk(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "openai-agents-risk.sqlite3")
    registry = ToolRegistry(
        [
            ToolSpec(
                "restart_cluster",
                "Restart a cluster.",
                lambda cluster: {"cluster": cluster, "restarted": True},
                risk=0.9,
                reversible=True,
                require_durable_receipt=True,
            )
        ],
        execution_ledger=ledger,
    )
    adapter = OpenAIAgentsApprovalAdapter(
        registry,
        auto_approve_max_risk=0.1,
    )
    tool = adapter.function_tool(
        "restart_cluster",
        params_json_schema={
            "type": "object",
            "properties": {"cluster": {"type": "string"}},
            "required": ["cluster"],
            "additionalProperties": False,
        },
    )

    assert asyncio.run(
        tool.needs_approval(None, {"cluster": "prod"}, "call-risk-1")
    ) is True

    interruption = ToolApprovalItem(
        agent=DummyAgent(),
        raw_item={
            "type": "function_call",
            "name": "restart_cluster",
            "arguments": '{"cluster":"prod"}',
            "call_id": "call-risk-1",
        },
        tool_name="restart_cluster",
    )
    decision = adapter.decide(interruption)
    assert decision.outcome is ApprovalOutcome.REQUIRE_HUMAN
    assert decision.reason_code == "risk_threshold"


def test_explicit_adapter_policy_is_reused_at_bound_tool_execution(tmp_path):
    class DenyPolicy(CapabilityAuthorizationPolicy):
        def authorize(
            self,
            context,
            *,
            tool_name,
            required_permissions,
            arguments,
        ):
            return AuthorizationDecision(
                allowed=False,
                principal_id=None,
                required_permissions=tuple(required_permissions),
                missing_permissions=tuple(required_permissions),
                reason_code="deployment_freeze",
                reason="Deployment freeze is active.",
                policy_id="test.freeze.v1",
            )

    ledger = SQLiteExecutionLedger(tmp_path / "openai-agents-policy.sqlite3")
    registry = ToolRegistry(
        [
            ToolSpec(
                "deploy",
                "Deploy a service.",
                lambda service: {"service": service, "deployed": True},
                risk=0.0,
                reversible=True,
            )
        ],
        execution_ledger=ledger,
    )
    policy = DenyPolicy()
    adapter = OpenAIAgentsApprovalAdapter(
        registry,
        auto_approve_max_risk=1.0,
        authorization_policy=policy,
    )
    tool = adapter.function_tool(
        "deploy",
        params_json_schema={
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
            "additionalProperties": False,
        },
    )
    assert registry.authorization_policy is policy

    context = ToolContext(
        context=None,
        tool_name="deploy",
        tool_call_id="call-policy-1",
        tool_arguments='{"service":"api"}',
    )
    with pytest.raises(AuthorizationDenied, match="Deployment freeze"):
        asyncio.run(tool.on_invoke_tool(context, '{"service":"api"}'))


def test_sdk_only_custom_data_is_not_replayed_to_model_input():
    output = ToolCallOutputItem(
        agent=DummyAgent(),
        raw_item={
            "type": "function_call_output",
            "call_id": "call-custom-data-1",
            "output": "ok",
        },
        output="ok",
        custom_data={
            "branchpoint": {
                "effect_id": "effect-1",
                "receipt_status": "succeeded",
                "replayed": True,
            }
        },
    )

    input_item = output.to_input_item()

    assert input_item["type"] == "function_call_output"
    assert input_item["call_id"] == "call-custom-data-1"
    assert "custom_data" not in input_item
