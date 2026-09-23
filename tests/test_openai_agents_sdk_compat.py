from __future__ import annotations

import asyncio
import inspect

from agents import function_tool
from agents.items import ToolApprovalItem
from agents.run_state import RunState

from branchpoint import ToolSpec
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
