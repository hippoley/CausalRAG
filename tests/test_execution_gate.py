from __future__ import annotations

import math

from branchpoint import AuthorizationContext, ToolRegistry, ToolSpec
from branchpoint.gateway import ExecutionGateOutcome, ToolExecutionGate


def test_low_risk_authorized_call_is_framework_neutral_allow():
    registry = ToolRegistry(
        [
            ToolSpec(
                "lookup_order",
                "read",
                lambda order_id: {"id": order_id},
                risk=0.05,
                required_permissions=("order.read",),
            )
        ]
    )
    gate = ToolExecutionGate(registry, auto_execute_max_risk=0.1)
    principal = AuthorizationContext.from_permissions(
        "support:alice",
        ["order.read"],
    )

    decision = gate.preview(
        "lookup_order",
        {"order_id": "O-1"},
        authorization_context=principal,
    )

    assert decision.outcome is ExecutionGateOutcome.ALLOW
    assert decision.reason_code == "auto_execute"
    assert decision.proposal_hash
    assert decision.required_permissions == ("order.read",)


def test_missing_authority_is_denied_before_risk_policy():
    registry = ToolRegistry(
        [
            ToolSpec(
                "refund_order",
                "refund",
                lambda order_id: {"ok": True},
                risk=0.0,
                required_permissions=("refund.write",),
            )
        ]
    )
    gate = ToolExecutionGate(registry, auto_execute_max_risk=1.0)

    decision = gate.preview("refund_order", {"order_id": "O-2"})

    assert decision.outcome is ExecutionGateOutcome.DENY
    assert decision.reason_code == "missing_principal"
    assert decision.missing_permissions == ("refund.write",)


def test_high_risk_and_irreversible_tools_require_trusted_human():
    registry = ToolRegistry(
        [
            ToolSpec("restart", "restart", lambda: None, risk=0.8),
            ToolSpec("delete", "delete", lambda: None, reversible=False),
        ]
    )
    gate = ToolExecutionGate(registry, auto_execute_max_risk=0.1)

    risky = gate.preview("restart", {})
    irreversible = gate.preview("delete", {})

    assert risky.outcome is ExecutionGateOutcome.REQUIRE_HUMAN
    assert risky.reason_code == "risk_threshold"
    assert irreversible.outcome is ExecutionGateOutcome.REQUIRE_HUMAN
    assert irreversible.reason_code == "irreversible_action"


def test_proposal_hash_is_stable_across_object_key_order():
    registry = ToolRegistry([ToolSpec("tool", "tool", lambda **kwargs: kwargs)])
    gate = ToolExecutionGate(registry)

    first = gate.preview("tool", {"a": 1, "b": {"x": 2, "y": 3}})
    second = gate.preview("tool", {"b": {"y": 3, "x": 2}, "a": 1})

    assert first.proposal_hash == second.proposal_hash


def test_non_canonical_arguments_and_invalid_policy_numbers_fail_closed():
    registry = ToolRegistry(
        [
            ToolSpec("bad_args", "bad", lambda value: value),
            ToolSpec("bad_risk", "bad", lambda: None, risk=math.nan),
            ToolSpec("bad_cost", "bad", lambda: None, cost=math.inf),
        ]
    )
    gate = ToolExecutionGate(registry, auto_execute_max_risk=1.0)

    bad_args = gate.preview("bad_args", {"value": object()})
    bad_risk = gate.preview("bad_risk", {})
    bad_cost = gate.preview("bad_cost", {})

    assert bad_args.outcome is ExecutionGateOutcome.DENY
    assert bad_args.reason_code == "non_canonical_arguments"
    assert bad_risk.outcome is ExecutionGateOutcome.REQUIRE_HUMAN
    assert bad_risk.reason_code == "invalid_tool_risk"
    assert bad_cost.outcome is ExecutionGateOutcome.REQUIRE_HUMAN
    assert bad_cost.reason_code == "invalid_tool_cost"


def test_unknown_tool_is_denied_not_invented():
    gate = ToolExecutionGate(ToolRegistry())

    decision = gate.preview("model_invented_tool", {})

    assert decision.outcome is ExecutionGateOutcome.DENY
    assert decision.reason_code == "unknown_tool"
