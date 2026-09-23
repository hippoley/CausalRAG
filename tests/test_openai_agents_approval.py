from __future__ import annotations

import asyncio

import pytest

from branchpoint import AuthorizationContext, ToolSpec
from branchpoint.integrations import ApprovalOutcome, OpenAIAgentsApprovalAdapter


class FakeInterruption:
    def __init__(self, tool_name, arguments, call_id="call-1"):
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_id = call_id


class FakeState:
    def __init__(self):
        self.approved = []
        self.rejected = []

    def approve(self, item):
        self.approved.append(item)

    def reject(self, item, *, rejection_message=None):
        self.rejected.append((item, rejection_message))


class FakeRunResult:
    def __init__(self, interruptions):
        self.interruptions = list(interruptions)
        self.state = FakeState()

    def to_state(self):
        return self.state


def test_low_risk_authorized_tool_is_auto_approved():
    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "lookup_order",
                "read order",
                lambda order_id: {"id": order_id},
                risk=0.05,
                required_permissions=("order.read",),
            )
        ],
        auto_approve_max_risk=0.10,
    )
    principal = AuthorizationContext.from_permissions(
        "support:alice",
        ["order.read"],
    )
    interruption = FakeInterruption(
        "lookup_order",
        '{"order_id":"O-1"}',
    )

    resolution = adapter.resolve(
        FakeRunResult([interruption]),
        authorization_context=principal,
    )

    assert resolution.can_resume is True
    assert resolution.pending == ()
    assert resolution.decisions[0].outcome is ApprovalOutcome.ALLOW
    assert resolution.state.approved == [interruption]
    assert resolution.state.rejected == []


def test_missing_permission_is_rejected_not_escalated():
    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "refund_order",
                "refund",
                lambda order_id: {"ok": True},
                required_permissions=("refund.write",),
            )
        ]
    )
    principal = AuthorizationContext.from_permissions("support:bob", [])
    interruption = FakeInterruption(
        "refund_order",
        '{"order_id":"O-2"}',
    )

    resolution = adapter.resolve(
        FakeRunResult([interruption]),
        authorization_context=principal,
    )

    decision = resolution.decisions[0]
    assert decision.outcome is ApprovalOutcome.DENY
    assert decision.reason_code == "missing_permission"
    assert decision.missing_permissions == ("refund.write",)
    assert resolution.pending == ()
    assert resolution.state.approved == []
    assert resolution.state.rejected[0][0] is interruption
    assert "refund.write" in resolution.state.rejected[0][1]


def test_high_risk_and_irreversible_calls_remain_pending_for_human():
    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "restart_service",
                "restart",
                lambda: None,
                risk=0.8,
            ),
            ToolSpec(
                "delete_account",
                "delete",
                lambda: None,
                risk=0.0,
                reversible=False,
            ),
        ],
        auto_approve_max_risk=0.1,
    )
    high_risk = FakeInterruption("restart_service", "{}")
    irreversible = FakeInterruption("delete_account", "{}", "call-2")

    resolution = adapter.resolve(FakeRunResult([high_risk, irreversible]))

    assert [d.outcome for d in resolution.decisions] == [
        ApprovalOutcome.REQUIRE_HUMAN,
        ApprovalOutcome.REQUIRE_HUMAN,
    ]
    assert [d.reason_code for d in resolution.decisions] == [
        "risk_threshold",
        "irreversible_action",
    ]
    assert resolution.pending == (high_risk, irreversible)
    assert resolution.can_resume is False
    assert resolution.state.approved == []
    assert resolution.state.rejected == []


def test_durable_tool_is_denied_when_sdk_execution_would_bypass_receipt_boundary():
    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "charge_card",
                "charge",
                lambda amount: {"ok": True},
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ]
    )
    interruption = FakeInterruption("charge_card", '{"amount":25}')

    resolution = adapter.resolve(FakeRunResult([interruption]))

    decision = resolution.decisions[0]
    assert decision.outcome is ApprovalOutcome.DENY
    assert decision.reason_code == "durable_execution_boundary_required"
    assert resolution.state.rejected[0][0] is interruption


@pytest.mark.parametrize(
    "arguments, reason_code",
    [
        ("", "missing_arguments"),
        ("not-json", "malformed_arguments"),
        ("[]", "non_object_arguments"),
        ("NaN", "malformed_arguments"),
        ('{"value":Infinity}', "malformed_arguments"),
    ],
)
def test_uninspectable_arguments_fail_closed_to_human(arguments, reason_code):
    adapter = OpenAIAgentsApprovalAdapter(
        [ToolSpec("read", "read", lambda: None)]
    )

    decision = adapter.decide(FakeInterruption("read", arguments))

    assert decision.outcome is ApprovalOutcome.REQUIRE_HUMAN
    assert decision.reason_code == reason_code


def test_unknown_tool_remains_pending():
    adapter = OpenAIAgentsApprovalAdapter(
        [ToolSpec("known", "known", lambda: None)]
    )
    result = FakeRunResult([FakeInterruption("model_invented_tool", "{}")])

    resolution = adapter.resolve(result)

    assert resolution.decisions[0].reason_code == "unknown_tool"
    assert resolution.pending == tuple(result.interruptions)


def test_needs_approval_callback_uses_same_policy_before_sdk_execution():
    seen = []

    def authority(run_context, tool_name, arguments, call_id):
        seen.append((run_context, tool_name, dict(arguments), call_id))
        return AuthorizationContext.from_permissions(
            "operator:1",
            ["tool.run"],
        )

    adapter = OpenAIAgentsApprovalAdapter(
        [
            ToolSpec(
                "safe_tool",
                "safe",
                lambda: None,
                risk=0.01,
                required_permissions=("tool.run",),
            ),
            ToolSpec(
                "risky_tool",
                "risky",
                lambda: None,
                risk=0.8,
                required_permissions=("tool.run",),
            ),
        ],
        auto_approve_max_risk=0.1,
        authorization_resolver=authority,
    )

    safe = adapter.needs_approval("safe_tool")
    risky = adapter.needs_approval("risky_tool")

    assert asyncio.run(safe("ctx", {"x": 1}, "call-safe")) is False
    assert asyncio.run(risky("ctx", {"x": 2}, "call-risky")) is True
    assert seen == [
        ("ctx", "safe_tool", {"x": 1}, "call-safe"),
        ("ctx", "risky_tool", {"x": 2}, "call-risky"),
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_auto_approval_threshold_must_be_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        OpenAIAgentsApprovalAdapter([], auto_approve_max_risk=value)
