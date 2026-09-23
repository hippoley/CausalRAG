from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from branchpoint import (
    AuthorizationContext,
    AuthorizationDenied,
    EffectIdentityConflict,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import (
    LangChainBranchpointError,
    LangChainBranchpointMiddleware,
    langchain_human_in_the_loop,
)


def _request(name, args, call_id="call-1", *, runtime_context=None):
    return ToolCallRequest(
        tool_call={
            "name": name,
            "args": dict(args),
            "id": call_id,
            "type": "tool_call",
        },
        tool=None,
        state={"messages": []},
        runtime=SimpleNamespace(context=runtime_context),
    )


def test_sync_durable_tool_executes_through_branchpoint_and_replays(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "langchain.sqlite3")

    def charge(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_lc_1", "amount": amount}

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "Charge a card.",
                charge,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )
    middleware = LangChainBranchpointMiddleware(registry)
    request = _request("charge", {"amount": 25}, "call-lc-1")
    passthrough_calls = []

    def forbidden_handler(req):
        passthrough_calls.append(req)
        raise AssertionError("canonical Branchpoint tools must not call LangChain handler")

    first = middleware.wrap_tool_call(request, forbidden_handler)
    second = middleware.wrap_tool_call(request, forbidden_handler)

    effect_id = "langchain:charge:call-lc-1"
    assert isinstance(first, ToolMessage)
    assert isinstance(second, ToolMessage)
    assert first.content == second.content == '{"amount":25,"charge_id":"ch_lc_1"}'
    assert calls == [(25, effect_id)]
    assert passthrough_calls == []

    receipt = ledger.get(effect_id)
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert first.artifact["branchpoint"]["replayed"] is False
    assert second.artifact["branchpoint"]["replayed"] is True
    assert second.artifact["branchpoint"]["effect_hash"] == receipt.effect_hash
    assert "arguments" not in second.artifact["branchpoint"]
    assert "principal_id" not in second.artifact["branchpoint"]


def test_async_durable_handler_replays_through_langchain_boundary(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-async.sqlite3")

    async def charge(amount, idempotency_key):
        await asyncio.sleep(0)
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_lc_async_1", "amount": amount}

    middleware = LangChainBranchpointMiddleware(
        ToolRegistry(
            [
                ToolSpec(
                    "charge_async",
                    "Async charge.",
                    charge,
                    require_durable_receipt=True,
                    idempotency_key_argument="idempotency_key",
                )
            ],
            execution_ledger=ledger,
        )
    )
    request = _request("charge_async", {"amount": 25}, "call-lc-async-1")

    async def forbidden_handler(req):
        raise AssertionError("canonical Branchpoint tools must not call LangChain handler")

    async def scenario():
        first = await middleware.awrap_tool_call(request, forbidden_handler)
        second = await middleware.awrap_tool_call(request, forbidden_handler)
        return first, second

    first, second = asyncio.run(scenario())
    effect_id = "langchain:charge_async:call-lc-async-1"

    assert first.content == second.content == (
        '{"amount":25,"charge_id":"ch_lc_async_1"}'
    )
    assert calls == [(25, effect_id)]
    assert ledger.get(effect_id).status == "succeeded"
    assert first.artifact["branchpoint"]["replayed"] is False
    assert second.artifact["branchpoint"]["replayed"] is True


def test_same_langchain_call_id_cannot_be_rebound_to_changed_arguments(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-conflict.sqlite3")

    def charge(amount):
        calls.append(amount)
        return {"amount": amount}

    middleware = LangChainBranchpointMiddleware(
        ToolRegistry(
            [ToolSpec("charge", "charge", charge, require_durable_receipt=True)],
            execution_ledger=ledger,
        )
    )

    middleware.wrap_tool_call(
        _request("charge", {"amount": 10}, "call-same"),
        lambda _request: None,
    )

    with pytest.raises(EffectIdentityConflict):
        middleware.wrap_tool_call(
            _request("charge", {"amount": 11}, "call-same"),
            lambda _request: None,
        )

    assert calls == [10]
    assert ledger.get("langchain:charge:call-same").result == {"amount": 10}


def test_execution_rechecks_dynamic_authorization_after_human_resume(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-auth.sqlite3")
    seen = []

    def resolver(runtime_context, tool_name, arguments, call_id):
        seen.append((runtime_context, tool_name, dict(arguments), call_id))
        return runtime_context

    middleware = LangChainBranchpointMiddleware(
        ToolRegistry(
            [
                ToolSpec(
                    "refund",
                    "Refund.",
                    lambda amount: {"refunded": amount},
                    required_permissions=("refund.write",),
                )
            ],
            execution_ledger=ledger,
        ),
        authorization_resolver=resolver,
    )

    denied = AuthorizationContext.from_permissions("support:bob", [])
    request = _request(
        "refund",
        {"amount": 10},
        "call-refund-1",
        runtime_context=denied,
    )

    with pytest.raises(AuthorizationDenied, match="refund.write"):
        middleware.wrap_tool_call(request, lambda _request: None)

    assert seen == [
        (denied, "refund", {"amount": 10}, "call-refund-1")
    ]


def test_unknown_tools_fail_closed_by_default():
    middleware = LangChainBranchpointMiddleware(
        [ToolSpec("known", "known", lambda: "ok")]
    )

    with pytest.raises(LangChainBranchpointError, match="No canonical"):
        middleware.wrap_tool_call(
            _request("invented", {}, "call-unknown"),
            lambda _request: ToolMessage(
                content="should not run",
                tool_call_id="call-unknown",
            ),
        )


def test_unknown_tool_passthrough_is_explicit_opt_in():
    middleware = LangChainBranchpointMiddleware(
        [ToolSpec("known", "known", lambda: "ok")],
        unknown_tool_policy="passthrough",
    )
    expected = ToolMessage(
        content="external",
        tool_call_id="call-external",
        name="external",
    )

    result = middleware.wrap_tool_call(
        _request("external", {"x": 1}, "call-external"),
        lambda _request: expected,
    )

    assert result is expected


def test_durable_tool_requires_langchain_call_id_before_execution(tmp_path):
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-no-id.sqlite3")
    middleware = LangChainBranchpointMiddleware(
        ToolRegistry(
            [
                ToolSpec(
                    "charge",
                    "charge",
                    lambda amount: {"amount": amount},
                    require_durable_receipt=True,
                )
            ],
            execution_ledger=ledger,
        )
    )

    with pytest.raises(LangChainBranchpointError, match="non-empty tool call id"):
        middleware.wrap_tool_call(
            _request("charge", {"amount": 25}, call_id=""),
            lambda _request: None,
        )


def test_human_in_the_loop_config_uses_approve_reject_only():
    middleware = langchain_human_in_the_loop(
        [
            ToolSpec("read", "read", lambda: None, risk=0.0),
            ToolSpec("restart", "restart", lambda: None, risk=0.8),
            ToolSpec(
                "delete",
                "delete",
                lambda: None,
                risk=0.0,
                reversible=False,
            ),
        ],
        auto_approve_max_risk=0.1,
    )

    assert "read" not in middleware.interrupt_on
    assert middleware.interrupt_on["restart"]["allowed_decisions"] == [
        "approve",
        "reject",
    ]
    assert middleware.interrupt_on["delete"]["allowed_decisions"] == [
        "approve",
        "reject",
    ]


def test_invalid_risk_is_forced_into_human_review():
    middleware = langchain_human_in_the_loop(
        [
            ToolSpec(
                "unknown_risk",
                "unknown",
                lambda: None,
                risk=float("nan"),
            )
        ],
        auto_approve_max_risk=1.0,
    )

    assert middleware.interrupt_on["unknown_risk"]["allowed_decisions"] == [
        "approve",
        "reject",
    ]


def test_branchpoint_registry_does_not_overwrite_langchain_middleware_tools_slot():
    middleware = LangChainBranchpointMiddleware(
        [ToolSpec("read", "read", lambda: "ok")]
    )

    assert middleware.tools == []
    assert "read" in middleware.registry.specs()
