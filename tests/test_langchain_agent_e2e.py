from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import Self, override

from branchpoint import (
    AuthorizationContext,
    AuthorizationDenied,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import (
    langchain_branchpoint_stack,
    langchain_tool_schema,
)


class _ToolCallingModel(GenericFakeChatModel):
    """Deterministic model that accepts LangChain tool binding."""

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        _ = (tools, tool_choice, kwargs)
        return self


def _model(tool_name: str, *, call_id: str, arguments: dict[str, Any]):
    return _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=tool_name,
                            args=arguments,
                            id=call_id,
                            type="tool_call",
                        )
                    ],
                ),
                AIMessage(content="Payment completed."),
            ]
        )
    )


def _tool_message(messages, tool_name: str) -> ToolMessage:
    matches = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == tool_name
    ]
    assert len(matches) == 1
    return matches[0]


def test_create_agent_hitl_resume_executes_through_branchpoint_once(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-agent-e2e.sqlite3")

    def charge_card(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_langchain_e2e_1", "amount": amount}

    spec = ToolSpec(
        "charge_card",
        "Charge a card.",
        charge_card,
        risk=0.9,
        reversible=True,
        require_durable_receipt=True,
        idempotency_key_argument="idempotency_key",
        required_permissions=("payments.charge",),
    )
    registry = ToolRegistry([spec], execution_ledger=ledger)
    principal = AuthorizationContext.from_permissions(
        "billing:alice",
        ["payments.charge"],
    )
    hitl, execution = langchain_branchpoint_stack(
        registry,
        auto_approve_max_risk=0.1,
        authorization_context=principal,
    )
    tool = langchain_tool_schema(
        spec,
        args_schema={
            "type": "object",
            "properties": {"amount": {"type": "number"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
    )
    call_id = "call-langchain-e2e-1"
    effect_id = f"langchain:charge_card:{call_id}"
    agent = create_agent(
        model=_model(
            "charge_card",
            call_id=call_id,
            arguments={"amount": 25},
        ),
        tools=[tool],
        middleware=[hitl, execution],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "branchpoint-langchain-e2e"}}

    interrupted = agent.invoke(
        {"messages": [HumanMessage("Charge 25.")]},
        config,
    )

    assert "__interrupt__" in interrupted
    assert calls == []
    assert ledger.get(effect_id) is None

    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
    )

    assert "__interrupt__" not in final
    assert isinstance(final["messages"][-1], AIMessage)
    assert final["messages"][-1].content == "Payment completed."
    assert calls == [(25, effect_id)]

    receipt = ledger.get(effect_id)
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert receipt.result == {
        "charge_id": "ch_langchain_e2e_1",
        "amount": 25,
    }

    tool_message = _tool_message(final["messages"], "charge_card")
    evidence = tool_message.artifact["branchpoint"]
    assert evidence["execution_boundary"] == "branchpoint"
    assert evidence["effect_id"] == effect_id
    assert evidence["receipt_status"] == "succeeded"
    assert evidence["authorization_rechecked"] is True
    assert evidence["replayed"] is False


def test_create_agent_resume_rechecks_authority_after_human_approval(tmp_path):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "langchain-agent-auth-e2e.sqlite3")

    def charge_card(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {"charge_id": "should_not_exist"}

    spec = ToolSpec(
        "charge_card",
        "Charge a card.",
        charge_card,
        risk=0.9,
        reversible=True,
        require_durable_receipt=True,
        idempotency_key_argument="idempotency_key",
        required_permissions=("payments.charge",),
    )
    registry = ToolRegistry([spec], execution_ledger=ledger)
    allowed = AuthorizationContext.from_permissions(
        "billing:alice",
        ["payments.charge"],
    )
    denied = AuthorizationContext.from_permissions("billing:alice", [])
    current = {"principal": allowed}

    def resolver(_runtime_context, _tool_name, _arguments, _call_id):
        return current["principal"]

    hitl, execution = langchain_branchpoint_stack(
        registry,
        auto_approve_max_risk=0.1,
        authorization_resolver=resolver,
    )
    tool = langchain_tool_schema(
        spec,
        args_schema={
            "type": "object",
            "properties": {"amount": {"type": "number"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
    )
    call_id = "call-langchain-auth-e2e-1"
    effect_id = f"langchain:charge_card:{call_id}"
    agent = create_agent(
        model=_model(
            "charge_card",
            call_id=call_id,
            arguments={"amount": 25},
        ),
        tools=[tool],
        middleware=[hitl, execution],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "branchpoint-langchain-auth-e2e"}}

    interrupted = agent.invoke(
        {"messages": [HumanMessage("Charge 25.")]},
        config,
    )
    assert "__interrupt__" in interrupted
    assert ledger.get(effect_id) is None
    assert calls == []

    current["principal"] = denied

    with pytest.raises(AuthorizationDenied, match="payments.charge"):
        agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config,
        )

    assert ledger.get(effect_id) is None
    assert calls == []
