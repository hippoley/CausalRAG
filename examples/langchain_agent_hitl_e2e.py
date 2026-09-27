from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import Self, override

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import (
    langchain_branchpoint_stack,
    langchain_tool_schema,
)


class DemoModel(GenericFakeChatModel):
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


def main() -> None:
    calls = []
    ledger = SQLiteExecutionLedger(
        Path(tempfile.mkdtemp(prefix="branchpoint-langchain-agent-"))
        / "effects.sqlite3"
    )

    def charge_card(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_demo_1", "amount": amount}

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
        "billing:demo",
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

    call_id = "call-langchain-demo-1"
    effect_id = f"langchain:charge_card:{call_id}"
    model = DemoModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="charge_card",
                            args={"amount": 25},
                            id=call_id,
                            type="tool_call",
                        )
                    ],
                ),
                AIMessage(content="Payment completed."),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=[tool],
        middleware=[hitl, execution],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "branchpoint-langchain-demo"}}

    interrupted = agent.invoke(
        {"messages": [HumanMessage("Charge 25.")]},
        config,
    )
    print("paused:", "__interrupt__" in interrupted)
    print("external_calls_before_approval:", len(calls))
    print("receipt_before_approval:", ledger.get(effect_id))

    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
    )
    receipt = ledger.get(effect_id)

    print("final_output:", final["messages"][-1].content)
    print("external_calls_after_resume:", len(calls))
    print("effect_id:", effect_id)
    print("receipt_status:", None if receipt is None else receipt.status)
    print("downstream_idempotency_key:", calls[0][1] if calls else None)


if __name__ == "__main__":
    main()
