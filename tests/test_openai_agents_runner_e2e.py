from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.testing import ScriptedModel, assistant_message, function_call

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import ApprovalOutcome, OpenAIAgentsApprovalAdapter


def test_runner_interrupt_resume_executes_high_risk_tool_through_branchpoint_once(
    tmp_path,
):
    calls = []
    ledger = SQLiteExecutionLedger(tmp_path / "runner-e2e.sqlite3")

    async def charge_card(amount, idempotency_key):
        await asyncio.sleep(0)
        calls.append((amount, idempotency_key))
        return {
            "charge_id": "ch_runner_1",
            "amount": amount,
        }

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge_card",
                "Charge a card.",
                charge_card,
                risk=0.9,
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
        auto_approve_max_risk=0.1,
        authorization_context=principal,
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

    model = ScriptedModel(
        [
            [
                function_call(
                    "charge_card",
                    {"amount": 25},
                    call_id="call-runner-e2e-1",
                )
            ],
            [assistant_message("Payment completed.")],
        ]
    )
    agent = Agent(
        name="billing",
        instructions="Use the payment tool when required.",
        model=model,
        tools=[tool],
    )

    async def scenario():
        first = await Runner.run(agent, "Charge 25.")
        assert len(first.interruptions) == 1
        interruption = first.interruptions[0]

        decision = adapter.decide(interruption)
        assert decision.outcome is ApprovalOutcome.REQUIRE_HUMAN
        assert decision.reason_code == "risk_threshold"

        effect_id = "openai-agents:charge_card:call-runner-e2e-1"
        assert ledger.get(effect_id) is None
        assert calls == []

        state = first.to_state()
        state.approve(interruption)
        resumed = await Runner.run(agent, state)
        return resumed, effect_id

    resumed, effect_id = asyncio.run(scenario())

    assert resumed.final_output == "Payment completed."
    assert calls == [(25, effect_id)]
    receipt = ledger.get(effect_id)
    assert receipt is not None
    assert receipt.status == "succeeded"
    assert receipt.result == {
        "charge_id": "ch_runner_1",
        "amount": 25,
    }
    assert len(model.calls) == 2
    model.assert_complete()
