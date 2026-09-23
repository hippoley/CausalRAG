from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agents import Agent, Runner
from agents.testing import ScriptedModel, assistant_message, function_call

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import ApprovalOutcome, OpenAIAgentsApprovalAdapter


async def run_demo():
    calls = []
    ledger_path = (
        Path(tempfile.mkdtemp(prefix="branchpoint-openai-runner-"))
        / "effects.sqlite3"
    )
    ledger = SQLiteExecutionLedger(ledger_path)

    async def charge_card(amount, idempotency_key):
        await asyncio.sleep(0)
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_runner_demo_1", "amount": amount}

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
        "billing:demo",
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
                    call_id="call-runner-demo-1",
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

    first = await Runner.run(agent, "Charge 25.")
    interruption = first.interruptions[0]
    decision = adapter.decide(interruption)

    effect_id = "openai-agents:charge_card:call-runner-demo-1"
    print("paused:", len(first.interruptions) == 1)
    print("branchpoint_decision:", decision.outcome.value)
    print("reason_code:", decision.reason_code)
    print("effect_before_approval:", ledger.get(effect_id))
    print("external_calls_before_approval:", len(calls))

    if decision.outcome is not ApprovalOutcome.REQUIRE_HUMAN:
        raise RuntimeError("demo expected a human approval boundary")

    state = first.to_state()
    state.approve(interruption)
    resumed = await Runner.run(agent, state)

    receipt = ledger.get(effect_id)
    print("final_output:", resumed.final_output)
    print("external_calls_after_resume:", len(calls))
    print("receipt_status:", None if receipt is None else receipt.status)
    print("effect_id:", effect_id)
    print("downstream_idempotency_key:", calls[0][1] if calls else None)

    return {
        "paused": len(first.interruptions) == 1,
        "decision": decision.outcome.value,
        "reason_code": decision.reason_code,
        "final_output": resumed.final_output,
        "external_calls": len(calls),
        "receipt_status": None if receipt is None else receipt.status,
        "effect_id": effect_id,
    }


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
