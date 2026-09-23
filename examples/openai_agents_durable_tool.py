from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agents.tool_context import ToolContext

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import OpenAIAgentsApprovalAdapter


def main() -> None:
    calls = []
    ledger_path = Path(tempfile.mkdtemp(prefix="branchpoint-openai-agents-")) / "effects.sqlite3"
    ledger = SQLiteExecutionLedger(ledger_path)

    def charge_card(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {
            "charge_id": "ch_demo_1",
            "amount": amount,
            "idempotency_key": idempotency_key,
        }

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge_card",
                "Charge a card through Branchpoint's durable execution boundary.",
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
        "billing:demo",
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

    call_id = "call-demo-1"
    context = ToolContext(
        context=None,
        tool_name="charge_card",
        tool_call_id=call_id,
        tool_arguments='{"amount":25}',
    )

    first = asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))
    replay = asyncio.run(tool.on_invoke_tool(context, '{"amount":25}'))

    effect_id = f"openai-agents:charge_card:{call_id}"
    receipt = ledger.get(effect_id)

    print("first:", first)
    print("replay:", replay)
    print("external_calls:", len(calls))
    print("effect_id:", effect_id)
    print("receipt_status:", None if receipt is None else receipt.status)


if __name__ == "__main__":
    main()
