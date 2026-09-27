from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ToolCallRequest

from branchpoint import SQLiteExecutionLedger, ToolRegistry, ToolSpec
from branchpoint.integrations import LangChainBranchpointMiddleware


def main() -> None:
    calls = []
    ledger = SQLiteExecutionLedger(
        Path(tempfile.mkdtemp(prefix="branchpoint-langchain-"))
        / "effects.sqlite3"
    )

    def charge(amount, idempotency_key):
        calls.append((amount, idempotency_key))
        return {"charge_id": "ch_lc_demo_1", "amount": amount}

    registry = ToolRegistry(
        [
            ToolSpec(
                "charge",
                "Charge exactly once.",
                charge,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
            )
        ],
        execution_ledger=ledger,
    )
    middleware = LangChainBranchpointMiddleware(registry)
    request = ToolCallRequest(
        tool_call={
            "name": "charge",
            "args": {"amount": 25},
            "id": "call-demo-1",
            "type": "tool_call",
        },
        tool=None,
        state={"messages": []},
        runtime=SimpleNamespace(context=None),
    )

    def should_not_run(_request):
        raise RuntimeError("LangChain handler should be bypassed")

    first = middleware.wrap_tool_call(request, should_not_run)
    replay = middleware.wrap_tool_call(request, should_not_run)

    effect_id = "langchain:charge:call-demo-1"
    print("content:", first.content)
    print("external_calls:", len(calls))
    print("effect_id:", effect_id)
    print("receipt_status:", ledger.get(effect_id).status)
    print("first_replayed:", first.artifact["branchpoint"]["replayed"])
    print("replay_replayed:", replay.artifact["branchpoint"]["replayed"])


if __name__ == "__main__":
    main()
