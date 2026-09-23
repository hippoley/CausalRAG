from __future__ import annotations

import os

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolExecutionGate,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import create_branchpoint_mcp_server


def read_order(order_id):
    return {"order_id": order_id, "status": "open"}


async def restart_service(service, idempotency_key):
    return {
        "service": service,
        "restart_id": idempotency_key,
        "status": "accepted",
    }


registry = ToolRegistry(
    [
        ToolSpec(
            "read_order",
            "Read an order.",
            read_order,
            risk=0.0,
            required_permissions=("order.read",),
        ),
        ToolSpec(
            "restart_service",
            "Restart a service.",
            restart_service,
            risk=0.8,
            reversible=True,
            require_durable_receipt=True,
            idempotency_key_argument="idempotency_key",
            required_permissions=("service.restart",),
        ),
    ],
    execution_ledger=SQLiteExecutionLedger(
        os.environ.get(
            "BRANCHPOINT_MCP_DB",
            "./branchpoint-mcp.sqlite3",
        )
    ),
)


async def principal_resolver(_ctx):
    # Demo-only stdio identity. Production servers should resolve a verified
    # application session/OAuth/JWT/mTLS identity; never trust raw MCP headers
    # as identity because headers are client-supplied input.
    return AuthorizationContext.from_permissions(
        "local:operator",
        ["order.read", "service.restart"],
    )


mcp = create_branchpoint_mcp_server(
    registry,
    principal_resolver=principal_resolver,
    gate=ToolExecutionGate(
        registry,
        auto_execute_max_risk=0.1,
    ),
    name="Branchpoint Execution Boundary",
)


if __name__ == "__main__":
    mcp.run()
