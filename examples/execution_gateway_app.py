from __future__ import annotations

import os

from fastapi import Request

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolExecutionGate,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.interface.execution_gateway import create_execution_gateway_app


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
            "Restart a production service.",
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
            "BRANCHPOINT_GATEWAY_DB",
            "./branchpoint-gateway.sqlite3",
        )
    ),
)


def principal_resolver(request: Request):
    # Demo only. Production must derive identity from a trusted session/JWT,
    # mTLS identity, or headers injected by an authenticated reverse proxy.
    principal = request.headers.get("x-principal")
    if not principal:
        return None
    permissions = [
        value.strip()
        for value in request.headers.get("x-permissions", "").split(",")
        if value.strip()
    ]
    return AuthorizationContext.from_permissions(principal, permissions)


def approval_validator(_request, decision, effect_id, token):
    # Demo only. Production should validate a signed/one-time approval artifact
    # from the application's trusted review system.
    return token == f"approved:{decision.proposal_hash}:{effect_id}"


app = create_execution_gateway_app(
    registry,
    principal_resolver=principal_resolver,
    gate=ToolExecutionGate(
        registry,
        auto_execute_max_risk=0.1,
    ),
    approval_validator=approval_validator,
)
