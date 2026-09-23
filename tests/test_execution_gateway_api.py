from __future__ import annotations

from fastapi.testclient import TestClient

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.gateway import ToolExecutionGate
from branchpoint.interface.execution_gateway import create_execution_gateway_app


def _client(tmp_path):
    calls = []

    def read_order(order_id):
        calls.append(("read", order_id))
        return {"order_id": order_id, "status": "open"}

    async def charge_card(amount, idempotency_key):
        calls.append(("charge", amount, idempotency_key))
        return {"charge_id": "ch_http_1", "amount": amount}

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
                "charge_card",
                "Charge a card.",
                charge_card,
                risk=0.9,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("payments.charge",),
            ),
        ],
        execution_ledger=SQLiteExecutionLedger(tmp_path / "gateway.sqlite3"),
    )

    def principal_resolver(request):
        principal = request.headers.get("x-principal")
        if not principal:
            return None
        permissions = [
            value.strip()
            for value in request.headers.get("x-permissions", "").split(",")
            if value.strip()
        ]
        return AuthorizationContext.from_permissions(principal, permissions)

    def approval_validator(_request, decision, token):
        return token == f"approved:{decision.proposal_hash}"

    app = create_execution_gateway_app(
        registry,
        principal_resolver=principal_resolver,
        gate=ToolExecutionGate(registry, auto_execute_max_risk=0.1),
        approval_validator=approval_validator,
    )
    return TestClient(app), registry, calls


def test_http_preview_and_execute_low_risk_call_with_replay(tmp_path):
    client, registry, calls = _client(tmp_path)
    headers = {
        "x-principal": "support:alice",
        "x-permissions": "order.read",
    }

    preview = client.post(
        "/v1/preview",
        headers=headers,
        json={"tool_name": "read_order", "arguments": {"order_id": "O-1"}},
    )
    assert preview.status_code == 200
    decision = preview.json()["decision"]
    assert decision["outcome"] == "allow"

    payload = {
        "tool_name": "read_order",
        "arguments": {"order_id": "O-1"},
        "effect_id": "http-read-1",
        "expected_proposal_hash": decision["proposal_hash"],
    }
    first = client.post("/v1/execute", headers=headers, json=payload)
    second = client.post("/v1/execute", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == {"order_id": "O-1", "status": "open"}
    assert first.json()["execution"]["replayed"] is False
    assert second.json()["execution"]["replayed"] is True
    assert calls == [("read", "O-1")]
    assert registry.execution_ledger.get("http-read-1").status == "succeeded"


def test_client_payload_cannot_grant_itself_permissions(tmp_path):
    client, registry, calls = _client(tmp_path)

    response = client.post(
        "/v1/execute",
        headers={"x-principal": "guest"},
        json={
            "tool_name": "read_order",
            "arguments": {"order_id": "O-1"},
            "effect_id": "self-grant-1",
            "permissions": ["order.read"],
        },
    )

    assert response.status_code == 422
    assert registry.execution_ledger.get("self-grant-1") is None
    assert calls == []


def test_high_risk_call_requires_trusted_approval_then_executes_once(tmp_path):
    client, registry, calls = _client(tmp_path)
    headers = {
        "x-principal": "billing:alice",
        "x-permissions": "payments.charge",
    }
    preview = client.post(
        "/v1/preview",
        headers=headers,
        json={"tool_name": "charge_card", "arguments": {"amount": 25}},
    )
    decision = preview.json()["decision"]
    assert decision["outcome"] == "require_human"
    assert decision["reason_code"] == "risk_threshold"

    base = {
        "tool_name": "charge_card",
        "arguments": {"amount": 25},
        "effect_id": "http-charge-1",
        "expected_proposal_hash": decision["proposal_hash"],
    }
    blocked = client.post("/v1/execute", headers=headers, json=base)
    invalid = client.post(
        "/v1/execute",
        headers=headers,
        json={**base, "approval_token": "wrong"},
    )
    approved = client.post(
        "/v1/execute",
        headers=headers,
        json={
            **base,
            "approval_token": f"approved:{decision['proposal_hash']}",
        },
    )
    replay = client.post(
        "/v1/execute",
        headers=headers,
        json={
            **base,
            "approval_token": f"approved:{decision['proposal_hash']}",
        },
    )

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "trusted_approval_required"
    assert invalid.status_code == 403
    assert invalid.json()["detail"]["code"] == "invalid_approval"
    assert approved.status_code == 200
    assert replay.status_code == 200
    assert approved.json()["execution"]["approval_verified"] is True
    assert replay.json()["execution"]["replayed"] is True
    assert calls == [("charge", 25, "http-charge-1")]
    assert registry.execution_ledger.get("http-charge-1").status == "succeeded"


def test_preview_hash_prevents_approved_payload_from_being_swapped(tmp_path):
    client, registry, calls = _client(tmp_path)
    headers = {
        "x-principal": "billing:alice",
        "x-permissions": "payments.charge",
    }
    preview = client.post(
        "/v1/preview",
        headers=headers,
        json={"tool_name": "charge_card", "arguments": {"amount": 25}},
    ).json()["decision"]

    response = client.post(
        "/v1/execute",
        headers=headers,
        json={
            "tool_name": "charge_card",
            "arguments": {"amount": 2500},
            "effect_id": "swapped-1",
            "expected_proposal_hash": preview["proposal_hash"],
            "approval_token": f"approved:{preview['proposal_hash']}",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "proposal_changed"
    assert registry.execution_ledger.get("swapped-1") is None
    assert calls == []


def test_effect_id_cannot_be_reused_for_different_semantic_call(tmp_path):
    client, registry, calls = _client(tmp_path)
    headers = {
        "x-principal": "support:alice",
        "x-permissions": "order.read",
    }

    first = client.post(
        "/v1/execute",
        headers=headers,
        json={
            "tool_name": "read_order",
            "arguments": {"order_id": "O-1"},
            "effect_id": "shared-effect-1",
        },
    )
    conflict = client.post(
        "/v1/execute",
        headers=headers,
        json={
            "tool_name": "read_order",
            "arguments": {"order_id": "O-2"},
            "effect_id": "shared-effect-1",
        },
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "effect_identity_conflict"
    assert calls == [("read", "O-1")]


def test_gateway_refuses_to_start_without_durable_ledger():
    registry = ToolRegistry([ToolSpec("read", "read", lambda: None)])

    try:
        create_execution_gateway_app(
            registry,
            principal_resolver=lambda _request: None,
        )
    except ValueError as exc:
        assert "requires an execution_ledger" in str(exc)
    else:
        raise AssertionError("gateway started without a durable ledger")
