from __future__ import annotations

import asyncio

from mcp import Client
from mcp.client import ClientRequestContext
from mcp_types import ElicitRequestParams, ElicitResult

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolExecutionGate,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import create_branchpoint_mcp_server


def _fixture(tmp_path, *, high_risk: bool = True):
    calls = []
    current = {
        "principal": AuthorizationContext.from_permissions(
            "operator:alice",
            ["order.read", "service.restart"],
        )
    }

    def read_order(order_id):
        calls.append(("read", order_id))
        return {"order_id": order_id, "status": "open"}

    async def restart_service(service, idempotency_key):
        await asyncio.sleep(0)
        calls.append(("restart", service, idempotency_key))
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
                risk=0.8 if high_risk else 0.0,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("service.restart",),
            ),
        ],
        execution_ledger=SQLiteExecutionLedger(tmp_path / "mcp.sqlite3"),
    )

    async def principal_resolver(_ctx):
        return current["principal"]

    server = create_branchpoint_mcp_server(
        registry,
        principal_resolver=principal_resolver,
        gate=ToolExecutionGate(registry, auto_execute_max_risk=0.1),
        name="Branchpoint Test",
    )
    return server, registry, calls, current


def test_mcp_schema_hides_principal_and_human_approval_from_model(tmp_path):
    server, _registry, _calls, _current = _fixture(tmp_path)

    async def scenario():
        async with Client(server) as client:
            listed = await client.list_tools()
            return client.session.protocol_version, listed

    protocol, listed = asyncio.run(scenario())
    tools = {tool.name: tool for tool in listed.tools}

    assert protocol == "2026-07-28"
    assert set(tools) == {"branchpoint_preview", "branchpoint_execute"}

    preview_props = set(tools["branchpoint_preview"].input_schema["properties"])
    execute_props = set(tools["branchpoint_execute"].input_schema["properties"])

    assert preview_props == {"tool_name", "arguments"}
    assert execute_props == {"tool_name", "arguments", "effect_id"}
    assert "principal" not in execute_props
    assert "permissions" not in execute_props
    assert "approval" not in execute_props


def test_low_risk_mcp_execution_never_elicits_and_replays_once(tmp_path):
    server, registry, calls, _current = _fixture(tmp_path)
    elicitation_count = 0

    async def never_elicit(
        _context: ClientRequestContext,
        _params: ElicitRequestParams,
    ) -> ElicitResult:
        nonlocal elicitation_count
        elicitation_count += 1
        raise AssertionError("low-risk authorized execution must not elicit")

    async def scenario():
        async with Client(
            server,
            elicitation_callback=never_elicit,
        ) as client:
            preview = await client.call_tool(
                "branchpoint_preview",
                {
                    "tool_name": "read_order",
                    "arguments": {"order_id": "O-1"},
                },
            )
            first = await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "read_order",
                    "arguments": {"order_id": "O-1"},
                    "effect_id": "mcp-read-1",
                },
            )
            replay = await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "read_order",
                    "arguments": {"order_id": "O-1"},
                    "effect_id": "mcp-read-1",
                },
            )
            return preview, first, replay

    preview, first, replay = asyncio.run(scenario())

    assert preview.is_error is not True
    assert preview.structured_content["decision"]["outcome"] == "allow"
    assert first.structured_content["executed"] is True
    assert first.structured_content["execution"]["replayed"] is False
    assert replay.structured_content["execution"]["replayed"] is True
    assert calls == [("read", "O-1")]
    assert elicitation_count == 0
    assert registry.execution_ledger.get("mcp-read-1").status == "succeeded"


def test_high_risk_mcp_execution_uses_protocol_native_elicitation(tmp_path):
    server, registry, calls, _current = _fixture(tmp_path)
    prompts = []

    async def approve(
        _context: ClientRequestContext,
        params: ElicitRequestParams,
    ) -> ElicitResult:
        prompts.append(params.message)
        return ElicitResult(
            action="accept",
            content={"approve": True},
        )

    async def scenario():
        async with Client(server, elicitation_callback=approve) as client:
            assert client.session.protocol_version == "2026-07-28"
            return await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "restart_service",
                    "arguments": {"service": "api"},
                    "effect_id": "mcp-restart-1",
                },
            )

    result = asyncio.run(scenario())

    assert result.is_error is not True
    assert result.structured_content["executed"] is True
    assert result.structured_content["decision"]["outcome"] == "require_human"
    assert result.structured_content["execution"]["status"] == "succeeded"
    assert calls == [("restart", "api", "mcp-restart-1")]
    assert len(prompts) == 1
    assert "restart_service" in prompts[0]
    assert "mcp-restart-1" in prompts[0]
    # Arguments are deliberately not copied into the human prompt.
    assert '"service"' not in prompts[0]
    assert registry.execution_ledger.get("mcp-restart-1").status == "succeeded"


def test_high_risk_mcp_rejection_creates_no_receipt(tmp_path):
    server, registry, calls, _current = _fixture(tmp_path)

    async def reject(
        _context: ClientRequestContext,
        _params: ElicitRequestParams,
    ) -> ElicitResult:
        return ElicitResult(
            action="accept",
            content={"approve": False},
        )

    async def scenario():
        async with Client(server, elicitation_callback=reject) as client:
            return await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "restart_service",
                    "arguments": {"service": "api"},
                    "effect_id": "mcp-restart-rejected-1",
                },
            )

    result = asyncio.run(scenario())

    assert result.is_error is not True
    assert result.structured_content["executed"] is False
    assert result.structured_content["reason"] == "human_rejected"
    assert registry.execution_ledger.get("mcp-restart-rejected-1") is None
    assert calls == []


def test_missing_permission_does_not_elicit_or_execute(tmp_path):
    server, registry, calls, current = _fixture(tmp_path)
    current["principal"] = AuthorizationContext.from_permissions(
        "guest:bob",
        [],
    )
    elicitation_count = 0

    async def never_elicit(
        _context: ClientRequestContext,
        _params: ElicitRequestParams,
    ) -> ElicitResult:
        nonlocal elicitation_count
        elicitation_count += 1
        raise AssertionError("denied authority must not be converted into human approval")

    async def scenario():
        async with Client(
            server,
            elicitation_callback=never_elicit,
        ) as client:
            return await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "restart_service",
                    "arguments": {"service": "api"},
                    "effect_id": "mcp-denied-1",
                },
            )

    result = asyncio.run(scenario())

    assert result.is_error is not True
    assert result.structured_content["executed"] is False
    assert result.structured_content["reason"] == "execution_denied"
    assert result.structured_content["decision"]["reason_code"] == "missing_permission"
    assert registry.execution_ledger.get("mcp-denied-1") is None
    assert calls == []
    assert elicitation_count == 0


def test_mcp_effect_identity_conflict_fails_closed(tmp_path):
    server, registry, calls, _current = _fixture(tmp_path)

    async def scenario():
        async with Client(server) as client:
            first = await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "read_order",
                    "arguments": {"order_id": "O-1"},
                    "effect_id": "mcp-shared-effect-1",
                },
            )
            conflict = await client.call_tool(
                "branchpoint_execute",
                {
                    "tool_name": "read_order",
                    "arguments": {"order_id": "O-2"},
                    "effect_id": "mcp-shared-effect-1",
                },
            )
            return first, conflict

    first, conflict = asyncio.run(scenario())

    assert first.is_error is not True
    assert conflict.is_error is True
    assert calls == [("read", "O-1")]
    assert registry.execution_ledger.get("mcp-shared-effect-1").status == "succeeded"


def test_mcp_principal_resolver_must_return_trusted_context(tmp_path):
    registry = ToolRegistry(
        [ToolSpec("read", "read", lambda: {"ok": True})],
        execution_ledger=SQLiteExecutionLedger(tmp_path / "bad-principal.sqlite3"),
    )

    async def bad_resolver(_ctx):
        return {"principal_id": "forged", "permissions": ["*"]}

    server = create_branchpoint_mcp_server(
        registry,
        principal_resolver=bad_resolver,
    )

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(
                "branchpoint_preview",
                {"tool_name": "read", "arguments": {}},
            )

    result = asyncio.run(scenario())

    assert result.is_error is True
    assert registry.execution_ledger.get("anything") is None
