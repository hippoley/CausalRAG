import asyncio
import inspect

from typing import Any, Awaitable, Callable, Mapping, Optional

from branchpoint.authorization import AuthorizationContext
from branchpoint.gateway import (
    ExecutionGateOutcome,
    ToolExecutionGate,
)
from branchpoint.tools import ToolRegistry


MCPPrincipalResolver = Callable[
    [Any],
    AuthorizationContext | None | Awaitable[AuthorizationContext | None],
]


async def _resolve(value):
    if inspect.isawaitable(value):
        return await value
    return value


def create_branchpoint_mcp_server(
    registry: ToolRegistry,
    *,
    principal_resolver: MCPPrincipalResolver,
    gate: Optional[ToolExecutionGate] = None,
    name: str = "Branchpoint",
    request_state_security: Any = None,
):
    """Expose Branchpoint preview + execution as protocol-native MCP tools.

    Principal identity is supplied by trusted application code through a hidden
    Resolve dependency. High-risk or irreversible calls use MCP v2 elicitation,
    so human approval is never a model-visible tool argument.
    """

    if registry.execution_ledger is None:
        raise ValueError(
            "MCP execution boundary requires an execution_ledger so retries "
            "have durable effect identity."
        )
    if not callable(principal_resolver):
        raise TypeError("principal_resolver must be callable")

    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver import (
            Context,
            Elicit,
            Resolve,
            RequestStateSecurity,
        )
        from pydantic import BaseModel, Field
        from typing import Annotated
    except ImportError as exc:
        raise RuntimeError(
            "MCP integration requires the optional SDK. "
            "Install this repository with: pip install -e ".[mcp]""
        ) from exc

    execution_gate = gate or ToolExecutionGate(registry)
    state_security = (
        request_state_security
        if request_state_security is not None
        else RequestStateSecurity.ephemeral()
    )
    mcp = MCPServer(
        name=name,
        instructions=(
            "Branchpoint consequence boundary. Preview tool calls before "
            "execution; execution always rechecks policy and uses a durable "
            "effect receipt."
        ),
        request_state_security=state_security,
    )

    class ResolvedPrincipal(BaseModel):
        principal_id: Optional[str] = None
        permissions: list[str] = Field(default_factory=list)
        roles: list[str] = Field(default_factory=list)
        attributes: dict[str, Any] = Field(default_factory=dict)
        issued_at: Optional[float] = None
        expires_at: Optional[float] = None

    class ConfirmExecution(BaseModel):
        approve: bool = Field(
            description="Approve this exact Branchpoint execution?"
        )

    async def resolve_principal(ctx: Context) -> ResolvedPrincipal:
        try:
            principal = await _resolve(principal_resolver(ctx))
        except Exception as exc:
            raise RuntimeError("trusted principal resolution failed") from exc
        if principal is None:
            return ResolvedPrincipal()
        if not isinstance(principal, AuthorizationContext):
            raise RuntimeError(
                "principal_resolver must return AuthorizationContext or None"
            )
        return ResolvedPrincipal(
            principal_id=principal.principal_id,
            permissions=list(principal.permissions),
            roles=list(principal.roles),
            attributes=dict(principal.attributes),
            issued_at=principal.issued_at,
            expires_at=principal.expires_at,
        )

    def authorization_context(
        principal: ResolvedPrincipal,
    ) -> Optional[AuthorizationContext]:
        if not principal.principal_id:
            return None
        return AuthorizationContext(
            principal_id=principal.principal_id,
            permissions=tuple(principal.permissions),
            roles=tuple(principal.roles),
            attributes=dict(principal.attributes),
            issued_at=principal.issued_at,
            expires_at=principal.expires_at,
        )

    async def resolve_approval(
        tool_name: str,
        arguments: dict[str, Any],
        effect_id: str,
        principal: Annotated[
            ResolvedPrincipal,
            Resolve(resolve_principal),
        ],
    ) -> ConfirmExecution | Elicit[ConfirmExecution]:
        decision = execution_gate.preview(
            tool_name,
            arguments,
            authorization_context=authorization_context(principal),
        )
        if decision.outcome is ExecutionGateOutcome.ALLOW:
            return ConfirmExecution(approve=True)
        if decision.outcome is ExecutionGateOutcome.DENY:
            return ConfirmExecution(approve=False)

        risk = "unknown" if decision.risk is None else f"{decision.risk:g}"
        return Elicit(
            (
                f"Approve Branchpoint execution of {tool_name!r}? "
                f"risk={risk}; reversible={decision.reversible}; "
                f"effect_id={effect_id!r}; reason={decision.reason_code}."
            ),
            ConfirmExecution,
        )

    @mcp.tool()
    async def branchpoint_preview(
        tool_name: str,
        arguments: dict[str, Any],
        principal: Annotated[
            ResolvedPrincipal,
            Resolve(resolve_principal),
        ],
    ) -> dict[str, Any]:
        """Preview canonical Branchpoint policy without executing a tool."""

        decision = execution_gate.preview(
            tool_name,
            arguments,
            authorization_context=authorization_context(principal),
        )
        return {
            "decision": decision.to_dict(),
            "executes_tool": False,
        }

    @mcp.tool()
    async def branchpoint_execute(
        tool_name: str,
        arguments: dict[str, Any],
        effect_id: str,
        principal: Annotated[
            ResolvedPrincipal,
            Resolve(resolve_principal),
        ],
        approval: Annotated[
            ConfirmExecution,
            Resolve(resolve_approval),
        ],
    ) -> dict[str, Any]:
        """Execute through Branchpoint authorization and durable receipts."""

        auth = authorization_context(principal)
        decision = execution_gate.preview(
            tool_name,
            arguments,
            authorization_context=auth,
        )
        if decision.outcome is ExecutionGateOutcome.DENY:
            return {
                "executed": False,
                "decision": decision.to_dict(),
                "reason": "execution_denied",
            }
        if (
            decision.outcome is ExecutionGateOutcome.REQUIRE_HUMAN
            and not approval.approve
        ):
            return {
                "executed": False,
                "decision": decision.to_dict(),
                "reason": "human_rejected",
            }

        ledger = registry.execution_ledger
        prior = await asyncio.to_thread(ledger.get, effect_id)
        replayed = bool(prior is not None and prior.status == "succeeded")

        result = await registry.execute_async(
            tool_name,
            arguments,
            effect_id=effect_id,
            authorization_context=auth,
        )
        receipt = await asyncio.to_thread(ledger.get, effect_id)
        if receipt is None:
            raise RuntimeError(
                "Branchpoint execution completed without a durable receipt"
            )

        return {
            "executed": True,
            "result": result,
            "decision": decision.to_dict(),
            "execution": {
                "effect_id": receipt.effect_id,
                "effect_hash": receipt.effect_hash,
                "status": receipt.status,
                "replayed": replayed,
            },
        }

    return mcp
