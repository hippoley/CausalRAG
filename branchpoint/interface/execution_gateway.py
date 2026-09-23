from __future__ import annotations

import asyncio
import inspect

from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from branchpoint import __version__
from branchpoint.authorization import AuthorizationContext, AuthorizationDenied
from branchpoint.execution import (
    EffectIdentityConflict,
    ExecutionBoundaryError,
    ExecutionInProgress,
    PreviousExecutionFailed,
)
from branchpoint.gateway import (
    ExecutionGateDecision,
    ExecutionGateOutcome,
    ToolExecutionGate,
)
from branchpoint.tools import ToolRegistry


PrincipalResolver = Callable[
    [Request],
    AuthorizationContext | None | Awaitable[AuthorizationContext | None],
]
ApprovalValidator = Callable[
    [Request, ExecutionGateDecision, str, str],
    bool | Awaitable[bool],
]


class GatewayPreviewRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=120)
    arguments: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class GatewayExecuteRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=120)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    effect_id: str = Field(..., min_length=1, max_length=500)
    expected_proposal_hash: Optional[str] = Field(default=None, max_length=128)
    approval_token: Optional[str] = Field(default=None, max_length=4096)

    class Config:
        extra = "forbid"


async def _resolve(value):
    if inspect.isawaitable(value):
        return await value
    return value


def create_execution_gateway_app(
    registry: ToolRegistry,
    *,
    principal_resolver: PrincipalResolver,
    gate: Optional[ToolExecutionGate] = None,
    approval_validator: Optional[ApprovalValidator] = None,
) -> FastAPI:
    """Create a framework-neutral HTTP execution boundary.

    The client may submit tool name, JSON arguments, an effect id, and an opaque
    approval token. It may not submit permissions, canonical risk, or execution
    policy. Principal authority is resolved by trusted server-side application
    code for every preview and again for every execution.
    """

    if registry.execution_ledger is None:
        raise ValueError(
            "HTTP execution gateway requires an execution_ledger so retries "
            "have durable effect identity."
        )
    if not callable(principal_resolver):
        raise TypeError("principal_resolver must be callable")

    execution_gate = gate or ToolExecutionGate(registry)
    app = FastAPI(
        title="Branchpoint Execution Gateway",
        description=(
            "Framework-neutral preview and durable tool execution boundary. "
            "Client payloads cannot grant permissions or override canonical policy."
        ),
        version=__version__,
    )

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": __version__,
            "execution_ledger": type(registry.execution_ledger).__name__,
        }

    async def current_principal(request: Request) -> Optional[AuthorizationContext]:
        try:
            principal = await _resolve(principal_resolver(request))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail={"code": "principal_resolution_failed"},
            ) from exc
        if principal is not None and not isinstance(
            principal,
            AuthorizationContext,
        ):
            raise HTTPException(
                status_code=500,
                detail={"code": "invalid_principal_resolver_result"},
            )
        return principal

    @app.post("/v1/preview")
    async def preview(payload: GatewayPreviewRequest, request: Request):
        principal = await current_principal(request)
        decision = execution_gate.preview(
            payload.tool_name,
            payload.arguments,
            authorization_context=principal,
        )
        return {
            "decision": decision.to_dict(),
            "executes_tool": False,
        }

    @app.post("/v1/execute")
    async def execute(payload: GatewayExecuteRequest, request: Request):
        principal = await current_principal(request)
        decision = execution_gate.preview(
            payload.tool_name,
            payload.arguments,
            authorization_context=principal,
        )

        if (
            payload.expected_proposal_hash is not None
            and payload.expected_proposal_hash != decision.proposal_hash
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "proposal_changed",
                    "message": (
                        "Tool identity or arguments changed after preview; "
                        "request a new preview before execution."
                    ),
                    "decision": decision.to_dict(),
                },
            )

        if decision.outcome is ExecutionGateOutcome.DENY:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "execution_denied",
                    "decision": decision.to_dict(),
                },
            )

        approval_verified = False
        if decision.outcome is ExecutionGateOutcome.REQUIRE_HUMAN:
            if payload.expected_proposal_hash is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "preview_required_for_approval",
                        "decision": decision.to_dict(),
                    },
                )
            if approval_validator is None or not payload.approval_token:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "trusted_approval_required",
                        "decision": decision.to_dict(),
                    },
                )
            try:
                approval_verified = bool(
                    await _resolve(
                        approval_validator(
                            request,
                            decision,
                            payload.effect_id,
                            payload.approval_token,
                        )
                    )
                )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "approval_validation_failed",
                        "decision": decision.to_dict(),
                    },
                ) from exc
            if not approval_verified:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "invalid_approval",
                        "decision": decision.to_dict(),
                    },
                )

        ledger = registry.execution_ledger
        prior_receipt = await asyncio.to_thread(
            ledger.get,
            payload.effect_id,
        )
        replayed = bool(
            prior_receipt is not None
            and prior_receipt.status == "succeeded"
        )

        try:
            result = await registry.execute_async(
                payload.tool_name,
                payload.arguments,
                effect_id=payload.effect_id,
                authorization_context=principal,
            )
        except AuthorizationDenied as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "authorization_changed",
                    "decision": {
                        "allowed": exc.decision.allowed,
                        "reason_code": exc.decision.reason_code,
                        "reason": exc.decision.reason,
                        "missing_permissions": list(
                            exc.decision.missing_permissions
                        ),
                        "policy_id": exc.decision.policy_id,
                    },
                },
            ) from exc
        except EffectIdentityConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "effect_identity_conflict", "message": str(exc)},
            ) from exc
        except ExecutionInProgress as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "effect_in_flight", "message": str(exc)},
            ) from exc
        except PreviousExecutionFailed as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "previous_execution_failed", "message": str(exc)},
            ) from exc
        except ExecutionBoundaryError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "execution_boundary_error", "message": str(exc)},
            ) from exc

        receipt = await asyncio.to_thread(ledger.get, payload.effect_id)
        if receipt is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "missing_execution_receipt",
                    "message": "Execution completed without a durable receipt.",
                },
            )

        return {
            "result": result,
            "execution": {
                "effect_id": receipt.effect_id,
                "effect_hash": receipt.effect_hash,
                "status": receipt.status,
                "replayed": replayed,
                "approval_verified": approval_verified,
                "proposal_hash": decision.proposal_hash,
                "policy_id": decision.policy_id,
            },
        }

    return app
