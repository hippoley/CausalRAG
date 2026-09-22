from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class AuthorizationError(RuntimeError):
    """Base error for execution authorization failures."""


class AuthorizationDenied(AuthorizationError):
    def __init__(self, decision: "AuthorizationDecision") -> None:
        self.decision = decision
        super().__init__(decision.reason)


@dataclass(frozen=True)
class AuthorizationContext:
    principal_id: str
    permissions: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    issued_at: Optional[float] = None
    expires_at: Optional[float] = None

    @classmethod
    def from_permissions(
        cls,
        principal_id: str,
        permissions: Iterable[str],
        *,
        roles: Iterable[str] = (),
        attributes: Optional[Mapping[str, Any]] = None,
        issued_at: Optional[float] = None,
        expires_at: Optional[float] = None,
    ) -> "AuthorizationContext":
        return cls(
            principal_id=str(principal_id),
            permissions=tuple(sorted({str(value) for value in permissions})),
            roles=tuple(sorted({str(value) for value in roles})),
            attributes=dict(attributes or {}),
            issued_at=issued_at,
            expires_at=expires_at,
        )


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    principal_id: Optional[str]
    required_permissions: Tuple[str, ...]
    missing_permissions: Tuple[str, ...] = ()
    reason_code: str = "allowed"
    reason: str = "Authorized"
    policy_id: str = "branchpoint.capability.v1"


class CapabilityAuthorizationPolicy:
    """Small fail-closed permission policy owned by application code.

    A model may propose a tool call, but required permissions come from the
    ToolSpec and cannot be supplied or relaxed by the proposer.
    """

    policy_id = "branchpoint.capability.v1"

    def authorize(
        self,
        context: Optional[AuthorizationContext],
        *,
        tool_name: str,
        required_permissions: Iterable[str],
        arguments: Mapping[str, Any],
    ) -> AuthorizationDecision:
        required = tuple(sorted({str(value) for value in required_permissions if str(value)}))
        if not required:
            return AuthorizationDecision(
                allowed=True,
                principal_id=None if context is None else context.principal_id,
                required_permissions=(),
                policy_id=self.policy_id,
            )

        if context is None or not str(context.principal_id).strip():
            return AuthorizationDecision(
                allowed=False,
                principal_id=None,
                required_permissions=required,
                missing_permissions=required,
                reason_code="missing_principal",
                reason=f"Tool {tool_name!r} requires an authenticated principal.",
                policy_id=self.policy_id,
            )

        now = time.time()
        if context.expires_at is not None and now >= float(context.expires_at):
            return AuthorizationDecision(
                allowed=False,
                principal_id=context.principal_id,
                required_permissions=required,
                missing_permissions=required,
                reason_code="expired_authority",
                reason=f"Authority for principal {context.principal_id!r} has expired.",
                policy_id=self.policy_id,
            )

        held = {str(value) for value in context.permissions}
        missing = tuple(value for value in required if value not in held)
        if missing:
            return AuthorizationDecision(
                allowed=False,
                principal_id=context.principal_id,
                required_permissions=required,
                missing_permissions=missing,
                reason_code="missing_permission",
                reason=(
                    f"Principal {context.principal_id!r} lacks permission(s): "
                    + ", ".join(missing)
                ),
                policy_id=self.policy_id,
            )

        return AuthorizationDecision(
            allowed=True,
            principal_id=context.principal_id,
            required_permissions=required,
            policy_id=self.policy_id,
        )
