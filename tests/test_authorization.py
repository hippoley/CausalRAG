import time

import pytest

from branchpoint import (
    ActionKind,
    AuthorizationContext,
    AuthorizationDenied,
    CandidateAction,
    CausalAgentLoop,
    ToolRegistry,
    ToolSpec,
)


def test_required_permission_fails_closed_without_principal():
    calls = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "unlock_front_door",
                "unlock",
                lambda: calls.append("executed"),
                required_permissions=("home.front_door.unlock",),
            )
        ]
    )

    with pytest.raises(AuthorizationDenied) as exc:
        registry.execute("unlock_front_door", {})

    assert exc.value.decision.reason_code == "missing_principal"
    assert calls == []


def test_principal_with_capability_can_execute():
    registry = ToolRegistry(
        [
            ToolSpec(
                "open_bedroom_window",
                "open",
                lambda percent: {"percent": percent},
                required_permissions=("home.window.open",),
            )
        ]
    )
    context = AuthorizationContext.from_permissions(
        "resident:alice",
        ["home.window.open"],
        roles=["resident"],
    )

    result = registry.execute(
        "open_bedroom_window",
        {"percent": 50},
        authorization_context=context,
    )
    assert result == {"percent": 50}


def test_expired_authority_is_denied_at_execution_time():
    calls = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "deploy",
                "deploy",
                lambda: calls.append("executed"),
                required_permissions=("prod.deploy",),
            )
        ]
    )
    context = AuthorizationContext.from_permissions(
        "engineer:1",
        ["prod.deploy"],
        expires_at=time.time() - 1,
    )

    with pytest.raises(AuthorizationDenied) as exc:
        registry.execute("deploy", {}, authorization_context=context)
    assert exc.value.decision.reason_code == "expired_authority"
    assert calls == []


def test_missing_one_of_multiple_permissions_is_explicit():
    registry = ToolRegistry(
        [
            ToolSpec(
                "refund",
                "refund",
                lambda: {"ok": True},
                required_permissions=("refund.write", "customer.read"),
            )
        ]
    )
    context = AuthorizationContext.from_permissions(
        "agent:7",
        ["customer.read"],
    )

    with pytest.raises(AuthorizationDenied) as exc:
        registry.execute("refund", {}, authorization_context=context)
    assert exc.value.decision.missing_permissions == ("refund.write",)


class _DoorReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    ActionKind.INTERVENE,
                    "unlock_front_door",
                    expected_goal_gain=1.0,
                )
            ]
        return [CandidateAction(ActionKind.STOP, "stop", rationale="done")]

    def uncertainty(self, state, world_model):
        return None


def test_agent_resolves_dynamic_authority_immediately_before_tool_execution():
    contexts = []
    calls = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "unlock_front_door",
                "unlock",
                lambda: calls.append("executed") or {"unlocked": True},
                required_permissions=("home.front_door.unlock",),
            )
        ]
    )

    def resolver(state, action):
        contexts.append((state.step, action.name))
        return AuthorizationContext.from_permissions(
            "resident:bob",
            ["home.front_door.unlock"],
        )

    loop = CausalAgentLoop(
        reasoner=_DoorReasoner(),
        tools=registry,
        authorization_resolver=resolver,
    )
    state = loop.run("unlock door", max_steps=2)

    assert contexts == [(0, "unlock_front_door")]
    assert calls == ["executed"]
    assert state.observations[0].result == {"unlocked": True}


def test_model_supplied_arguments_cannot_grant_permission():
    calls = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "unlock_front_door",
                "unlock",
                lambda **kwargs: calls.append(kwargs),
                required_permissions=("home.front_door.unlock",),
            )
        ]
    )
    context = AuthorizationContext.from_permissions("guest:1", [])

    with pytest.raises(AuthorizationDenied):
        registry.execute(
            "unlock_front_door",
            {"permission": "home.front_door.unlock", "authorized": True},
            authorization_context=context,
        )
    assert calls == []
