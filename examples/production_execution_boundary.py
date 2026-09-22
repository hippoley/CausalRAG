from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

from branchpoint import (
    ActionKind,
    AuthorizationContext,
    AuthorizationDenied,
    CandidateAction,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
    decide,
)


class FakeSubmissionService:
    """Tiny external system with downstream idempotency support."""

    def __init__(self) -> None:
        self.session_status = "expired"
        self.submit_calls = 0
        self.seen_keys = {}

    def inspect_submission_state(self):
        return {"session_status": self.session_status}

    def reauthenticate(self, idempotency_key):
        if idempotency_key in self.seen_keys:
            return self.seen_keys[idempotency_key]
        self.session_status = "active"
        result = {"session_status": "active", "reauthenticated": True}
        self.seen_keys[idempotency_key] = result
        return result

    def submit_form(self, form_id, idempotency_key):
        if idempotency_key in self.seen_keys:
            return self.seen_keys[idempotency_key]
        self.submit_calls += 1
        if self.session_status != "active":
            raise RuntimeError("session_expired")
        result = {"submitted": True, "form_id": form_id, "submission_id": "sub-001"}
        self.seen_keys[idempotency_key] = result
        return result


def run_demo(db_path: Optional[str | Path] = None):
    service = FakeSubmissionService()
    if db_path is None:
        db_path = Path(tempfile.mkdtemp(prefix="branchpoint-demo-")) / "effects.sqlite3"
    ledger = SQLiteExecutionLedger(db_path)

    tools = ToolRegistry(
        [
            ToolSpec(
                "inspect_submission_state",
                "Read current submission/session state",
                service.inspect_submission_state,
                cost=0.02,
                reversible=True,
            ),
            ToolSpec(
                "reauthenticate",
                "Refresh the authenticated session",
                service.reauthenticate,
                cost=0.05,
                risk=0.05,
                reversible=True,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("session.reauthenticate",),
            ),
            ToolSpec(
                "submit_form",
                "Create an external submission",
                service.submit_form,
                risk=0.72,
                reversible=False,
                require_durable_receipt=True,
                idempotency_key_argument="idempotency_key",
                required_permissions=("form.submit",),
            ),
        ],
        execution_ledger=ledger,
    )

    proposer_order = [
        CandidateAction(
            ActionKind.INTERVENE,
            "submit_form",
            arguments={"form_id": "F-42"},
            expected_goal_gain=0.95,
            rationale="The form looks ready. Submit now.",
        ),
        CandidateAction(
            ActionKind.OBSERVE,
            "inspect_submission_state",
            expected_goal_gain=0.15,
            expected_information_gain=0.70,
            rationale="Inspect the session before an irreversible submit.",
        ),
    ]
    first = decide(proposer_order, tools=tools)
    inspection = tools.execute(first.selected.name, first.selected.arguments)

    operator = AuthorizationContext.from_permissions(
        "operator:alice",
        ["session.reauthenticate", "form.submit"],
        roles=["operator"],
    )

    reauth = tools.execute(
        "reauthenticate",
        {},
        effect_id="run-1:reauthenticate",
        authorization_context=operator,
    )
    submitted = tools.execute(
        "submit_form",
        {"form_id": "F-42"},
        effect_id="run-1:submit-form-F-42",
        authorization_context=operator,
    )
    replayed = tools.execute(
        "submit_form",
        {"form_id": "F-42"},
        effect_id="run-1:submit-form-F-42",
        authorization_context=operator,
    )

    guest = AuthorizationContext.from_permissions("guest:bob", [])
    guest_denied = False
    try:
        tools.execute(
            "submit_form",
            {"form_id": "F-99"},
            effect_id="run-2:submit-form-F-99",
            authorization_context=guest,
        )
    except AuthorizationDenied:
        guest_denied = True

    return {
        "proposer_first": proposer_order[0].name,
        "runtime_first": first.selected.name,
        "inspection": inspection,
        "reauthentication": reauth,
        "submission": submitted,
        "replayed_submission": replayed,
        "external_submit_calls": service.submit_calls,
        "guest_denied_before_effect": guest_denied,
        "submit_receipt_status": ledger.get("run-1:submit-form-F-42").status,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
