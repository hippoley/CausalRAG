from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from branchpoint.agent.actions import CandidateAction


DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when a Jev proposal request cannot be completed or validated."""


@dataclass(frozen=True)
class JevProposal:
    selected_name: str
    probabilities: Dict[str, float]
    confidence: float | None
    model: str | None
    usage: Dict[str, Any]
    raw_response: Dict[str, Any]


def _criteria_for(candidate: CandidateAction) -> str:
    rationale = str(candidate.rationale or "").strip()
    if rationale:
        return rationale
    return f"{candidate.kind.value} action: {candidate.name}"


def _validate_candidates(candidates: Sequence[CandidateAction]) -> None:
    if not candidates:
        raise ValueError("At least one candidate action is required.")
    if len(candidates) > 255:
        raise ValueError("Jev Choice supports at most 255 candidate actions.")
    names = [str(candidate.name).strip() for candidate in candidates]
    if any(not name for name in names):
        raise ValueError("Candidate action names must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError("Candidate action names must be unique for Jev Choice mapping.")


def propose_actions_with_jev(
    state: Any,
    candidates: Sequence[CandidateAction],
    *,
    instructions: str = "Which candidate action should be attempted next given the current state?",
    api_key: str | None = None,
    endpoint: str = DEFAULT_JEV_ENDPOINT,
    model: str = DEFAULT_JEV_MODEL,
    timeout: float = 10.0,
) -> JevProposal:
    """Ask Jev for a typed Choice over already-bounded candidate actions.

    This function deliberately returns a *proposal*, not execution authority.
    Feed the reordered candidates into :func:`branchpoint.decide` together with
    canonical ToolSpec policy if the application wants Branchpoint to arbitrate
    the model recommendation before execution.
    """

    _validate_candidates(candidates)
    key = str(api_key or os.getenv("TYPESAFE_API_KEY", "")).strip()
    if not key:
        raise JevError("Missing TypeSafe API key. Set TYPESAFE_API_KEY or pass api_key=.")

    criteria = {candidate.name: _criteria_for(candidate) for candidate in candidates}
    payload = {
        "model": model,
        "state": state,
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise JevError(f"Jev request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise JevError(f"Jev request failed: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JevError("Jev returned invalid JSON.") from exc

    answer = (data.get("answers") or {}).get("next_action")
    if not isinstance(answer, Mapping):
        raise JevError("Jev response is missing answers.next_action.")
    selected = str(answer.get("choice") or "").strip()
    if selected not in criteria:
        raise JevError(f"Jev selected an unknown candidate: {selected!r}")

    probabilities_raw = answer.get("probabilities") or {}
    probabilities: Dict[str, float] = {}
    if isinstance(probabilities_raw, Mapping):
        for name in criteria:
            try:
                probabilities[name] = float(probabilities_raw.get(name, 0.0))
            except (TypeError, ValueError):
                probabilities[name] = 0.0

    confidence_raw = answer.get("confidence")
    try:
        confidence = None if confidence_raw is None else float(confidence_raw)
    except (TypeError, ValueError):
        confidence = None

    return JevProposal(
        selected_name=selected,
        probabilities=probabilities,
        confidence=confidence,
        model=str(data.get("model") or model) if data.get("model") or model else None,
        usage=dict(data.get("usage") or {}),
        raw_response=dict(data),
    )


def reorder_candidates_from_jev(
    candidates: Sequence[CandidateAction],
    proposal: JevProposal,
) -> Tuple[CandidateAction, ...]:
    """Return candidates in Jev probability order while preserving all runtime metadata."""

    _validate_candidates(candidates)
    by_name = {candidate.name: candidate for candidate in candidates}
    return tuple(
        by_name[name]
        for name in sorted(
            by_name,
            key=lambda item: (
                proposal.probabilities.get(item, 0.0),
                item == proposal.selected_name,
            ),
            reverse=True,
        )
    )


def jev_then_branchpoint(
    state: Any,
    candidates: Sequence[CandidateAction],
    *,
    tools=None,
    world_model=None,
    capabilities=None,
    instructions: str = "Which candidate action should be attempted next given the current state?",
    api_key: str | None = None,
    endpoint: str = DEFAULT_JEV_ENDPOINT,
    model: str = DEFAULT_JEV_MODEL,
    timeout: float = 10.0,
):
    """Convenience composition: Jev proposes an order, Branchpoint arbitrates it.

    Import is local to keep this module free of circular imports.
    """

    from branchpoint.decision import decide

    proposal = propose_actions_with_jev(
        state,
        candidates,
        instructions=instructions,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
    )
    ordered = reorder_candidates_from_jev(candidates, proposal)
    decision = decide(
        ordered,
        tools=tools,
        world_model=world_model,
        capabilities=capabilities,
    )
    return proposal, decision
