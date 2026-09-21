from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from branchpoint import (
    ActionKind,
    CandidateAction,
    CausalWorldModel,
    ToolSpec,
    __version__,
    decide,
)
from branchpoint.agent import RuntimeCapabilities
from branchpoint.generator.llm_interface import LLMInterface
from branchpoint.probe import (
    ProbeRunConfig,
    SESSION_MANAGER,
    available_probe_config,
    run_probe_episode,
    run_probe_comparison,
    run_probe_ladder,
    sse_stream,
)


app = FastAPI(
    title="Branchpoint Playable Probe",
    description="Human-in-the-loop research probe for causal-runtime and proposer-model ablations.",
    version=__version__,
)


class ProbeRunRequest(BaseModel):
    scenario: str = Field("hvac_hidden_world", min_length=1, max_length=120)
    hidden_hypothesis: str = Field("H2", min_length=1, max_length=40)
    outcome_mode: Literal["deterministic", "stochastic"] = "stochastic"
    seed: int = 0
    max_steps: int = Field(6, ge=1, le=32)
    max_probes: int = Field(3, ge=0, le=16)
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    goal: Optional[str] = Field(default=None, max_length=4000)
    proposer_family: Literal["deterministic", "small", "frontier"] = "deterministic"
    provider: Optional[Literal["openai", "anthropic", "local"]] = None
    model: Optional[str] = Field(default=None, max_length=200)
    capabilities: Dict[str, bool] = Field(
        default_factory=lambda: RuntimeCapabilities.full().to_dict()
    )


class DecisionCandidateInput(BaseModel):
    kind: Literal["observe", "retrieve", "ask", "intervene", "wait", "stop"]
    name: str = Field(min_length=1, max_length=120)
    expected_goal_gain: float = Field(0.0, ge=0.0, le=1.0)
    expected_information_gain: float = Field(0.0, ge=0.0, le=1.0)
    cost: float = Field(0.0, ge=0.0, le=1000.0)
    risk: float = Field(0.0, ge=0.0, le=1000.0)
    irreversibility: float = Field(0.0, ge=0.0, le=1000.0)
    rationale: str = Field("", max_length=1000)
    tests_hypotheses: List[str] = Field(default_factory=list)


class DecisionToolInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    cost: float = Field(0.0, ge=0.0, le=1000.0)
    risk: float = Field(0.0, ge=0.0, le=1000.0)
    reversible: bool = True


class DecisionHypothesisInput(BaseModel):
    hypothesis_id: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=1, max_length=1000)
    probability: float = Field(gt=0.0, le=1000000.0)


class OneShotDecisionRequest(BaseModel):
    candidates: List[DecisionCandidateInput] = Field(min_length=1, max_length=20)
    tools: List[DecisionToolInput] = Field(default_factory=list, max_length=20)
    hypotheses: List[DecisionHypothesisInput] = Field(default_factory=list, max_length=20)


class DecisionRequest(BaseModel):
    action: Literal["approve", "choose", "replan"]
    candidate_index: Optional[int] = Field(default=None, ge=0)


class CounterfactualRequest(BaseModel):
    candidate_index: int = Field(ge=0)


class HumanHypothesisRequest(BaseModel):
    hypothesis_id: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=1, max_length=1000)
    probability: float = Field(0.2, ge=0.01, le=0.4)
    rationale: str = Field(
        "Added by human operator during Playable Probe.",
        max_length=1000,
    )


class ModelTestRequest(BaseModel):
    provider: Literal["openai", "anthropic", "local"]
    model: str = Field(min_length=1, max_length=200)


class OperatorMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    replan: bool = True


class AccessRequest(BaseModel):
    token: str = Field(min_length=1, max_length=500)


_AUTH_COOKIE = "branchpoint_probe_access"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _auth_required() -> bool:
    return _env_flag("BRANCHPOINT_REQUIRE_PROBE_AUTH", False)


def _access_token() -> str:
    return str(os.getenv("BRANCHPOINT_PROBE_ACCESS_TOKEN", "")).strip()


def _session_credential(token: str) -> str:
    return hashlib.sha256(("branchpoint-probe-session:" + token).encode("utf-8")).hexdigest()


def _authenticated(request: Request) -> bool:
    if not _auth_required():
        return True
    configured = _access_token()
    if not configured:
        return False
    expected_session = _session_credential(configured)
    cookie = str(request.cookies.get(_AUTH_COOKIE, ""))
    if cookie and secrets.compare_digest(cookie, expected_session):
        return True
    authorization = str(request.headers.get("authorization", ""))
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
        return bool(supplied) and (
            secrets.compare_digest(supplied, configured)
            or secrets.compare_digest(supplied, expected_session)
        )
    return False


def _require_auth(request: Request) -> None:
    if not _auth_required():
        return
    if not _access_token():
        raise HTTPException(
            status_code=503,
            detail="Probe auth is required but BRANCHPOINT_PROBE_ACCESS_TOKEN is not configured.",
        )
    if not _authenticated(request):
        raise HTTPException(status_code=401, detail="Owner access required for external model use.")


def _require_payload_access(request: Request, payload: ProbeRunRequest) -> None:
    if payload.proposer_family != "deterministic":
        _require_auth(request)


def _require_session_access(request: Request, session) -> None:
    if getattr(session.config, "proposer_family", "deterministic") != "deterministic":
        _require_auth(request)


def _config(payload: ProbeRunRequest) -> ProbeRunConfig:
    return ProbeRunConfig(
        scenario=payload.scenario,
        hidden_hypothesis=payload.hidden_hypothesis,
        outcome_mode=payload.outcome_mode,
        seed=payload.seed,
        max_steps=payload.max_steps,
        max_probes=payload.max_probes,
        confidence_threshold=payload.confidence_threshold,
        goal=payload.goal,
        proposer_family=payload.proposer_family,
        provider=payload.provider,
        model=payload.model,
        capabilities=payload.capabilities,
    )


def _session(session_id: str):
    try:
        return SESSION_MANAGER.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="probe session not found") from exc


@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__, "surface": "playable_probe"}


@app.get("/api/config")
def probe_config(request: Request):
    config = available_probe_config()
    config["access"] = {
        "required_for_external_models": _auth_required(),
        "configured": (not _auth_required()) or bool(_access_token()),
        "authenticated": _authenticated(request),
    }
    return config


@app.get("/api/access/status")
def access_status(request: Request):
    return {
        "required_for_external_models": _auth_required(),
        "configured": (not _auth_required()) or bool(_access_token()),
        "authenticated": _authenticated(request),
    }


@app.post("/api/access")
def unlock_probe_access(payload: AccessRequest, request: Request, response: Response):
    if not _auth_required():
        return {"authenticated": True, "required_for_external_models": False}
    configured = _access_token()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="BRANCHPOINT_PROBE_ACCESS_TOKEN is not configured.",
        )
    if not secrets.compare_digest(payload.token, configured):
        raise HTTPException(status_code=401, detail="Invalid access token.")
    response.set_cookie(
        _AUTH_COOKIE,
        _session_credential(configured),
        httponly=True,
        secure=_env_flag("BRANCHPOINT_PROBE_COOKIE_SECURE", False),
        samesite="strict",
        path="/",
    )
    return {"authenticated": True, "required_for_external_models": True}


@app.post("/api/access/logout")
def logout_probe_access(response: Response):
    response.delete_cookie(_AUTH_COOKIE, path="/")
    return {"authenticated": False}


def _candidate_payload(candidate: CandidateAction, index: int):
    return {
        "index": int(index),
        "kind": candidate.kind.value,
        "name": candidate.name,
        "expected_goal_gain": float(candidate.expected_goal_gain),
        "expected_information_gain": float(candidate.expected_information_gain),
        "cost": float(candidate.cost),
        "risk": float(candidate.risk),
        "irreversibility": float(candidate.irreversibility),
        "rationale": candidate.rationale,
        "tests_hypotheses": list(candidate.tests_hypotheses),
    }


@app.post("/api/decide")
def one_shot_decision(payload: OneShotDecisionRequest):
    """Arbitrate one bounded decision without starting an agent session."""
    candidate_names = [row.name for row in payload.candidates]
    if len(candidate_names) != len(set(candidate_names)):
        raise HTTPException(status_code=400, detail="candidate names must be unique")
    tool_names = [row.name for row in payload.tools]
    if len(tool_names) != len(set(tool_names)):
        raise HTTPException(status_code=400, detail="tool names must be unique")

    candidates = [
        CandidateAction(
            kind=ActionKind(row.kind),
            name=row.name,
            expected_goal_gain=row.expected_goal_gain,
            expected_information_gain=row.expected_information_gain,
            cost=row.cost,
            risk=row.risk,
            irreversibility=row.irreversibility,
            rationale=row.rationale,
            tests_hypotheses=list(row.tests_hypotheses),
        )
        for row in payload.candidates
    ]
    tools = [
        ToolSpec(
            name=row.name,
            description=row.description or row.name,
            handler=lambda **_kwargs: None,
            cost=row.cost,
            risk=row.risk,
            reversible=row.reversible,
        )
        for row in payload.tools
    ]

    world = None
    if payload.hypotheses:
        total = sum(float(row.probability) for row in payload.hypotheses)
        if total <= 0.0:
            raise HTTPException(status_code=400, detail="hypothesis probability mass must be positive")
        world = CausalWorldModel()
        seen_hypotheses = set()
        for row in payload.hypotheses:
            if row.hypothesis_id in seen_hypotheses:
                raise HTTPException(status_code=400, detail="hypothesis ids must be unique")
            seen_hypotheses.add(row.hypothesis_id)
            world.upsert_hypothesis(
                row.hypothesis_id,
                row.statement,
                probability=float(row.probability) / total,
            )

    result = decide(candidates, tools=tools or None, world_model=world)
    tool_by_name = {row.name: row for row in payload.tools}
    candidate_by_name = {row.name: row for row in payload.candidates}

    ranking = []
    for rank, (action, score) in enumerate(zip(result.ranked_actions, result.scores), start=1):
        submitted = candidate_by_name[action.name]
        registered = tool_by_name.get(action.name)
        canonical_overrides = []
        if registered is not None:
            if float(registered.cost) > float(submitted.cost):
                canonical_overrides.append("cost")
            if float(registered.risk) > float(submitted.risk):
                canonical_overrides.append("risk")
            if not registered.reversible and float(submitted.irreversibility) < 1.0:
                canonical_overrides.append("irreversibility")
        ranking.append(
            {
                "rank": rank,
                "candidate": _candidate_payload(action, score.candidate_index),
                "score": {
                    "total_utility": score.total_utility,
                    "goal_gain": score.goal_gain,
                    "information_gain": score.information_gain,
                    "information_source": score.information_source,
                    "model_information_gain": score.model_information_gain,
                    "discrimination_score": score.discrimination_score,
                    "bayesian_information_gain": score.bayesian_information_gain,
                    "cost": score.cost,
                    "risk": score.risk,
                    "irreversibility": score.irreversibility,
                    "decision_value": score.decision_value,
                    "decision_value_source": score.decision_value_source,
                    "expected_value_of_sample_information": score.expected_value_of_sample_information,
                    "net_value_of_sampling": score.net_value_of_sampling,
                },
                "registered_tool": (
                    {
                        "cost": registered.cost,
                        "risk": registered.risk,
                        "reversible": registered.reversible,
                    }
                    if registered is not None
                    else None
                ),
                "canonical_overrides": canonical_overrides,
            }
        )

    proposer_index = 0
    selected_index = next(
        row["candidate"]["index"]
        for row in ranking
        if row["candidate"]["name"] == result.selected.name
        and row["candidate"]["kind"] == result.selected.kind.value
    )
    proposer_row = next(row for row in ranking if row["candidate"]["index"] == proposer_index)
    selected_row = next(row for row in ranking if row["candidate"]["index"] == selected_index)

    reasons = []
    if result.changed_proposer_order:
        reasons.append(
            {
                "code": "higher_runtime_utility",
                "delta": (
                    selected_row["score"]["total_utility"]
                    - proposer_row["score"]["total_utility"]
                ),
            }
        )
    if proposer_row["canonical_overrides"]:
        reasons.append(
            {
                "code": "canonical_tool_policy",
                "fields": proposer_row["canonical_overrides"],
            }
        )
    if selected_row["score"]["information_source"] not in {"model_estimate", "unanchored_model_estimate"}:
        reasons.append(
            {
                "code": "runtime_information_source",
                "source": selected_row["score"]["information_source"],
            }
        )

    return {
        "proposer_first": _candidate_payload(result.proposer_first, 0),
        "selected": _candidate_payload(result.selected, selected_index),
        "changed_proposer_order": result.changed_proposer_order,
        "ranking": ranking,
        "reasons": reasons,
        "hypotheses": (
            [
                {
                    "id": row.hypothesis_id,
                    "statement": row.statement,
                    "probability": row.probability,
                }
                for row in world.hypotheses(include_rejected=False)
            ]
            if world is not None
            else []
        ),
    }


@app.post("/api/run")
def run_probe(payload: ProbeRunRequest, request: Request):
    """One-shot compatibility endpoint used by scripts and CI."""
    _require_payload_access(request, payload)
    try:
        return run_probe_episode(_config(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/compare")
def compare_probe(payload: ProbeRunRequest, request: Request):
    """Run vanilla and causal control planes on the same frozen task inputs."""
    _require_payload_access(request, payload)
    try:
        return run_probe_comparison(_config(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ladder")
def ladder_probe(payload: ProbeRunRequest, request: Request):
    """Run cumulative causal-runtime capability profiles on paired task inputs."""
    _require_payload_access(request, payload)
    try:
        return run_probe_ladder(_config(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions")
def create_session(payload: ProbeRunRequest, request: Request):
    _require_payload_access(request, payload)
    try:
        session = SESSION_MANAGER.create(_config(payload))
        return session.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    session = _session(session_id)
    _require_session_access(request, session)
    return session.snapshot()


@app.get("/api/sessions/{session_id}/export")
def export_session(session_id: str, request: Request):
    """Export a self-contained session for offline replay and review."""
    session = _session(session_id)
    _require_session_access(request, session)
    return session.export_payload()


@app.get("/api/sessions/{session_id}/steps/{step}")
def get_step_context(session_id: str, step: int, request: Request):
    try:
        session = _session(session_id)
        _require_session_access(request, session)
        return session.step_context(step)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="step not found") from exc


@app.post("/api/sessions/{session_id}/steps/{step}/counterfactual")
def run_step_counterfactual(
    session_id: str,
    step: int,
    payload: CounterfactualRequest,
    request: Request,
):
    try:
        session = _session(session_id)
        _require_session_access(request, session)
        return session.run_counterfactual(step, payload.candidate_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="step not found") from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/events")
def session_events(session_id: str, request: Request):
    session = _session(session_id)
    _require_session_access(request, session)
    return StreamingResponse(
        sse_stream(session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sessions/{session_id}/decision")
def resolve_session_decision(session_id: str, payload: DecisionRequest, request: Request):
    session = _session(session_id)
    _require_session_access(request, session)
    try:
        session.resolve_decision(payload.action, payload.candidate_index)
        return session.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/hypotheses")
def add_human_hypothesis(session_id: str, payload: HumanHypothesisRequest, request: Request):
    session = _session(session_id)
    _require_session_access(request, session)
    try:
        hypothesis = session.add_hypothesis(
            payload.hypothesis_id,
            payload.statement,
            probability=payload.probability,
            rationale=payload.rationale,
        )
        return {"hypothesis": hypothesis, "session": session.snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/messages")
def add_operator_message(session_id: str, payload: OperatorMessageRequest, request: Request):
    session = _session(session_id)
    _require_session_access(request, session)
    try:
        message = session.add_operator_message(payload.message, replan=payload.replan)
        return {"message": message, "session": session.snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/model/test")
def test_model_connection(payload: ModelTestRequest, request: Request):
    """User-triggered live model check using server-side credentials only."""
    _require_auth(request)
    try:
        llm = LLMInterface(model=payload.model, provider=payload.provider)
        text = llm.generate(
            "Reply with exactly BRANCHPOINT_MODEL_OK and nothing else.",
            temperature=0.0,
            max_tokens=32,
        )
        ok = str(text).strip() == "BRANCHPOINT_MODEL_OK"
        if str(text).startswith("Error generating response:"):
            ok = False
        return {
            "provider": payload.provider,
            "model": payload.model,
            "ok": ok,
            "response": str(text)[:240],
        }
    except Exception as exc:
        return {
            "provider": payload.provider,
            "model": payload.model,
            "ok": False,
            "response": f"{type(exc).__name__}: {exc}"[:240],
        }


def _template_response(name: str) -> HTMLResponse:
    template = Path(__file__).resolve().parents[1] / "templates" / name
    return HTMLResponse(template.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def index():
    return _template_response("probe_landing.html")


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return _template_response("probe_demo.html")


@app.get("/decide", response_class=HTMLResponse)
def decision_lab():
    return _template_response("decision_lab.html")


@app.get("/workbench", response_class=HTMLResponse)
def workbench():
    return _template_response("agent_workbench.html")


@app.get("/research", response_class=HTMLResponse)
def research():
    return _template_response("playable_probe.html")
