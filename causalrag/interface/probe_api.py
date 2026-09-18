from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from causalrag import __version__
from causalrag.agent import RuntimeCapabilities
from causalrag.generator.llm_interface import LLMInterface
from causalrag.probe import (
    ProbeRunConfig,
    SESSION_MANAGER,
    available_probe_config,
    run_probe_episode,
    sse_stream,
)


app = FastAPI(
    title="CausalRAG Playable Probe",
    description="Human-in-the-loop research probe for causal-runtime and proposer-model ablations.",
    version=__version__,
)


class ProbeRunRequest(BaseModel):
    hidden_hypothesis: Literal["H1", "H2", "H3"] = "H2"
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


class DecisionRequest(BaseModel):
    action: Literal["approve", "choose", "replan"]
    candidate_index: Optional[int] = Field(default=None, ge=0)


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


def _config(payload: ProbeRunRequest) -> ProbeRunConfig:
    return ProbeRunConfig(
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
def probe_config():
    return available_probe_config()


@app.post("/api/run")
def run_probe(payload: ProbeRunRequest):
    """One-shot compatibility endpoint used by scripts and CI."""
    try:
        return run_probe_episode(_config(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions")
def create_session(payload: ProbeRunRequest):
    try:
        session = SESSION_MANAGER.create(_config(payload))
        return session.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    return _session(session_id).snapshot()


@app.get("/api/sessions/{session_id}/events")
def session_events(session_id: str):
    session = _session(session_id)
    return StreamingResponse(
        sse_stream(session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sessions/{session_id}/decision")
def resolve_session_decision(session_id: str, payload: DecisionRequest):
    session = _session(session_id)
    try:
        session.resolve_decision(payload.action, payload.candidate_index)
        return session.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/hypotheses")
def add_human_hypothesis(session_id: str, payload: HumanHypothesisRequest):
    session = _session(session_id)
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


@app.post("/api/model/test")
def test_model_connection(payload: ModelTestRequest):
    """User-triggered live model check using server-side credentials only."""
    llm = LLMInterface(model=payload.model, provider=payload.provider)
    text = llm.generate(
        "Reply with exactly CAUSALRAG_MODEL_OK and nothing else.",
        temperature=0.0,
        max_tokens=32,
    )
    ok = str(text).strip() == "CAUSALRAG_MODEL_OK"
    if str(text).startswith("Error generating response:"):
        ok = False
    return {
        "provider": payload.provider,
        "model": payload.model,
        "ok": ok,
        "response": str(text)[:240],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    template = Path(__file__).resolve().parents[1] / "templates" / "playable_probe.html"
    return HTMLResponse(template.read_text(encoding="utf-8"))
