from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from causalrag import __version__
from causalrag.agent import RuntimeCapabilities
from causalrag.probe import (
    InteractiveProbeSession,
    ProbeRunConfig,
    available_probe_config,
    run_probe_episode,
)


app = FastAPI(
    title="CausalRAG Playable Probe",
    description="Interactive research probe for causal-runtime, proposer-model, and human-in-the-loop ablations.",
    version=__version__,
)


class ProbeRunRequest(BaseModel):
    hidden_hypothesis: Literal["H1", "H2", "H3"] = "H2"
    outcome_mode: Literal["deterministic", "stochastic"] = "stochastic"
    seed: int = 0
    max_steps: int = Field(6, ge=1, le=32)
    max_probes: int = Field(3, ge=0, le=16)
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    proposer_family: Literal["deterministic", "small", "frontier"] = "deterministic"
    provider: Optional[Literal["openai", "anthropic", "local"]] = None
    model: Optional[str] = None
    capabilities: Dict[str, bool] = Field(
        default_factory=lambda: RuntimeCapabilities.full().to_dict()
    )


class ProbeCommitRequest(BaseModel):
    selection: str = "runtime"
    human_note: str = ""


class ProbePreviewRequest(BaseModel):
    force_replan: bool = False


_SESSIONS: Dict[str, InteractiveProbeSession] = {}
_SESSIONS_LOCK = RLock()


def _to_config(payload: ProbeRunRequest) -> ProbeRunConfig:
    return ProbeRunConfig(
        hidden_hypothesis=payload.hidden_hypothesis,
        outcome_mode=payload.outcome_mode,
        seed=payload.seed,
        max_steps=payload.max_steps,
        max_probes=payload.max_probes,
        confidence_threshold=payload.confidence_threshold,
        proposer_family=payload.proposer_family,
        provider=payload.provider,
        model=payload.model,
        capabilities=payload.capabilities,
    )


def _get_session(session_id: str) -> InteractiveProbeSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="unknown probe session")
    return session


@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__, "surface": "playable_probe"}


@app.get("/api/config")
def probe_config():
    return available_probe_config()


@app.post("/api/run")
def run_probe(payload: ProbeRunRequest):
    try:
        return run_probe_episode(_to_config(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions")
def create_probe_session(payload: ProbeRunRequest):
    try:
        session = InteractiveProbeSession(_to_config(payload))
        preview = session.preview()
        with _SESSIONS_LOCK:
            _SESSIONS[session.session_id] = session
        return {
            "session_id": session.session_id,
            "preview": preview,
            "snapshot": session.snapshot(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def get_probe_session(session_id: str):
    return _get_session(session_id).snapshot()


@app.post("/api/sessions/{session_id}/preview")
def preview_probe_session(session_id: str, payload: ProbePreviewRequest):
    try:
        session = _get_session(session_id)
        return {
            "session_id": session.session_id,
            "preview": session.preview(force_replan=payload.force_replan),
            "snapshot": session.snapshot(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/commit")
def commit_probe_session(session_id: str, payload: ProbeCommitRequest):
    try:
        session = _get_session(session_id)
        snapshot = session.commit(payload.selection, human_note=payload.human_note)
        next_preview = None if snapshot["done"] else session.preview()
        return {
            "session_id": session.session_id,
            "snapshot": session.snapshot(),
            "preview": next_preview,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}")
def delete_probe_session(session_id: str):
    with _SESSIONS_LOCK:
        existed = _SESSIONS.pop(str(session_id), None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail="unknown probe session")
    return {"deleted": True, "session_id": session_id}


@app.get("/", response_class=HTMLResponse)
def index():
    template = Path(__file__).resolve().parents[1] / "templates" / "playable_probe.html"
    return HTMLResponse(template.read_text(encoding="utf-8"))
