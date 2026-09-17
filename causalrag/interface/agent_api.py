"""FastAPI surface for the causal decision runtime and Playable Probe.

Run with:
    uvicorn causalrag.interface.agent_api:app --reload

Then open:
    http://127.0.0.1:8000/probe
"""

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from causalrag import __version__, create_agent
from causalrag.benchmarks.ablation import AblationArm, RuntimeFeatures
from causalrag.observability import InMemoryEventSink, instrument_agent

from .probe_ui import PLAYABLE_PROBE_HTML


app = FastAPI(
    title="CausalRAG Agent API",
    description="Goal-directed causal agent runtime with explicit beliefs, decision traces, and a Playable Probe.",
    version=__version__,
)


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    max_steps: int = Field(8, ge=1, le=64)
    model: str = "gpt-5.6-terra"
    provider: Literal["openai", "anthropic", "local"] = "openai"
    documents: Optional[List[str]] = None
    embedding_provider: Optional[Literal["openai", "local"]] = "openai"
    embedding_model: str = "text-embedding-3-small"
    vector_backend: Literal["memory", "faiss"] = "memory"


class ProbeFeatureRequest(BaseModel):
    causal_runtime: bool = True
    eig: bool = True
    evsi: bool = True
    temporal_attribution: bool = True
    open_world_discovery: bool = True
    retrieval: bool = False


class ProbeRunRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    max_steps: int = Field(8, ge=1, le=64)
    lane: Literal["frontier", "small"] = "frontier"
    model: str = "gpt-5.6-terra"
    provider: Literal["openai", "anthropic", "local"] = "openai"
    features: ProbeFeatureRequest = Field(default_factory=ProbeFeatureRequest)


@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__}


@app.post("/agent/run")
def run_agent(payload: AgentRunRequest):
    """Run the causal learning/action loop and return its full decision trace."""
    try:
        agent = create_agent(
            model_name=payload.model,
            provider=payload.provider,
            documents=payload.documents,
            embedding_provider_name=payload.embedding_provider,
            embedding_model=payload.embedding_model,
            vector_backend=payload.vector_backend,
        )
        return agent.run(payload.goal, max_steps=payload.max_steps).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/probe", response_class=HTMLResponse)
def playable_probe():
    return HTMLResponse(PLAYABLE_PROBE_HTML)


@app.post("/probe/run")
def run_playable_probe(payload: ProbeRunRequest):
    """Run one model arm and return both result and replayable probe events.

    Model lane selection is active now. Runtime feature switches are returned as
    an explicit ablation manifest; until each switch is runtime-enforced they
    are marked ``feature_switches_enforced=false`` to prevent false experiment
    claims in the UI or paper artifact.
    """
    try:
        features = RuntimeFeatures(**payload.features.model_dump())
        arm = AblationArm(
            arm_id=f"{payload.lane}:{payload.provider}:{payload.model}",
            proposer_family=payload.lane,
            provider=payload.provider,
            model=payload.model,
            features=features,
        )
        sink = InMemoryEventSink()
        agent = create_agent(
            model_name=payload.model,
            provider=payload.provider,
            enable_retrieval=bool(features.retrieval),
        )
        instrument_agent(agent, sink)
        run = agent.run(payload.goal, max_steps=payload.max_steps)
        result = run.to_dict()
        run_id = str(run.state.scratch.get("run_id") or "")
        return {
            "run_id": run_id,
            "arm": arm.to_dict(),
            "feature_switches_enforced": False,
            "feature_switches_note": (
                "Model lane is enforced. Mechanism flags are recorded now and "
                "become executable ablations when policy/runtime gates land."
            ),
            "events": sink.snapshot(run_id),
            "result": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
