from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from causalrag import __version__
from causalrag.agent import RuntimeCapabilities
from causalrag.probe import ProbeRunConfig, available_probe_config, run_probe_episode


app = FastAPI(
    title="CausalRAG Playable Probe",
    description="Interactive research probe for causal-runtime and proposer-model ablations.",
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


@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__, "surface": "playable_probe"}


@app.get("/api/config")
def probe_config():
    return available_probe_config()


@app.post("/api/run")
def run_probe(payload: ProbeRunRequest):
    try:
        config = ProbeRunConfig(
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
        return run_probe_episode(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index():
    template = Path(__file__).resolve().parents[1] / "templates" / "playable_probe.html"
    return HTMLResponse(template.read_text(encoding="utf-8"))
