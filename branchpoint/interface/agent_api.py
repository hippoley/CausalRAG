"""Standalone FastAPI surface for the v0.3 causal decision runtime.

Run with:
    uvicorn branchpoint.interface.agent_api:app --reload
"""

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from branchpoint import __version__, create_agent


app = FastAPI(
    title="Branchpoint Agent API",
    description="Goal-directed causal agent runtime with explicit beliefs and decision traces.",
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
