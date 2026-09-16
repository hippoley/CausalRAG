"""Standalone FastAPI surface for the v0.2 causal-agent runtime.

Run with:
    uvicorn causalrag.interface.agent_api:app --reload
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from causalrag import __version__, create_agent


app = FastAPI(
    title="CausalRAG Agent API",
    description="Goal-directed causal agent runtime with explicit beliefs and decision traces.",
    version=__version__,
)


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Goal for the causal agent")
    max_steps: int = Field(8, ge=1, le=50)
    model: str = Field("gpt-5.6-terra")
    provider: str = Field("openai")
    documents: Optional[List[str]] = Field(
        None,
        description="Optional documents to index for this run. Prefer a persistent index for production.",
    )
    embedding_provider: str = Field(
        "openai",
        description="Embedding provider: 'openai' or opt-in 'local'.",
    )
    embedding_model: str = Field("text-embedding-3-small")
    vector_backend: str = Field(
        "memory",
        description="Vector backend: 'memory' or opt-in 'faiss'.",
    )


@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__, "runtime": "causal-agent"}


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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
