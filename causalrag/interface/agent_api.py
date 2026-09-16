"""FastAPI surface for the v0.2 causal-agent runtime.

This module reuses the legacy API app and adds the new goal-directed endpoint.
Run with: uvicorn causalrag.interface.agent_api:app --reload
"""

from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from causalrag import create_agent

from .api import app


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Goal for the causal agent")
    max_steps: int = Field(8, ge=1, le=50)
    model: str = Field("gpt-4o-mini")
    provider: str = Field("openai")
    documents: Optional[List[str]] = Field(
        None,
        description="Optional documents to index for this run. Prefer a persistent index for production.",
    )


@app.post("/agent/run")
def run_agent(payload: AgentRunRequest):
    """Run the causal learning/action loop and return its full decision trace."""
    try:
        agent = create_agent(
            model_name=payload.model,
            provider=payload.provider,
            documents=payload.documents,
        )
        return agent.run(payload.goal, max_steps=payload.max_steps).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
