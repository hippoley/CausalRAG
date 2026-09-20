from __future__ import annotations

from typing import Any, Optional

from .capabilities import RuntimeCapabilities
from .runtime import CausalAgent, create_agent


def apply_runtime_capabilities(
    agent: CausalAgent,
    capabilities: Optional[RuntimeCapabilities] = None,
) -> CausalAgent:
    """Attach execution-level ablation switches to an existing canonical agent.

    The causal loop owns the behavior switches; observability, tool execution,
    world-model state and result tracing continue to use the normal CausalAgent
    implementation. This keeps ablation orthogonal to telemetry.
    """

    resolved = capabilities or RuntimeCapabilities.full()
    agent.loop.capabilities = resolved
    return agent


def create_ablation_agent(
    *,
    capabilities: Optional[RuntimeCapabilities] = None,
    **kwargs: Any,
) -> CausalAgent:
    """Create the normal observable agent, then apply runtime capabilities.

    Retrieval-off arms disable retrieval before construction so the ablation
    cannot accidentally initialize or call the retrieval stack it claims to
    remove.
    """

    resolved = capabilities or RuntimeCapabilities.full()
    if not resolved.retrieval:
        kwargs["enable_retrieval"] = False
        if kwargs.get("documents"):
            raise ValueError(
                "documents cannot be supplied when RuntimeCapabilities.retrieval is disabled"
            )
        if kwargs.get("graph_path") or kwargs.get("index_path"):
            raise ValueError(
                "graph_path/index_path cannot be supplied when RuntimeCapabilities.retrieval is disabled"
            )
    agent = create_agent(**kwargs)
    return apply_runtime_capabilities(agent, resolved)
