from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from causalrag.agent import RuntimeCapabilities, create_ablation_agent
from causalrag.benchmarks.hidden_world import (
    HiddenWorldEnvironment,
    HiddenWorldReasoner,
    build_hvac_hidden_world,
)
from causalrag.observability import CausalTelemetry


_CAPABILITY_NAMES = tuple(RuntimeCapabilities.full().to_dict())


@dataclass(frozen=True)
class ProbeRunConfig:
    """Frozen inputs for one Playable Probe episode.

    The same object can be serialized into a paper artifact. HiddenWorld is the
    first executable scenario; the interface is deliberately environment-agnostic
    so BOPTEST can be added without changing the front-end contract.
    """

    scenario: str = "hvac_hidden_world"
    hidden_hypothesis: str = "H2"
    outcome_mode: str = "stochastic"
    seed: int = 0
    max_steps: int = 6
    max_probes: int = 3
    confidence_threshold: float = 0.8
    proposer_family: str = "deterministic"
    provider: Optional[str] = None
    model: Optional[str] = None
    user_context: str = ""
    capabilities: Mapping[str, bool] = field(
        default_factory=lambda: RuntimeCapabilities.full().to_dict()
    )

    def __post_init__(self) -> None:
        if self.scenario != "hvac_hidden_world":
            raise ValueError("only hvac_hidden_world is executable in the first probe slice")
        if self.hidden_hypothesis not in {"H1", "H2", "H3"}:
            raise ValueError("hidden_hypothesis must be H1, H2, or H3")
        if self.outcome_mode not in {"deterministic", "stochastic"}:
            raise ValueError("outcome_mode must be deterministic or stochastic")
        if self.proposer_family not in {"deterministic", "small", "frontier"}:
            raise ValueError("proposer_family must be deterministic, small, or frontier")
        if int(self.max_steps) <= 0 or int(self.max_probes) < 0:
            raise ValueError("max_steps must be positive and max_probes non-negative")
        unknown = set(self.capabilities) - set(_CAPABILITY_NAMES)
        if unknown:
            raise ValueError(f"unknown runtime capabilities: {sorted(unknown)}")

    def resolved_capabilities(self) -> RuntimeCapabilities:
        values = RuntimeCapabilities.full().to_dict()
        values.update({str(key): bool(value) for key, value in self.capabilities.items()})
        return RuntimeCapabilities(**values)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = self.resolved_capabilities().to_dict()
        return payload


def available_probe_config() -> Dict[str, Any]:
    """Describe the probe control surface without leaking credentials."""

    return {
        "scenarios": [
            {
                "id": "hvac_hidden_world",
                "label": "HVAC hidden mechanism",
                "hidden_hypotheses": ["H1", "H2", "H3"],
                "description": "Diagnose filter, fan, or duct faults under noisy observations.",
            }
        ],
        "outcome_modes": ["deterministic", "stochastic"],
        "proposer_families": {
            "deterministic": {"requires_model": False},
            "small": {
                "requires_model": True,
                "default_provider": "local",
                "default_model": os.getenv("CAUSALRAG_SMALL_MODEL", "local-model"),
            },
            "frontier": {
                "requires_model": True,
                "default_provider": "openai",
                "default_model": os.getenv("CAUSALRAG_FRONTIER_MODEL", "gpt-5.6-terra"),
            },
        },
        "capabilities": list(_CAPABILITY_NAMES),
        "backend_status": {
            "openai": {
                "configured": bool(os.getenv("OPENAI_API_KEY")),
                "default_model": os.getenv("CAUSALRAG_FRONTIER_MODEL", "gpt-5.6-terra"),
            },
            "anthropic": {
                "configured": bool(os.getenv("ANTHROPIC_API_KEY")),
            },
            "local": {
                "configured": True,
                "endpoint": os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
                "default_model": os.getenv("CAUSALRAG_SMALL_MODEL", "local-model"),
            },
        },
    }


def _resolve_model(config: ProbeRunConfig) -> tuple[Optional[str], Optional[str]]:
    if config.proposer_family == "deterministic":
        return None, None
    if config.proposer_family == "small":
        return (
            config.provider or "local",
            config.model or os.getenv("CAUSALRAG_SMALL_MODEL", "local-model"),
        )
    return (
        config.provider or "openai",
        config.model or os.getenv("CAUSALRAG_FRONTIER_MODEL", "gpt-5.6-terra"),
    )


def _build_probe_agent(config: ProbeRunConfig, decision_hook=None) -> Dict[str, Any]:
    """Build the canonical probe runtime without executing an episode."""

    scenario = build_hvac_hidden_world(config.hidden_hypothesis)
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
    )
    world = environment.world_model()
    telemetry = CausalTelemetry(capture_content=False)
    capabilities = config.resolved_capabilities()

    provider, model = _resolve_model(config)
    kwargs: Dict[str, Any] = {
        "capabilities": capabilities,
        "world_model": world,
        "tools": environment.tools(),
        "telemetry": telemetry,
        "decision_hook": decision_hook,
    }
    if config.proposer_family == "deterministic":
        kwargs["reasoner"] = HiddenWorldReasoner(
            scenario,
            confidence_threshold=float(config.confidence_threshold),
            max_probes=int(config.max_probes),
        )
    else:
        kwargs["provider"] = provider
        kwargs["model_name"] = model

    agent = create_ablation_agent(**kwargs)
    goal = (
        "Identify the hidden HVAC causal mechanism using the available diagnostic "
        "experiments, then apply the intervention most likely to fix it. Minimize "
        "unnecessary probes, cost, and incorrect interventions."
    )
    if str(config.user_context).strip():
        goal += " Operator context: " + str(config.user_context).strip()
    return {
        "scenario": scenario,
        "environment": environment,
        "world": world,
        "telemetry": telemetry,
        "capabilities": capabilities,
        "agent": agent,
        "goal": goal,
    }


def run_probe_episode(config: ProbeRunConfig) -> Dict[str, Any]:
    """Run one real episode through the canonical agent + telemetry stack."""

    built = _build_probe_agent(config)
    environment = built["environment"]
    telemetry = built["telemetry"]
    capabilities = built["capabilities"]
    agent = built["agent"]
    goal = built["goal"]
    result = agent.run(goal, max_steps=int(config.max_steps))
    metrics = environment.metrics(result)
    payload = result.to_dict()
    return {
        "config": config.to_dict(),
        "metrics": metrics.to_dict(),
        "answer": payload["answer"],
        "trace_id": payload["trace_id"],
        "hypotheses": payload["hypotheses"],
        "open_world": payload["open_world"],
        "decisions": payload["decisions"],
        "observations": payload["observations"],
        "transitions": payload["transitions"],
        "causal_trace": payload["causal_trace"],
        "runtime_capabilities": capabilities.to_dict(),
    }
