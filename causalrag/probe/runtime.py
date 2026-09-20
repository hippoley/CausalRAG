from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from causalrag.agent import RuntimeCapabilities, create_ablation_agent
from causalrag.observability import CausalTelemetry

from .scenarios import (
    build_scenario_runtime,
    scenario_metrics,
    scenario_summaries,
    validate_scenario_config,
)


_CAPABILITY_NAMES = tuple(RuntimeCapabilities.full().to_dict())


@dataclass(frozen=True)
class ProbeRunConfig:
    """Frozen inputs for one Playable Probe episode.

    The same object can be serialized into a paper artifact. Scenario-specific
    validation and construction are delegated to the probe scenario adapters.
    """

    scenario: str = "hvac_hidden_world"
    hidden_hypothesis: str = "H2"
    outcome_mode: str = "stochastic"
    stochastic_coupling: str = "sequence"
    seed: int = 0
    max_steps: int = 6
    max_probes: int = 3
    confidence_threshold: float = 0.8
    goal: Optional[str] = None
    proposer_family: str = "deterministic"
    provider: Optional[str] = None
    model: Optional[str] = None
    capabilities: Mapping[str, bool] = field(
        default_factory=lambda: RuntimeCapabilities.full().to_dict()
    )

    def __post_init__(self) -> None:
        validate_scenario_config(
            self.scenario,
            self.hidden_hypothesis,
            self.outcome_mode,
        )
        if self.stochastic_coupling not in {"sequence", "action_indexed"}:
            raise ValueError("stochastic_coupling must be sequence or action_indexed")
        if self.proposer_family not in {"deterministic", "small", "frontier"}:
            raise ValueError("proposer_family must be deterministic, small, or frontier")
        if int(self.max_steps) <= 0 or int(self.max_probes) < 0:
            raise ValueError("max_steps must be positive and max_probes non-negative")
        if self.goal is not None and not str(self.goal).strip():
            raise ValueError("goal must be non-empty when supplied")
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
        "scenarios": scenario_summaries(),
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
        "model_connections": {
            "openai": {
                "configured": bool(os.getenv("OPENAI_API_KEY")),
                "credential_source": "server_environment",
            },
            "anthropic": {
                "configured": bool(os.getenv("ANTHROPIC_API_KEY")),
                "credential_source": "server_environment",
            },
            "local": {
                "configured": bool(os.getenv("LOCAL_LLM_URL")),
                "endpoint": os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
                "credential_source": "local_openai_compatible",
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


DEFAULT_PROBE_GOAL = (
    "Identify the hidden HVAC causal mechanism using the available diagnostic "
    "experiments, then apply the intervention most likely to fix it. Minimize "
    "unnecessary probes, cost, and incorrect interventions."
)


def build_probe_agent(
    config: ProbeRunConfig,
    *,
    telemetry: Optional[CausalTelemetry] = None,
    decision_gate: Optional[Any] = None,
    llm: Optional[Any] = None,
):
    """Build one probe episode without running it.

    This is the shared construction path for one-shot benchmark runs and the
    human-in-the-loop Playable Probe session. The world, tools, model tier, and
    runtime capabilities therefore stay identical across both surfaces.
    """

    scenario_runtime = build_scenario_runtime(config)
    environment = scenario_runtime.environment
    telemetry = telemetry or CausalTelemetry(capture_content=False)
    capabilities = config.resolved_capabilities()

    provider, model = _resolve_model(config)
    kwargs: Dict[str, Any] = {
        "capabilities": capabilities,
        "world_model": scenario_runtime.world_model,
        "tools": scenario_runtime.tools,
        "telemetry": telemetry,
        "decision_gate": decision_gate,
    }
    if scenario_runtime.time_driver is not None:
        kwargs["time_driver"] = scenario_runtime.time_driver
    if scenario_runtime.mismatch_policy is not None:
        kwargs["mismatch_policy"] = scenario_runtime.mismatch_policy

    if config.proposer_family == "deterministic":
        kwargs["reasoner"] = scenario_runtime.default_reasoner
    else:
        kwargs["provider"] = provider
        kwargs["model_name"] = model
        if llm is not None:
            kwargs["llm"] = llm

    agent = create_ablation_agent(**kwargs)
    goal = str(config.goal or scenario_runtime.goal)
    return environment, agent, goal, capabilities


def run_probe_episode(
    config: ProbeRunConfig,
    *,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one real episode through the canonical agent + telemetry stack."""

    environment, agent, goal, capabilities = build_probe_agent(config, llm=llm)
    result = agent.run(goal, max_steps=int(config.max_steps))
    metrics = scenario_metrics(environment, result)
    payload = result.to_dict()
    # Keep one-shot Examples and interactive sessions on the same canonical
    # ledger shape. Import lazily to avoid a module-import cycle: session.py
    # depends on ProbeRunConfig/build_probe_agent from this module.
    from .session import _episode_ledger

    episode_ledger = _episode_ledger(result.state, result.world_model, agent.loop.tools)
    return {
        "config": config.to_dict(),
        "metrics": metrics,
        "answer": payload["answer"],
        "trace_id": payload["trace_id"],
        "hypotheses": payload["hypotheses"],
        "open_world": payload["open_world"],
        "decisions": payload["decisions"],
        "observations": payload["observations"],
        "transitions": payload["transitions"],
        "causal_trace": payload["causal_trace"],
        "runtime_capabilities": capabilities.to_dict(),
        "episode_ledger": episode_ledger,
    }
