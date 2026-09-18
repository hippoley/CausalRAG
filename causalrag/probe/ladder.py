from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from causalrag.agent import RuntimeCapabilities
from causalrag.generator.llm_interface import LLMInterface

from .compare import MemoizedLLM, SharedPromptMemo
from .runtime import ProbeRunConfig, run_probe_episode


def capability_ladder_profiles() -> List[Dict[str, Any]]:
    """Cumulative execution profiles used by the Playable Probe ladder."""

    return [
        {
            "id": "vanilla",
            "label": "Vanilla tool loop",
            "description": "No causal selection, posterior learning, EIG/EVSI, temporal attribution, or open-world handling.",
            "capabilities": RuntimeCapabilities.vanilla_tool_loop(),
        },
        {
            "id": "runtime_eig",
            "label": "Runtime EIG",
            "description": "Runtime chooses information-seeking actions using experiment information gain, but observations do not update posterior state.",
            "capabilities": RuntimeCapabilities(
                causal_selection=True,
                causal_updates=False,
                bayesian_updates=False,
                eig=True,
                evsi=False,
                temporal_attribution=False,
                open_world=False,
                retrieval=False,
            ),
        },
        {
            "id": "bayesian_learning",
            "label": "Bayesian learning",
            "description": "Adds runtime-owned Bayesian posterior updates after experiment outcomes.",
            "capabilities": RuntimeCapabilities(
                causal_selection=True,
                causal_updates=False,
                bayesian_updates=True,
                eig=True,
                evsi=False,
                temporal_attribution=False,
                open_world=False,
                retrieval=False,
            ),
        },
        {
            "id": "decision_value",
            "label": "Decision value / EVSI",
            "description": "Adds expected downstream decision value and value of sample information.",
            "capabilities": RuntimeCapabilities(
                causal_selection=True,
                causal_updates=False,
                bayesian_updates=True,
                eig=True,
                evsi=True,
                temporal_attribution=False,
                open_world=False,
                retrieval=False,
            ),
        },
        {
            "id": "temporal_open_world",
            "label": "Temporal + open world",
            "description": "Adds temporal attribution, mismatch detection, provisional discovery, and generic causal updates.",
            "capabilities": RuntimeCapabilities(
                causal_selection=True,
                causal_updates=True,
                bayesian_updates=True,
                eig=True,
                evsi=True,
                temporal_attribution=True,
                open_world=True,
                retrieval=False,
            ),
        },
        {
            "id": "full",
            "label": "Full runtime",
            "description": "Full causal runtime including retrieval capability.",
            "capabilities": RuntimeCapabilities.full(),
        },
    ]


def _external_model(config: ProbeRunConfig) -> tuple[str, str]:
    if config.proposer_family == "small":
        return config.provider or "local", config.model or "local-model"
    return config.provider or "openai", config.model or "gpt-5.6-terra"


def _metric_delta(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, float]:
    names = (
        "causal_regret",
        "total_cost",
        "probes",
        "interventions",
        "decision_rounds",
        "true_hypothesis_posterior",
        "brier_score",
    )
    before = previous.get("metrics") or {}
    after = current.get("metrics") or {}
    result: Dict[str, float] = {}
    for name in names:
        if name not in before or name not in after:
            continue
        try:
            result[name] = float(after[name]) - float(before[name])
        except (TypeError, ValueError):
            continue
    result["success"] = float(bool(after.get("success"))) - float(bool(before.get("success")))
    return result


def _first_action_difference(left: Dict[str, Any], right: Dict[str, Any]):
    left_rows = left.get("decisions") or []
    right_rows = right.get("decisions") or []
    width = max(len(left_rows), len(right_rows))
    for step in range(width):
        a = (left_rows[step].get("selected") or {}) if step < len(left_rows) else None
        b = (right_rows[step].get("selected") or {}) if step < len(right_rows) else None
        if a is None or b is None:
            return {"step": step, "previous": a, "current": b}
        if (a.get("kind"), a.get("name")) != (b.get("kind"), b.get("name")):
            return {
                "step": step,
                "previous": {"kind": a.get("kind"), "name": a.get("name")},
                "current": {"kind": b.get("kind"), "name": b.get("name")},
            }
    return None


def run_probe_ladder(config: ProbeRunConfig) -> Dict[str, Any]:
    """Run cumulative causal-runtime capabilities on one paired task family."""

    paired = replace(
        config,
        stochastic_coupling=(
            "action_indexed"
            if config.outcome_mode == "stochastic"
            else config.stochastic_coupling
        ),
    )
    memo = SharedPromptMemo() if paired.proposer_family != "deterministic" else None
    provider = model = None
    if memo is not None:
        provider, model = _external_model(paired)

    arms: List[Dict[str, Any]] = []
    previous_episode = None
    for profile in capability_ladder_profiles():
        arm_config = replace(
            paired,
            capabilities=profile["capabilities"].to_dict(),
        )
        llm = None
        if memo is not None:
            llm = MemoizedLLM(
                LLMInterface(model=model, provider=provider),
                memo,
                str(profile["id"]),
            )
        episode = run_probe_episode(arm_config, llm=llm)
        row = {
            "id": profile["id"],
            "label": profile["label"],
            "description": profile["description"],
            "capabilities": profile["capabilities"].to_dict(),
            "episode": episode,
            "marginal_delta_from_previous": (
                None if previous_episode is None else _metric_delta(previous_episode, episode)
            ),
            "first_action_difference_from_previous": (
                None
                if previous_episode is None
                else _first_action_difference(previous_episode, episode)
            ),
        }
        arms.append(row)
        previous_episode = episode

    return {
        "comparison": {
            "scenario": paired.scenario,
            "hidden_hypothesis": paired.hidden_hypothesis,
            "outcome_mode": paired.outcome_mode,
            "seed": int(paired.seed),
            "proposer_family": paired.proposer_family,
            "provider": paired.provider,
            "model": paired.model,
            "paired_randomness": (
                "identical_deterministic_outcomes"
                if paired.outcome_mode == "deterministic"
                else "action_indexed_common_random_numbers"
            ),
            "shared_identical_prompt_outputs": memo is not None,
            "shared_prompt_provider_calls": None if memo is None else memo.provider_calls,
            "shared_prompt_replays": None if memo is None else memo.replays,
        },
        "arms": arms,
    }
