from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from causalrag.agent import RuntimeCapabilities

from .runtime import ProbeRunConfig, run_probe_episode


_DEFAULT_METRICS = (
    "causal_regret",
    "total_cost",
    "probes",
    "interventions",
    "decision_rounds",
    "true_hypothesis_posterior",
    "brier_score",
)


def _selected_actions(episode: Dict[str, Any]) -> list[Dict[str, Any]]:
    actions: list[Dict[str, Any]] = []
    for index, decision in enumerate(episode.get("decisions") or []):
        selected = decision.get("selected") or {}
        actions.append(
            {
                "step": index,
                "name": selected.get("name"),
                "kind": selected.get("kind"),
                "rationale": selected.get("rationale") or decision.get("rationale") or "",
            }
        )
    return actions


def _first_divergence(
    baseline: Dict[str, Any],
    treatment: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    left = _selected_actions(baseline)
    right = _selected_actions(treatment)
    width = max(len(left), len(right))
    for step in range(width):
        a = left[step] if step < len(left) else None
        b = right[step] if step < len(right) else None
        if a is None or b is None:
            return {"step": step, "vanilla": a, "causal": b}
        if (a.get("name"), a.get("kind")) != (b.get("name"), b.get("kind")):
            return {"step": step, "vanilla": a, "causal": b}
    return None


def _metric_deltas(
    baseline: Dict[str, Any],
    treatment: Dict[str, Any],
    names: Iterable[str] = _DEFAULT_METRICS,
) -> Dict[str, float]:
    left = baseline.get("metrics") or {}
    right = treatment.get("metrics") or {}
    deltas: Dict[str, float] = {}
    for name in names:
        if name not in left or name not in right:
            continue
        try:
            deltas[name] = float(right[name]) - float(left[name])
        except (TypeError, ValueError):
            continue
    return deltas


def run_probe_comparison(config: ProbeRunConfig) -> Dict[str, Any]:
    """Run the same frozen task with and without the causal control plane.

    The proposer family/provider/model, hidden world, seed, goal, tools and
    budgets are copied unchanged. Only RuntimeCapabilities differ.

    For stochastic HiddenWorld the comparison upgrades both arms to
    action-indexed common random numbers. The random variate for an experiment
    is a deterministic function of seed + experiment_id + occurrence index, so
    the same experiment receives the same noise even when arm action order
    diverges.
    """

    paired_config = replace(
        config,
        stochastic_coupling=(
            "action_indexed"
            if config.outcome_mode == "stochastic"
            else config.stochastic_coupling
        ),
    )
    causal_config = paired_config
    vanilla_config = replace(
        paired_config,
        capabilities=RuntimeCapabilities.vanilla_tool_loop().to_dict(),
    )

    vanilla = run_probe_episode(vanilla_config)
    causal = run_probe_episode(causal_config)
    first_divergence = _first_divergence(vanilla, causal)

    return {
        "comparison": {
            "scenario": config.scenario,
            "hidden_hypothesis": config.hidden_hypothesis,
            "outcome_mode": config.outcome_mode,
            "seed": int(config.seed),
            "goal": config.goal,
            "proposer_family": config.proposer_family,
            "provider": config.provider,
            "model": config.model,
            "same_world_inputs": True,
            "same_proposer_configuration": True,
            "paired_randomness": (
                "identical_deterministic_outcomes"
                if config.outcome_mode == "deterministic"
                else "action_indexed_common_random_numbers"
            ),
        },
        "vanilla": vanilla,
        "causal": causal,
        "first_divergence": first_divergence,
        "metric_deltas_causal_minus_vanilla": _metric_deltas(vanilla, causal),
        "success_delta": int(bool(causal["metrics"]["success"]))
        - int(bool(vanilla["metrics"]["success"])),
    }
