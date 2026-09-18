from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional, Tuple

from causalrag.agent import RuntimeCapabilities
from causalrag.generator.llm_interface import LLMInterface

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


class SharedPromptMemo:
    """Share exact proposer outputs across A/B arms while prompts remain identical.

    Each arm gets its own adapter/telemetry surface but both adapters reference
    the same cache. An identical prompt therefore produces one provider call
    and one replay; after runtime state diverges, prompt keys diverge and each
    arm may call the model independently.
    """

    def __init__(self) -> None:
        self.values: Dict[Tuple[str, float, int, bool, bool], Any] = {}
        self.provider_calls = 0
        self.replays = 0


class MemoizedLLM:
    def __init__(self, inner: LLMInterface, memo: SharedPromptMemo, arm: str) -> None:
        self.inner = inner
        self.memo = memo
        self.arm = arm
        self.model = inner.model
        self.provider = inner.provider
        self.telemetry = None
        self.last_usage: Dict[str, int] = {}

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
        stream: bool = False,
        json_mode: bool = False,
    ):
        key = (str(prompt), float(temperature), int(max_tokens), bool(stream), bool(json_mode))
        if key in self.memo.values:
            self.memo.replays += 1
            self.last_usage = {}
            if self.telemetry is not None:
                self.telemetry.event(
                    "causalrag.llm.memoized_replay",
                    {
                        "causalrag.ab.arm": self.arm,
                        "gen_ai.request.model": self.model,
                        "gen_ai.provider.name": self.provider,
                        "causalrag.llm.prompt_characters": len(prompt),
                    },
                )
            return self.memo.values[key]

        self.memo.provider_calls += 1
        self.inner.telemetry = self.telemetry
        result = self.inner.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            json_mode=json_mode,
        )
        self.last_usage = dict(getattr(self.inner, "last_usage", {}) or {})
        self.memo.values[key] = result
        return result


def _paired_llms(config: ProbeRunConfig):
    if config.proposer_family == "deterministic":
        return None, None, None

    if config.proposer_family == "small":
        provider = config.provider or "local"
        model = config.model or "local-model"
    else:
        provider = config.provider or "openai"
        model = config.model or "gpt-5.6-terra"

    memo = SharedPromptMemo()
    vanilla = MemoizedLLM(
        LLMInterface(model=model, provider=provider),
        memo,
        "vanilla",
    )
    causal = MemoizedLLM(
        LLMInterface(model=model, provider=provider),
        memo,
        "causal",
    )
    return vanilla, causal, memo


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
) -> Dict[str, float]:
    left = baseline.get("metrics") or {}
    right = treatment.get("metrics") or {}
    deltas: Dict[str, float] = {}
    for name in sorted(set(left).intersection(right)):
        a, b = left[name], right[name]
        if isinstance(a, bool) and isinstance(b, bool):
            deltas[name] = float(int(b) - int(a))
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas[name] = float(b) - float(a)
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

    vanilla_llm, causal_llm, memo = _paired_llms(paired_config)
    vanilla = run_probe_episode(vanilla_config, llm=vanilla_llm)
    causal = run_probe_episode(causal_config, llm=causal_llm)
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
            "shared_identical_prompt_outputs": bool(memo is not None),
            "shared_prompt_provider_calls": None if memo is None else int(memo.provider_calls),
            "shared_prompt_replays": None if memo is None else int(memo.replays),
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
