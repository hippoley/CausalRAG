from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RuntimeFeatures:
    """The causal-runtime switches that may be ablated independently."""

    causal_runtime: bool = True
    eig: bool = True
    evsi: bool = True
    temporal_attribution: bool = True
    open_world: bool = True
    retrieval: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """A proposer model identity without embedding benchmark-specific policy."""

    tier: str
    provider: str
    model: str
    endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier not in {"deterministic", "local_small", "frontier"}:
            raise ValueError("tier must be deterministic, local_small, or frontier")
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model must be non-empty")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    model: ModelSpec
    features: RuntimeFeatures = field(default_factory=RuntimeFeatures)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variant name must be non-empty")


@dataclass(frozen=True)
class ExperimentProtocol:
    """Conditions that must remain fixed across compared variants."""

    benchmark: str
    scenario: str
    seeds: Sequence[int]
    max_steps: int
    observation_budget: Optional[int] = None
    intervention_budget: Optional[int] = None
    tool_surface_id: str = "default"
    observation_surface_id: str = "default"
    protocol_version: str = "0.1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.benchmark.strip() or not self.scenario.strip():
            raise ValueError("benchmark and scenario must be non-empty")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(int(seed) for seed in self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EpisodeResult:
    variant: str
    seed: int
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    stddev: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    model_tier: str
    model: str
    features: RuntimeFeatures
    metrics: Mapping[str, MetricSummary]


EpisodeRunner = Callable[[AblationVariant, ExperimentProtocol, int], Mapping[str, Any]]


def _summary(values: Sequence[float]) -> MetricSummary:
    count = len(values)
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if count > 1 else 0.0
    half_width = 1.96 * stddev / math.sqrt(count) if count > 1 else 0.0
    return MetricSummary(
        count=count,
        mean=mean,
        stddev=stddev,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
    )


def validate_variants(variants: Sequence[AblationVariant]) -> None:
    if len(variants) < 2:
        raise ValueError("an ablation requires at least two variants")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")


def run_ablation(
    variants: Sequence[AblationVariant],
    protocol: ExperimentProtocol,
    run_episode: EpisodeRunner,
) -> Dict[str, Any]:
    """Run every variant on exactly the same seeded protocol.

    ``run_episode`` is environment-specific and must return a ``metrics``
    mapping. The harness owns ordering, protocol identity and aggregation so a
    model/runtime comparison cannot silently use different seeds or budgets.
    """

    validate_variants(variants)
    episodes: list[EpisodeResult] = []
    for seed in protocol.seeds:
        for variant in variants:
            raw = dict(run_episode(variant, protocol, int(seed)))
            raw_metrics = raw.get("metrics")
            if not isinstance(raw_metrics, Mapping) or not raw_metrics:
                raise ValueError(f"episode {variant.name}/{seed} returned no metrics")
            metrics: Dict[str, float] = {}
            for key, value in raw_metrics.items():
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"metric {key!r} is not numeric: {value!r}") from exc
                if not math.isfinite(number):
                    raise ValueError(f"metric {key!r} must be finite")
                metrics[str(key)] = number
            episodes.append(
                EpisodeResult(
                    variant=variant.name,
                    seed=int(seed),
                    metrics=metrics,
                    metadata=dict(raw.get("metadata") or {}),
                )
            )

    summaries: list[VariantSummary] = []
    for variant in variants:
        own = [episode for episode in episodes if episode.variant == variant.name]
        metric_names = sorted({name for episode in own for name in episode.metrics})
        metric_summaries = {
            name: _summary([episode.metrics[name] for episode in own if name in episode.metrics])
            for name in metric_names
        }
        summaries.append(
            VariantSummary(
                variant=variant.name,
                model_tier=variant.model.tier,
                model=variant.model.model,
                features=variant.features,
                metrics=metric_summaries,
            )
        )

    return {
        "protocol": asdict(protocol),
        "protocol_fingerprint": protocol.fingerprint(),
        "variants": [asdict(variant) for variant in variants],
        "episodes": [asdict(episode) for episode in episodes],
        "summaries": [asdict(summary) for summary in summaries],
    }


def canonical_model_ablation(
    *,
    deterministic_model: str = "deterministic-policy",
    local_model: str = "local-small-model",
    frontier_model: str = "frontier-model",
) -> list[AblationVariant]:
    """Reference three-tier comparison with an identical causal runtime."""

    return [
        AblationVariant(
            name="deterministic+causal",
            model=ModelSpec("deterministic", "builtin", deterministic_model),
        ),
        AblationVariant(
            name="local-small+causal",
            model=ModelSpec("local_small", "local", local_model),
        ),
        AblationVariant(
            name="frontier+causal",
            model=ModelSpec("frontier", "openai", frontier_model),
        ),
    ]


def causal_runtime_ablation(model: ModelSpec) -> list[AblationVariant]:
    """Reference architecture ablation while holding the proposer fixed."""

    return [
        AblationVariant("model-only-loop", model, RuntimeFeatures(causal_runtime=False, eig=False, evsi=False, temporal_attribution=False, open_world=False, retrieval=False)),
        AblationVariant("causal-no-eig", model, RuntimeFeatures(eig=False)),
        AblationVariant("causal-no-evsi", model, RuntimeFeatures(evsi=False)),
        AblationVariant("causal-no-temporal", model, RuntimeFeatures(temporal_attribution=False)),
        AblationVariant("causal-no-open-world", model, RuntimeFeatures(open_world=False)),
        AblationVariant("causal-full", model, RuntimeFeatures()),
    ]
