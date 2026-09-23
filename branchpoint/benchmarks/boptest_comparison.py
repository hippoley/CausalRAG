from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from branchpoint.environments import BOPTESTClient

from .boptest_protocol import (
    BOPTESTControlContext,
    BOPTESTEpisodeResult,
    BOPTESTScenarioManifest,
    Controller,
    run_boptest_episode,
)


COMPARISON_PROTOCOL_VERSION = "branchpoint.boptest-comparison.v1"


class BOPTESTComparisonError(RuntimeError):
    """The paired external-environment comparison is not valid."""


@dataclass(frozen=True)
class BOPTESTControllerSpec:
    controller_id: str
    controller: Controller
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        controller_id = str(self.controller_id).strip()
        if not controller_id:
            raise BOPTESTComparisonError("controller_id must be non-empty")
        object.__setattr__(self, "controller_id", controller_id)
        if not callable(self.controller):
            raise BOPTESTComparisonError("controller must be callable")
        try:
            json.dumps(
                dict(self.metadata),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                default=None,
            )
        except (TypeError, ValueError) as exc:
            raise BOPTESTComparisonError(
                "controller metadata must be JSON-serializable"
            ) from exc


@dataclass(frozen=True)
class BOPTESTPairedEpisode:
    manifest: BOPTESTScenarioManifest
    episodes: Mapping[str, BOPTESTEpisodeResult]
    reference_controller_id: str
    controller_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def kpi_deltas(self) -> Dict[str, Dict[str, float]]:
        reference = self.episodes[self.reference_controller_id].kpis
        deltas: Dict[str, Dict[str, float]] = {}
        for controller_id, episode in self.episodes.items():
            if controller_id == self.reference_controller_id:
                continue
            common = sorted(set(reference) & set(episode.kpis))
            deltas[controller_id] = {
                name: float(episode.kpis[name]) - float(reference[name])
                for name in common
            }
        return deltas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.manifest_hash,
            "reference_controller_id": self.reference_controller_id,
            "controller_metadata": {key: dict(value) for key, value in self.controller_metadata.items()},
            "episodes": {key: value.to_dict() for key, value in self.episodes.items()},
            "kpi_delta_from_reference": self.kpi_deltas(),
        }


@dataclass(frozen=True)
class BOPTESTComparisonReport:
    reference_controller_id: str
    paired_episodes: Tuple[BOPTESTPairedEpisode, ...]
    protocol_version: str = COMPARISON_PROTOCOL_VERSION

    def aggregate_paired_deltas(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        collected: Dict[str, Dict[str, list[float]]] = {}
        for pair in self.paired_episodes:
            for controller_id, deltas in pair.kpi_deltas().items():
                controller_rows = collected.setdefault(controller_id, {})
                for kpi, value in deltas.items():
                    controller_rows.setdefault(kpi, []).append(float(value))

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for controller_id, kpis in collected.items():
            summary[controller_id] = {}
            for kpi, values in kpis.items():
                summary[controller_id][kpi] = {
                    "count": len(values),
                    "mean_delta": float(statistics.fmean(values)),
                    "min_delta": float(min(values)),
                    "max_delta": float(max(values)),
                }
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "reference_controller_id": self.reference_controller_id,
            "paired_episodes": [pair.to_dict() for pair in self.paired_episodes],
            "aggregate_paired_kpi_deltas": self.aggregate_paired_deltas(),
        }


@dataclass(frozen=True)
class BOPTESTBootstrapInterval:
    controller_id: str
    kpi: str
    count: int
    mean_delta: float
    confidence: float
    lower: Optional[float]
    upper: Optional[float]
    resamples: int
    bootstrap_seed: int
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "kpi": self.kpi,
            "count": self.count,
            "mean_delta": self.mean_delta,
            "confidence": self.confidence,
            "lower": self.lower,
            "upper": self.upper,
            "resamples": self.resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "status": self.status,
        }


@dataclass(frozen=True)
class BOPTESTBootstrapReport:
    reference_controller_id: str
    manifest_hashes: Tuple[str, ...]
    confidence: float
    resamples: int
    bootstrap_seed: int
    min_pairs: int
    intervals: Mapping[str, Mapping[str, BOPTESTBootstrapInterval]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "paired_percentile_bootstrap",
            "reference_controller_id": self.reference_controller_id,
            "manifest_hashes": list(self.manifest_hashes),
            "confidence": self.confidence,
            "resamples": self.resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "min_pairs": self.min_pairs,
            "intervals": {
                controller_id: {kpi: interval.to_dict() for kpi, interval in kpis.items()}
                for controller_id, kpis in self.intervals.items()
            },
        }


def expand_seeded_manifests(
    base: BOPTESTScenarioManifest,
    seeds: Sequence[int],
) -> Tuple[BOPTESTScenarioManifest, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise BOPTESTComparisonError("at least one seed is required")
    if len(set(normalized)) != len(normalized):
        raise BOPTESTComparisonError("seed list must be unique")
    return tuple(replace(base, seed=seed) for seed in normalized)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BOPTESTComparisonError("cannot compute percentile of empty values")
    position = (len(ordered) - 1) * float(q)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def _series_seed(master_seed: int, controller_id: str, kpi: str) -> int:
    material = f"{int(master_seed)}:{controller_id}:{kpi}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def bootstrap_paired_kpi_intervals(
    report: BOPTESTComparisonReport,
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
    min_pairs: int = 2,
) -> BOPTESTBootstrapReport:
    """Estimate uncertainty over manifest-level paired KPI deltas."""

    confidence = float(confidence)
    resamples = int(resamples)
    min_pairs = int(min_pairs)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if min_pairs < 2:
        raise ValueError("min_pairs must be at least 2")

    collected: Dict[str, Dict[str, list[float]]] = {}
    for pair in report.paired_episodes:
        for controller_id, deltas in pair.kpi_deltas().items():
            bucket = collected.setdefault(controller_id, {})
            for kpi, value in deltas.items():
                bucket.setdefault(kpi, []).append(float(value))

    alpha = (1.0 - confidence) / 2.0
    intervals: Dict[str, Dict[str, BOPTESTBootstrapInterval]] = {}
    for controller_id in sorted(collected):
        intervals[controller_id] = {}
        for kpi in sorted(collected[controller_id]):
            values = collected[controller_id][kpi]
            mean_delta = float(statistics.fmean(values))
            series_seed = _series_seed(seed, controller_id, kpi)
            if len(values) < min_pairs:
                interval = BOPTESTBootstrapInterval(
                    controller_id=controller_id,
                    kpi=kpi,
                    count=len(values),
                    mean_delta=mean_delta,
                    confidence=confidence,
                    lower=None,
                    upper=None,
                    resamples=resamples,
                    bootstrap_seed=series_seed,
                    status="insufficient_pairs",
                )
            else:
                rng = random.Random(series_seed)
                sample_means = []
                count = len(values)
                for _ in range(resamples):
                    sample = [values[rng.randrange(count)] for _ in range(count)]
                    sample_means.append(float(statistics.fmean(sample)))
                interval = BOPTESTBootstrapInterval(
                    controller_id=controller_id,
                    kpi=kpi,
                    count=count,
                    mean_delta=mean_delta,
                    confidence=confidence,
                    lower=float(_percentile(sample_means, alpha)),
                    upper=float(_percentile(sample_means, 1.0 - alpha)),
                    resamples=resamples,
                    bootstrap_seed=series_seed,
                    status="ok",
                )
            intervals[controller_id][kpi] = interval

    return BOPTESTBootstrapReport(
        reference_controller_id=report.reference_controller_id,
        manifest_hashes=tuple(
            pair.manifest.manifest_hash for pair in report.paired_episodes
        ),
        confidence=confidence,
        resamples=resamples,
        bootstrap_seed=int(seed),
        min_pairs=min_pairs,
        intervals=intervals,
    )

def constant_controller(controls: Mapping[str, Any]) -> Controller:
    """Return a controller that emits the same explicit controls every step."""
    frozen = dict(controls)

    def controller(_context: BOPTESTControlContext) -> Mapping[str, Any]:
        return dict(frozen)

    return controller


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def run_boptest_comparison(
    manifests: Sequence[BOPTESTScenarioManifest],
    controllers: Sequence[BOPTESTControllerSpec],
    *,
    reference_controller_id: str,
    client_factory: Callable[[], BOPTESTClient] = BOPTESTClient,
    require_same_service_version: bool = True,
    queue_timeout_seconds: float = 120.0,
    poll_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> BOPTESTComparisonReport:
    """Run every controller against each frozen manifest as paired episodes."""

    manifests = tuple(manifests)
    controllers = tuple(controllers)
    if not manifests:
        raise BOPTESTComparisonError("at least one manifest is required")
    if not controllers:
        raise BOPTESTComparisonError("at least one controller is required")

    hashes = [manifest.manifest_hash for manifest in manifests]
    if len(set(hashes)) != len(hashes):
        raise BOPTESTComparisonError("comparison manifests must have unique manifest hashes")

    controller_ids = [spec.controller_id for spec in controllers]
    if len(set(controller_ids)) != len(controller_ids):
        raise BOPTESTComparisonError("controller ids must be unique")
    reference_controller_id = str(reference_controller_id).strip()
    if reference_controller_id not in set(controller_ids):
        raise BOPTESTComparisonError("reference_controller_id must name one controller")

    paired = []
    controller_metadata = {spec.controller_id: dict(spec.metadata) for spec in controllers}

    version_fingerprint: Optional[str] = None
    for manifest in manifests:
        episodes: Dict[str, BOPTESTEpisodeResult] = {}
        for spec in controllers:
            client = client_factory()
            result = run_boptest_episode(
                manifest,
                spec.controller,
                controller_id=spec.controller_id,
                client=client,
                queue_timeout_seconds=queue_timeout_seconds,
                poll_seconds=poll_seconds,
                sleep=sleep,
            )
            if require_same_service_version:
                current = _canonical_json(result.service_version)
                if version_fingerprint is None:
                    version_fingerprint = current
                elif current != version_fingerprint:
                    raise BOPTESTComparisonError(
                        "BOPTEST service version changed between comparison episodes"
                    )
            episodes[spec.controller_id] = result

        paired.append(
            BOPTESTPairedEpisode(
                manifest=manifest,
                episodes=episodes,
                reference_controller_id=reference_controller_id,
                controller_metadata=controller_metadata,
            )
        )

    return BOPTESTComparisonReport(
        reference_controller_id=reference_controller_id,
        paired_episodes=tuple(paired),
    )
