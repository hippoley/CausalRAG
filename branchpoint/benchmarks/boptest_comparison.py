from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
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

    def aggregate_paired_deltas(self) -> Dict[str, Dict[str, Dict[str, float]]]:
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
                    "count": float(len(values)),
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

    for manifest in manifests:
        episodes: Dict[str, BOPTESTEpisodeResult] = {}
        version_fingerprint: Optional[str] = None
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
                        "BOPTEST service version changed between paired controller arms"
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
