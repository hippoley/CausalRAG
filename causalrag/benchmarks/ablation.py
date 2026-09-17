from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from causalrag.agent.features import RuntimeFeatureFlags

# Backwards-friendly public benchmark name. The implementation lives in the
# runtime layer so an ablation switch can change behavior rather than merely
# annotate an experiment row.
RuntimeFeatures = RuntimeFeatureFlags


@dataclass(frozen=True)
class AblationArm:
    arm_id: str
    proposer_family: str
    provider: Optional[str] = None
    model: Optional[str] = None
    features: RuntimeFeatureFlags = field(default_factory=RuntimeFeatureFlags)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.arm_id).strip():
            raise ValueError("arm_id must be non-empty")
        if self.proposer_family not in {"frontier", "small", "deterministic"}:
            raise ValueError("proposer_family must be frontier, small, or deterministic")
        if self.proposer_family != "deterministic" and not str(self.model or "").strip():
            raise ValueError("model is required for frontier/small arms")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "proposer_family": self.proposer_family,
            "provider": self.provider,
            "model": self.model,
            "features": self.features.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class AblationEpisode:
    arm_id: str
    scenario_id: str
    seed: int
    metrics: Dict[str, float]
    trace_id: Optional[str] = None
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationReport:
    arms: List[AblationArm]
    episodes: List[AblationEpisode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": [arm.to_dict() for arm in self.arms],
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def by_arm(self) -> Dict[str, List[AblationEpisode]]:
        grouped: Dict[str, List[AblationEpisode]] = {arm.arm_id: [] for arm in self.arms}
        for episode in self.episodes:
            grouped.setdefault(episode.arm_id, []).append(episode)
        return grouped


def run_ablation_matrix(
    *,
    arms: Iterable[AblationArm],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    run_episode: Callable[[AblationArm, str, int], AblationEpisode],
) -> AblationReport:
    """Run a rectangular ablation matrix with identical scenarios and seeds."""

    arm_rows = list(arms)
    scenario_rows = [str(value) for value in scenarios]
    seed_rows = [int(value) for value in seeds]
    episodes: List[AblationEpisode] = []
    for arm in arm_rows:
        for scenario_id in scenario_rows:
            for seed in seed_rows:
                episode = run_episode(arm, scenario_id, seed)
                if episode.arm_id != arm.arm_id:
                    raise ValueError("run_episode returned an episode for a different arm")
                if episode.scenario_id != scenario_id or int(episode.seed) != seed:
                    raise ValueError("run_episode returned mismatched scenario/seed metadata")
                episodes.append(episode)
    return AblationReport(arms=arm_rows, episodes=episodes)
