from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from causalrag.agent.capabilities import RuntimeCapabilities


@dataclass(frozen=True)
class ScenarioManifest:
    """Frozen environment inputs shared by every arm in a paired experiment."""

    scenario_id: str
    environment: str
    seed: int
    horizon_steps: int
    observation_budget: Optional[int] = None
    intervention_budget: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.scenario_id).strip():
            raise ValueError("scenario_id must be non-empty")
        if not str(self.environment).strip():
            raise ValueError("environment must be non-empty")
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.observation_budget is not None and int(self.observation_budget) < 0:
            raise ValueError("observation_budget must be non-negative")
        if self.intervention_budget is not None and int(self.intervention_budget) < 0:
            raise ValueError("intervention_budget must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "environment": self.environment,
            "seed": int(self.seed),
            "horizon_steps": int(self.horizon_steps),
            "observation_budget": self.observation_budget,
            "intervention_budget": self.intervention_budget,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AblationArm:
    """One proposer-model/runtime configuration in an ablation matrix."""

    arm_id: str
    proposer_family: str
    provider: Optional[str] = None
    model: Optional[str] = None
    capabilities: RuntimeCapabilities = field(default_factory=RuntimeCapabilities.full)
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
            "capabilities": self.capabilities.to_dict(),
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
    failure: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationReport:
    arms: List[AblationArm]
    scenarios: List[ScenarioManifest]
    episodes: List[AblationEpisode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": [arm.to_dict() for arm in self.arms],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def by_arm(self) -> Dict[str, List[AblationEpisode]]:
        grouped: Dict[str, List[AblationEpisode]] = {arm.arm_id: [] for arm in self.arms}
        for episode in self.episodes:
            grouped.setdefault(episode.arm_id, []).append(episode)
        return grouped

    def paired_metric_deltas(self, baseline_arm: str, treatment_arm: str, metric: str) -> List[float]:
        """Return treatment-baseline deltas for scenarios present in both arms."""
        by_key: Dict[tuple[str, int, str], AblationEpisode] = {}
        for episode in self.episodes:
            by_key[(episode.scenario_id, int(episode.seed), episode.arm_id)] = episode
        deltas: List[float] = []
        for scenario in self.scenarios:
            baseline = by_key.get((scenario.scenario_id, int(scenario.seed), baseline_arm))
            treatment = by_key.get((scenario.scenario_id, int(scenario.seed), treatment_arm))
            if baseline is None or treatment is None:
                continue
            if metric not in baseline.metrics or metric not in treatment.metrics:
                continue
            deltas.append(float(treatment.metrics[metric]) - float(baseline.metrics[metric]))
        return deltas


def run_ablation_matrix(
    *,
    arms: Iterable[AblationArm],
    scenarios: Iterable[ScenarioManifest],
    run_episode: Callable[[AblationArm, ScenarioManifest], AblationEpisode],
) -> AblationReport:
    """Run every arm against exactly the same frozen scenario manifests."""

    arm_rows = list(arms)
    scenario_rows = list(scenarios)
    if len({arm.arm_id for arm in arm_rows}) != len(arm_rows):
        raise ValueError("arm_id values must be unique")
    if len({(s.scenario_id, int(s.seed)) for s in scenario_rows}) != len(scenario_rows):
        raise ValueError("scenario_id/seed pairs must be unique")

    episodes: List[AblationEpisode] = []
    for scenario in scenario_rows:
        frozen = scenario.to_dict()
        for arm in arm_rows:
            episode = run_episode(arm, scenario)
            if episode.arm_id != arm.arm_id:
                raise ValueError("run_episode returned an episode for a different arm")
            if episode.scenario_id != scenario.scenario_id or int(episode.seed) != int(scenario.seed):
                raise ValueError("run_episode returned mismatched scenario/seed metadata")
            if scenario.to_dict() != frozen:
                raise ValueError("run_episode mutated the frozen ScenarioManifest")
            episodes.append(episode)
    return AblationReport(arms=arm_rows, scenarios=scenario_rows, episodes=episodes)
