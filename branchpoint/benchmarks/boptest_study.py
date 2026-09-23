from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Tuple

from branchpoint.environments import BOPTESTClient

from .boptest_branchpoint import (
    TemperatureBandProposalConfig,
    temperature_band_controller_specs,
)
from .boptest_comparison import (
    BOPTESTBootstrapReport,
    BOPTESTComparisonReport,
    bootstrap_paired_kpi_intervals,
    run_boptest_comparison,
)
from .boptest_protocol import BOPTESTScenarioManifest


STUDY_PROTOCOL_VERSION = "branchpoint.boptest-study.v1"


class BOPTESTStudyPlanError(ValueError):
    """The preregistered BOPTEST study plan is invalid."""


@dataclass(frozen=True)
class BOPTESTStudyPeriod:
    period_id: str
    day_index: int
    start_hour: float = 0.0

    def __post_init__(self) -> None:
        period_id = str(self.period_id).strip()
        if not period_id:
            raise BOPTESTStudyPlanError("period_id must be non-empty")
        day_index = int(self.day_index)
        start_hour = float(self.start_hour)
        if day_index < 0 or day_index > 365:
            raise BOPTESTStudyPlanError("day_index must be between 0 and 365")
        if not 0.0 <= start_hour < 24.0:
            raise BOPTESTStudyPlanError("start_hour must be in [0, 24)")
        object.__setattr__(self, "period_id", period_id)
        object.__setattr__(self, "day_index", day_index)
        object.__setattr__(self, "start_hour", start_hour)

    @property
    def start_time(self) -> float:
        return float(self.day_index * 86400) + self.start_hour * 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_id": self.period_id,
            "day_index": self.day_index,
            "start_hour": self.start_hour,
            "start_time": self.start_time,
        }


@dataclass(frozen=True)
class BOPTESTArbitrationStudyPlan:
    base_manifest: BOPTESTScenarioManifest
    periods: Tuple[BOPTESTStudyPeriod, ...]
    proposal_config: TemperatureBandProposalConfig
    reference_controller_id: str = "proposal-order"
    confidence: float = 0.95
    resamples: int = 5000
    bootstrap_seed: int = 20260923
    protocol_version: str = STUDY_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        periods = tuple(self.periods)
        if len(periods) < 2:
            raise BOPTESTStudyPlanError("study requires at least two periods")
        ids = [period.period_id for period in periods]
        if len(set(ids)) != len(ids):
            raise BOPTESTStudyPlanError("study period ids must be unique")
        starts = [float(period.start_time) for period in periods]
        if len(set(starts)) != len(starts):
            raise BOPTESTStudyPlanError("study period start times must be unique")
        object.__setattr__(self, "periods", periods)

        if self.base_manifest.seed is not None:
            raise BOPTESTStudyPlanError(
                "base_manifest.seed must be null for this non-forecast controller study"
            )
        if (
            self.base_manifest.temperature_uncertainty is not None
            or self.base_manifest.solar_uncertainty is not None
        ):
            raise BOPTESTStudyPlanError(
                "forecast uncertainty must be disabled because this proposer does not consume forecasts"
            )

        reference = str(self.reference_controller_id).strip()
        if reference not in {"proposal-order", "branchpoint"}:
            raise BOPTESTStudyPlanError(
                "reference_controller_id must be proposal-order or branchpoint"
            )
        object.__setattr__(self, "reference_controller_id", reference)

        if not 0.0 < float(self.confidence) < 1.0:
            raise BOPTESTStudyPlanError("confidence must be between 0 and 1")
        if int(self.resamples) < 100:
            raise BOPTESTStudyPlanError("resamples must be at least 100")
        if str(self.protocol_version) != STUDY_PROTOCOL_VERSION:
            raise BOPTESTStudyPlanError(
                f"unsupported protocol_version {self.protocol_version!r}"
            )

        required_controls = {
            self.proposal_config.heating_setpoint_input,
            self.proposal_config.cooling_setpoint_input,
        }
        missing_controls = sorted(
            required_controls - set(self.base_manifest.controlled_inputs)
        )
        if missing_controls:
            raise BOPTESTStudyPlanError(
                "base manifest is missing proposal control(s): "
                + ", ".join(missing_controls)
            )

        if (
            self.proposal_config.temperature_measurement
            not in set(self.base_manifest.measurement_points)
        ):
            raise BOPTESTStudyPlanError(
                "base manifest must record the proposal temperature measurement"
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "BOPTESTArbitrationStudyPlan":
        data = dict(payload)
        base_manifest = BOPTESTScenarioManifest.from_dict(
            data.pop("base_manifest")
        )
        proposal_config = TemperatureBandProposalConfig(
            **dict(data.pop("proposal_config"))
        )
        periods = []
        for raw in data.pop("periods"):
            row = dict(raw)
            row.pop("start_time", None)
            periods.append(BOPTESTStudyPeriod(**row))
        periods = tuple(periods)
        return cls(
            base_manifest=base_manifest,
            periods=periods,
            proposal_config=proposal_config,
            **data,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "base_manifest": self.base_manifest.to_dict(),
            "periods": [period.to_dict() for period in self.periods],
            "proposal_config": {
                "lower_kelvin": float(self.proposal_config.lower_kelvin),
                "upper_kelvin": float(self.proposal_config.upper_kelvin),
                "temperature_measurement": self.proposal_config.temperature_measurement,
                "heating_setpoint_input": self.proposal_config.heating_setpoint_input,
                "cooling_setpoint_input": self.proposal_config.cooling_setpoint_input,
                "intervention_goal_gain": float(
                    self.proposal_config.intervention_goal_gain
                ),
                "embedded_goal_gain": float(
                    self.proposal_config.embedded_goal_gain
                ),
                "intervention_risk": float(
                    self.proposal_config.intervention_risk
                ),
                "intervention_cost": float(
                    self.proposal_config.intervention_cost
                ),
            },
            "reference_controller_id": self.reference_controller_id,
            "confidence": float(self.confidence),
            "resamples": int(self.resamples),
            "bootstrap_seed": int(self.bootstrap_seed),
        }

    @property
    def study_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def manifests(self) -> Tuple[BOPTESTScenarioManifest, ...]:
        return tuple(
            replace(
                self.base_manifest,
                start_time=float(period.start_time),
                seed=None,
            )
            for period in self.periods
        )


@dataclass(frozen=True)
class BOPTESTArbitrationStudyResult:
    plan: BOPTESTArbitrationStudyPlan
    comparison: BOPTESTComparisonReport
    uncertainty: BOPTESTBootstrapReport

    def to_dict(self) -> Dict[str, Any]:
        manifests = self.plan.manifests()
        return {
            "study_plan": self.plan.to_dict(),
            "study_hash": self.plan.study_hash,
            "period_manifest_hashes": {
                period.period_id: manifest.manifest_hash
                for period, manifest in zip(self.plan.periods, manifests)
            },
            "comparison": self.comparison.to_dict(),
            "uncertainty": {
                **self.uncertainty.to_dict(),
                "interpretation": (
                    "descriptive robustness interval over a purposively selected "
                    "set of official BESTEST Air regimes; not a random-sampling "
                    "population confidence interval"
                ),
            },
        }


def run_boptest_arbitration_study(
    plan: BOPTESTArbitrationStudyPlan,
    *,
    client_factory: Callable[[], BOPTESTClient] = BOPTESTClient,
) -> BOPTESTArbitrationStudyResult:
    controllers = temperature_band_controller_specs(
        plan.proposal_config,
    )
    comparison = run_boptest_comparison(
        plan.manifests(),
        controllers,
        reference_controller_id=plan.reference_controller_id,
        client_factory=client_factory,
    )
    uncertainty = bootstrap_paired_kpi_intervals(
        comparison,
        confidence=plan.confidence,
        resamples=plan.resamples,
        seed=plan.bootstrap_seed,
        min_pairs=2,
    )
    return BOPTESTArbitrationStudyResult(
        plan=plan,
        comparison=comparison,
        uncertainty=uncertainty,
    )
