from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
    expand_seeded_manifests,
    run_boptest_comparison,
)
from .boptest_protocol import BOPTESTScenarioManifest


STUDY_PROTOCOL_VERSION = "branchpoint.boptest-study.v1"


class BOPTESTStudyPlanError(ValueError):
    """The preregistered BOPTEST study plan is invalid."""


@dataclass(frozen=True)
class BOPTESTArbitrationStudyPlan:
    base_manifest: BOPTESTScenarioManifest
    seeds: Tuple[int, ...]
    proposal_config: TemperatureBandProposalConfig
    reference_controller_id: str = "proposal-order"
    confidence: float = 0.95
    resamples: int = 5000
    bootstrap_seed: int = 20260923
    protocol_version: str = STUDY_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if len(seeds) < 2:
            raise BOPTESTStudyPlanError("study requires at least two seeds")
        if len(set(seeds)) != len(seeds):
            raise BOPTESTStudyPlanError("study seeds must be unique")
        object.__setattr__(self, "seeds", seeds)

        if self.base_manifest.seed is not None:
            raise BOPTESTStudyPlanError(
                "base_manifest.seed must be null; study seeds are declared separately"
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
        seeds = tuple(data.pop("seeds"))
        return cls(
            base_manifest=base_manifest,
            seeds=seeds,
            proposal_config=proposal_config,
            **data,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "base_manifest": self.base_manifest.to_dict(),
            "seeds": list(self.seeds),
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


@dataclass(frozen=True)
class BOPTESTArbitrationStudyResult:
    plan: BOPTESTArbitrationStudyPlan
    comparison: BOPTESTComparisonReport
    uncertainty: BOPTESTBootstrapReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_plan": self.plan.to_dict(),
            "study_hash": self.plan.study_hash,
            "comparison": self.comparison.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
        }


def run_boptest_arbitration_study(
    plan: BOPTESTArbitrationStudyPlan,
    *,
    client_factory: Callable[[], BOPTESTClient] = BOPTESTClient,
) -> BOPTESTArbitrationStudyResult:
    manifests = expand_seeded_manifests(
        plan.base_manifest,
        plan.seeds,
    )
    controllers = temperature_band_controller_specs(
        plan.proposal_config,
    )
    comparison = run_boptest_comparison(
        manifests,
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
