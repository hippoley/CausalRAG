from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from branchpoint.agent.actions import ActionKind, CandidateAction
from branchpoint.agent.capabilities import RuntimeCapabilities
from branchpoint.decision import decide
from branchpoint.tools import ToolSpec
from branchpoint.world_model import CausalWorldModel

from .boptest_protocol import BOPTESTControlContext, Controller


class BOPTESTDecisionAdapterError(RuntimeError):
    """Invalid proposal/control mapping at the BOPTEST decision boundary."""


@dataclass(frozen=True)
class BOPTESTControlCandidate:
    candidate: CandidateAction
    controls: Mapping[str, Any]
    tool: Optional[ToolSpec] = None

    def __post_init__(self) -> None:
        if self.tool is not None and self.tool.name != self.candidate.name:
            raise BOPTESTDecisionAdapterError(
                "ToolSpec name must match its CandidateAction name"
            )


ProposalProvider = Callable[[BOPTESTControlContext], Sequence[BOPTESTControlCandidate]]


@dataclass(frozen=True)
class BOPTESTControlDecision:
    step_index: int
    mode: str
    proposer_first: str
    selected: str
    changed_proposer_order: bool
    controls: Mapping[str, Any]
    scores: Tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "mode": self.mode,
            "proposer_first": self.proposer_first,
            "selected": self.selected,
            "changed_proposer_order": self.changed_proposer_order,
            "controls": dict(self.controls),
            "scores": [dict(score) for score in self.scores],
        }


def _plans_for_context(
    provider: ProposalProvider,
    context: BOPTESTControlContext,
) -> Tuple[BOPTESTControlCandidate, ...]:
    plans = tuple(provider(context))
    if not plans:
        raise BOPTESTDecisionAdapterError("proposal provider returned no candidates")
    names = [str(plan.candidate.name) for plan in plans]
    if len(set(names)) != len(names):
        raise BOPTESTDecisionAdapterError("candidate names must be unique per step")
    return plans


class ProposalOrderBOPTESTController:
    """Execute the proposer-first control mapping without Branchpoint arbitration."""

    def __init__(self, proposal_provider: ProposalProvider) -> None:
        self.proposal_provider = proposal_provider
        self.decisions: list[BOPTESTControlDecision] = []

    def __call__(self, context: BOPTESTControlContext) -> Mapping[str, Any]:
        plans = _plans_for_context(self.proposal_provider, context)
        selected = plans[0]
        controls = dict(selected.controls)
        self.decisions.append(
            BOPTESTControlDecision(
                step_index=context.step_index,
                mode="proposal_order",
                proposer_first=selected.candidate.name,
                selected=selected.candidate.name,
                changed_proposer_order=False,
                controls=controls,
            )
        )
        return controls


class BranchpointBOPTESTController:
    """Arbitrate the same proposals through Branchpoint before returning controls."""

    def __init__(
        self,
        proposal_provider: ProposalProvider,
        *,
        world_model: Optional[CausalWorldModel] = None,
        capabilities: Optional[RuntimeCapabilities] = None,
    ) -> None:
        self.proposal_provider = proposal_provider
        self.world_model = world_model
        self.capabilities = capabilities
        self.decisions: list[BOPTESTControlDecision] = []

    def __call__(self, context: BOPTESTControlContext) -> Mapping[str, Any]:
        plans = _plans_for_context(self.proposal_provider, context)
        candidates = [plan.candidate for plan in plans]
        tools = [plan.tool for plan in plans if plan.tool is not None]
        result = decide(
            candidates,
            tools=tools,
            world_model=self.world_model,
            capabilities=self.capabilities,
        )
        by_identity = {id(plan.candidate): plan for plan in plans}
        try:
            selected = by_identity[id(result.selected)]
        except KeyError as exc:
            raise BOPTESTDecisionAdapterError(
                "Branchpoint selected a candidate outside the proposal set"
            ) from exc
        controls = dict(selected.controls)
        self.decisions.append(
            BOPTESTControlDecision(
                step_index=context.step_index,
                mode="branchpoint",
                proposer_first=result.proposer_first.name,
                selected=result.selected.name,
                changed_proposer_order=result.changed_proposer_order,
                controls=controls,
                scores=tuple(asdict(score) for score in result.scores),
            )
        )
        return controls


@dataclass(frozen=True)
class TemperatureBandProposalConfig:
    """Transparent benchmark proposer with BESTEST Air signal-name defaults."""

    lower_kelvin: float
    upper_kelvin: float
    temperature_measurement: str = "zon_reaTRooAir_y"
    heating_setpoint_input: str = "con_oveTSetHea_u"
    cooling_setpoint_input: str = "con_oveTSetCoo_u"
    intervention_goal_gain: float = 0.70
    embedded_goal_gain: float = 0.20
    intervention_risk: float = 0.10
    intervention_cost: float = 0.02

    def __post_init__(self) -> None:
        lower = float(self.lower_kelvin)
        upper = float(self.upper_kelvin)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("temperature band must be finite with lower < upper")
        for name in (
            "intervention_goal_gain",
            "embedded_goal_gain",
            "intervention_risk",
            "intervention_cost",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def heating_activate_input(self) -> str:
        return self.heating_setpoint_input.removesuffix("_u") + "_activate"

    @property
    def cooling_activate_input(self) -> str:
        return self.cooling_setpoint_input.removesuffix("_u") + "_activate"


def temperature_band_proposal_provider(
    config: TemperatureBandProposalConfig,
) -> ProposalProvider:
    """Create one shared proposer for proposal-order and Branchpoint arms."""

    def provider(context: BOPTESTControlContext) -> Sequence[BOPTESTControlCandidate]:
        try:
            temperature = float(context.observation[config.temperature_measurement])
        except KeyError as exc:
            raise BOPTESTDecisionAdapterError(
                f"missing temperature measurement {config.temperature_measurement!r}"
            ) from exc
        if not math.isfinite(temperature):
            raise BOPTESTDecisionAdapterError("temperature measurement must be finite")

        embedded_controls = {
            config.heating_activate_input: 0,
            config.cooling_activate_input: 0,
        }
        embedded = BOPTESTControlCandidate(
            candidate=CandidateAction(
                ActionKind.WAIT,
                "embedded_control",
                expected_goal_gain=float(config.embedded_goal_gain),
                rationale="Leave the BOPTEST embedded controller in charge.",
            ),
            controls=embedded_controls,
        )

        if config.lower_kelvin <= temperature <= config.upper_kelvin:
            return (embedded,)

        override_controls = {
            config.heating_setpoint_input: float(config.lower_kelvin),
            config.heating_activate_input: 1,
            config.cooling_setpoint_input: float(config.upper_kelvin),
            config.cooling_activate_input: 1,
        }
        override = BOPTESTControlCandidate(
            candidate=CandidateAction(
                ActionKind.INTERVENE,
                "apply_temperature_band",
                expected_goal_gain=float(config.intervention_goal_gain),
                rationale="Apply the explicit heating/cooling temperature band.",
            ),
            controls=override_controls,
            tool=ToolSpec(
                "apply_temperature_band",
                "Override BOPTEST heating and cooling setpoints for one control step.",
                lambda: None,
                risk=float(config.intervention_risk),
                cost=float(config.intervention_cost),
                reversible=True,
                metadata={"kind": "boptest_control"},
            ),
        )
        return (override, embedded)

    return provider
