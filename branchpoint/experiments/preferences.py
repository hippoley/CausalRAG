from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

from .decision import InterventionContract


@dataclass(frozen=True)
class DecisionPreferences:
    """Deployment-owned consequence preferences for intervention decisions.

    Tool contracts describe what an action can do. Preferences describe how a
    deployment values those outcomes. An override is keyed by tool/action name
    and maps causal hypothesis ids to utility values.

    This first version deliberately keeps capability cost/risk/reversibility on
    ToolSpec. It only separates consequence utility from benchmark or proposer
    code, so the runtime can consume deployment policy without giving the LLM
    authority to rewrite it.
    """

    intervention_utilities: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    descriptions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action_name, utilities in self.intervention_utilities.items():
            if not str(action_name).strip():
                raise ValueError("intervention action names must be non-empty")
            if len(utilities) < 2:
                raise ValueError(
                    f"preference override for {action_name} requires at least two hypotheses"
                )
            if any(not str(hypothesis_id).strip() for hypothesis_id in utilities):
                raise ValueError("hypothesis ids must be non-empty")

    def contract_for(
        self,
        action_name: str,
        fallback: Optional[InterventionContract] = None,
    ) -> Optional[InterventionContract]:
        utilities = self.intervention_utilities.get(str(action_name))
        if utilities is None:
            return fallback
        return InterventionContract(
            intervention_id=str(action_name),
            utilities={
                str(hypothesis_id): float(value)
                for hypothesis_id, value in utilities.items()
            },
            description=str(
                self.descriptions.get(
                    str(action_name),
                    "Deployment-owned intervention utility override.",
                )
            ),
        )

    def apply_to_tool(self, tool):
        """Return a ToolSpec-like dataclass with effective intervention utility."""
        contract = self.contract_for(
            getattr(tool, "name", ""),
            getattr(tool, "intervention_contract", None),
        )
        if contract is getattr(tool, "intervention_contract", None):
            return tool
        return replace(tool, intervention_contract=contract)

    def summary(self):
        """Safe runtime summary: action/hypothesis domains, not hidden truth."""
        return {
            "interventions": {
                str(action_name): [str(hypothesis_id) for hypothesis_id in utilities]
                for action_name, utilities in self.intervention_utilities.items()
            }
        }
