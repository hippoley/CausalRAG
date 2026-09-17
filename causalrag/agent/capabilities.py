from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Explicit causal-runtime feature switches used by research ablations.

    These switches live in the execution runtime, not only in benchmark labels,
    so an ablation cannot accidentally continue using the feature it claims to
    disable.
    """

    causal_selection: bool = True
    causal_updates: bool = True
    bayesian_updates: bool = True
    eig: bool = True
    evsi: bool = True
    temporal_attribution: bool = True
    open_world: bool = True
    retrieval: bool = True

    @classmethod
    def full(cls) -> "RuntimeCapabilities":
        return cls()

    @classmethod
    def vanilla_tool_loop(cls) -> "RuntimeCapabilities":
        """Same outer loop and tools, without the causal control plane."""
        return cls(
            causal_selection=False,
            causal_updates=False,
            bayesian_updates=False,
            eig=False,
            evsi=False,
            temporal_attribution=False,
            open_world=False,
            retrieval=False,
        )

    def with_feature(self, feature: str, enabled: bool) -> "RuntimeCapabilities":
        if feature not in asdict(self):
            raise KeyError(f"unknown runtime capability: {feature}")
        return replace(self, **{feature: bool(enabled)})

    def to_dict(self) -> Dict[str, bool]:
        return {key: bool(value) for key, value in asdict(self).items()}
