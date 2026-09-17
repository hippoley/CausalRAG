from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RuntimeFeatureFlags:
    """Runtime-owned feature gates used for controlled ablations.

    Defaults preserve the full causal runtime. Disabling a flag changes actual
    decision/runtime behavior; it is not metadata-only.
    """

    causal_runtime: bool = True
    eig: bool = True
    evsi: bool = True
    temporal_attribution: bool = True
    open_world_discovery: bool = True
    retrieval: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)
