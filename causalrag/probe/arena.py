from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Dict, Optional

from causalrag.agent import RuntimeCapabilities

from .runtime import ProbeRunConfig
from .session import ProbeSession


@dataclass
class ArenaArm:
    label: str
    session: ProbeSession

    def snapshot(self) -> Dict[str, Any]:
        return self.session.snapshot()


class TesterArena:
    """Paired Playable Probe sessions that differ only in runtime capabilities."""

    def __init__(self, config: ProbeRunConfig) -> None:
        self.arena_id = uuid.uuid4().hex
        self.config = config
        baseline_config = replace(
            config,
            capabilities=RuntimeCapabilities.vanilla_tool_loop().to_dict(),
        )
        tester_config = replace(
            config,
            capabilities=config.resolved_capabilities().to_dict(),
        )
        self.baseline = ArenaArm("plain_tool_loop", ProbeSession(baseline_config))
        self.tester = ArenaArm("causal_tester", ProbeSession(tester_config))
        self.baseline.session.start()
        self.tester.session.start()

    @staticmethod
    def _selected(snapshot: Dict[str, Any]) -> Optional[str]:
        pending = snapshot.get("pending_decision")
        if pending:
            return (pending.get("runtime_selected") or {}).get("name")
        result = snapshot.get("result") or {}
        decisions = result.get("decisions") or []
        if decisions:
            return ((decisions[-1].get("selected") or {}).get("name"))
        return None

    @staticmethod
    def _metric_delta(
        baseline: Dict[str, Any],
        tester: Dict[str, Any],
        key: str,
    ) -> Optional[float]:
        left = ((baseline.get("result") or {}).get("metrics") or {}).get(key)
        right = ((tester.get("result") or {}).get("metrics") or {}).get(key)
        if left is None or right is None:
            return None
        return float(right) - float(left)

    def snapshot(self) -> Dict[str, Any]:
        baseline = self.baseline.snapshot()
        tester = self.tester.snapshot()
        baseline_action = self._selected(baseline)
        tester_action = self._selected(tester)
        both_completed = (
            baseline.get("status") == "completed"
            and tester.get("status") == "completed"
        )
        return {
            "arena_id": self.arena_id,
            "config": self.config.to_dict(),
            "comparison_contract": {
                "same_hidden_world": True,
                "same_seed": True,
                "same_outcome_mode": True,
                "same_proposer_family": True,
                "same_provider": True,
                "same_model": True,
                "same_budgets": True,
                "only_runtime_capabilities_differ": True,
                "stochastic_pairing_note": (
                    "Both arms start from the same seed, but once action sequences "
                    "diverge they may consume random draws differently. Use "
                    "deterministic mode for the strictest visual counterfactual."
                ),
            },
            "baseline": baseline,
            "tester": tester,
            "divergence": {
                "baseline_action": baseline_action,
                "tester_action": tester_action,
                "actions_differ": bool(
                    baseline_action
                    and tester_action
                    and baseline_action != tester_action
                ),
            },
            "completed": both_completed,
            "metric_delta_tester_minus_baseline": {
                key: self._metric_delta(baseline, tester, key)
                for key in (
                    "success",
                    "causal_regret",
                    "total_cost",
                    "probes",
                    "interventions",
                    "true_hypothesis_posterior",
                    "brier_score",
                )
            }
            if both_completed
            else None,
        }

    def approve_waiting(self) -> Dict[str, Any]:
        for arm in (self.baseline, self.tester):
            if arm.session.status() == "waiting_for_human":
                arm.session.resolve_decision("approve")
        return self.snapshot()

    def resolve_arm(
        self,
        arm: str,
        action: str,
        candidate_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        target = self.baseline if arm == "baseline" else self.tester if arm == "tester" else None
        if target is None:
            raise ValueError("arm must be baseline or tester")
        target.session.resolve_decision(action, candidate_index)
        return self.snapshot()

    def close(self) -> None:
        self.baseline.session.gate.close()
        self.tester.session.gate.close()


class TesterArenaManager:
    def __init__(self) -> None:
        self._arenas: Dict[str, TesterArena] = {}
        self._lock = RLock()

    def create(self, config: ProbeRunConfig) -> TesterArena:
        arena = TesterArena(config)
        with self._lock:
            self._arenas[arena.arena_id] = arena
        return arena

    def get(self, arena_id: str) -> TesterArena:
        with self._lock:
            arena = self._arenas.get(str(arena_id))
        if arena is None:
            raise KeyError(arena_id)
        return arena

    def delete(self, arena_id: str) -> bool:
        with self._lock:
            arena = self._arenas.pop(str(arena_id), None)
        if arena is None:
            return False
        arena.close()
        return True


ARENA_MANAGER = TesterArenaManager()
