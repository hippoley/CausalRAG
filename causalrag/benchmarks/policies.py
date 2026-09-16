from __future__ import annotations

import random
from typing import Optional, Sequence

from causalrag.agent.actions import ActionKind, CandidateAction


class _BaseHiddenWorldPolicy:
    policy_id = "base"

    def __init__(
        self,
        scenario,
        confidence_threshold: float = 0.8,
        max_probes: int = 3,
        min_probes: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.scenario = scenario
        self.confidence_threshold = confidence_threshold
        self.max_probes = max_probes
        self.min_probes = min_probes
        self._rng = random.Random(seed)

    def _latest_is_intervention(self, state) -> bool:
        if not state.observations:
            return False
        latest = state.observations[-1].result
        return isinstance(latest, dict) and "success" in latest

    def _stop_after_intervention(self, state):
        latest = state.observations[-1].result
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={
                    "answer": "HiddenWorld intervention succeeded."
                    if latest["success"]
                    else "HiddenWorld intervention failed."
                },
                rationale="Intervention outcome observed.",
            )
        ]

    def _leader(self, world_model):
        return max(
            world_model.hypotheses(include_rejected=False),
            key=lambda item: item.probability,
        )

    def _probe_count(self, state) -> int:
        return sum(
            1
            for observation in state.observations
            if observation.action_name in self.scenario.experiments
        )

    def _should_intervene(self, state, world_model) -> bool:
        leader = self._leader(world_model)
        probes = self._probe_count(state)
        enough_confidence = (
            probes >= self.min_probes
            and leader.probability >= self.confidence_threshold
        )
        return enough_confidence or probes >= self.max_probes

    def _intervention(self, world_model):
        leader = self._leader(world_model)
        intervention = next(
            name
            for name, target in self.scenario.interventions.items()
            if target == leader.hypothesis_id
        )
        return [
            CandidateAction(
                kind=ActionKind.INTERVENE,
                name=intervention,
                expected_goal_gain=1.0,
                rationale=f"Act on leading hypothesis {leader.hypothesis_id}.",
            )
        ]

    def uncertainty(self, state, world_model) -> str:
        active = sorted(
            world_model.hypotheses(include_rejected=False),
            key=lambda item: item.probability,
            reverse=True,
        )
        return " vs ".join(
            f"{item.hypothesis_id}={item.probability:.3f}" for item in active
        )

    def _experiment_action(self, name: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.OBSERVE,
            name=name,
            expected_information_gain=0.0,
            tests_hypotheses=list(self.scenario.hypotheses),
            rationale=f"Policy {self.policy_id} proposes diagnostic experiment {name}.",
        )


class GreedyEIGPolicy(_BaseHiddenWorldPolicy):
    """Expose all experiments and let runtime Bayesian EIG/cost rank them."""

    policy_id = "greedy_eig"

    def propose(self, state, world_model) -> Sequence[CandidateAction]:
        if self._latest_is_intervention(state):
            return self._stop_after_intervention(state)
        if self._should_intervene(state, world_model):
            return self._intervention(world_model)
        return [self._experiment_action(name) for name in self.scenario.experiments]


class ConservativeEIGPolicy(GreedyEIGPolicy):
    """Require repeated evidence before intervention in noisy worlds."""

    policy_id = "conservative_eig"

    def __init__(
        self,
        scenario,
        confidence_threshold: float = 0.9,
        max_probes: int = 4,
        min_probes: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            scenario,
            confidence_threshold=confidence_threshold,
            max_probes=max_probes,
            min_probes=min_probes,
            seed=seed,
        )


class CheapestProbePolicy(_BaseHiddenWorldPolicy):
    """Always choose the cheapest available diagnostic until intervention."""

    policy_id = "cheapest_probe"

    def propose(self, state, world_model) -> Sequence[CandidateAction]:
        if self._latest_is_intervention(state):
            return self._stop_after_intervention(state)
        if self._should_intervene(state, world_model):
            return self._intervention(world_model)
        name = min(
            self.scenario.experiments,
            key=lambda value: self.scenario.experiment_costs.get(value, 0.05),
        )
        return [self._experiment_action(name)]


class RandomProbePolicy(_BaseHiddenWorldPolicy):
    """Random diagnostic baseline with a seeded policy RNG."""

    policy_id = "random_probe"

    def propose(self, state, world_model) -> Sequence[CandidateAction]:
        if self._latest_is_intervention(state):
            return self._stop_after_intervention(state)
        if self._should_intervene(state, world_model):
            return self._intervention(world_model)
        name = self._rng.choice(list(self.scenario.experiments))
        return [self._experiment_action(name)]
