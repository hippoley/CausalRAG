from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Optional

from causalrag.experiments import (
    ModelMismatchPolicy,
    apply_experiment_observation,
    assess_model_mismatch,
    expanded_experiment_contract,
    maybe_resolve_model_mismatch,
)
from causalrag.reasoning.policy import rank_actions
from causalrag.tools.base import ToolRegistry
from causalrag.world_model.models import CausalWorldModel, Evidence, Transition

from .actions import ActionKind, CandidateAction, DecisionRecord
from .capabilities import RuntimeCapabilities
from .state import AgentState, Observation
from .temporal import TimeDriver, VirtualTimeDriver, pending_effect_from_contract

BeliefUpdater = Callable[[AgentState, CausalWorldModel, DecisionRecord, Observation], None]
HypothesisUpdater = Callable[[AgentState, CausalWorldModel, DecisionRecord, Observation], None]
GoalEvaluator = Callable[[AgentState, CausalWorldModel], bool]
class DecisionGateReplan(RuntimeError):
    """Signal that a paused, not-yet-executed decision must be recomputed."""


DecisionGate = Callable[[AgentState, CausalWorldModel, DecisionRecord], Optional[CandidateAction]]


def _candidate_trace_payload(candidate: CandidateAction) -> dict[str, Any]:
    return {
        "kind": candidate.kind.value,
        "name": candidate.name,
        "arguments": dict(candidate.arguments),
        "expected_goal_gain": float(candidate.expected_goal_gain),
        "expected_information_gain": float(candidate.expected_information_gain),
        "cost": float(candidate.cost),
        "risk": float(candidate.risk),
        "irreversibility": float(candidate.irreversibility),
        "rationale": candidate.rationale,
        "tests_hypotheses": list(candidate.tests_hypotheses),
        "falsification_target": candidate.falsification_target,
    }


def _hypothesis_trace_payload(proposal: Any) -> dict[str, Any]:
    if isinstance(proposal, dict):
        return dict(proposal)
    return {
        "id": getattr(proposal, "hypothesis_id", None),
        "statement": getattr(proposal, "statement", ""),
        "probability": getattr(proposal, "probability", None),
        "rationale": getattr(proposal, "rationale", ""),
        "falsifiers": list(getattr(proposal, "falsifiers", []) or []),
        "experiment_predictions": dict(
            getattr(proposal, "experiment_predictions", {}) or {}
        ),
    }


class CausalAgentLoop:
    """Thin runtime for goal-directed causal learning and action."""

    def __init__(
        self,
        reasoner,
        tools: Optional[ToolRegistry] = None,
        world_model: Optional[CausalWorldModel] = None,
        belief_updater: Optional[BeliefUpdater] = None,
        hypothesis_updater: Optional[HypothesisUpdater] = None,
        goal_evaluator: Optional[GoalEvaluator] = None,
        time_driver: Optional[TimeDriver] = None,
        mismatch_policy: Optional[ModelMismatchPolicy] = None,
        capabilities: Optional[RuntimeCapabilities] = None,
        decision_gate: Optional[DecisionGate] = None,
    ) -> None:
        self.reasoner = reasoner
        self.tools = tools or ToolRegistry()
        self.world_model = world_model or CausalWorldModel()
        self.belief_updater = belief_updater
        self.hypothesis_updater = hypothesis_updater
        self.goal_evaluator = goal_evaluator
        self.time_driver = time_driver or VirtualTimeDriver()
        self.mismatch_policy = mismatch_policy or ModelMismatchPolicy()
        self.capabilities = capabilities or RuntimeCapabilities.full()
        self.decision_gate = decision_gate

    def _sync_state_time(self, state: AgentState) -> None:
        state.virtual_time_seconds = float(self.time_driver.now_seconds)
        for effect in state.pending_effects:
            effect.refresh(state.virtual_time_seconds)

    def _record_expired_effects(self, state: AgentState) -> None:
        if not self.capabilities.temporal_attribution:
            return
        now = state.virtual_time_seconds
        for effect in state.pending_effects:
            effect.refresh(now)
            if not effect.expired or effect.expiry_recorded:
                continue
            event = {
                "kind": "missed_observation_window",
                "effect_id": effect.effect_id,
                "intervention": effect.intervention,
                "observe_with": effect.observe_with,
                "ready_at": effect.ready_at,
                "expires_at": effect.expires_at,
                "detected_at": now,
            }
            state.scratch.setdefault("temporal_events", []).append(event)
            self.world_model.record_transition(
                Transition(
                    action="missed_observation_window",
                    arguments={"effect_id": effect.effect_id},
                    observation=event,
                    expected_effects={
                        "temporal_failure": True,
                        "belief_update": False,
                    },
                )
            )
            effect.expiry_recorded = True

    def _observation_for_effect(self, effect, rationale: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.OBSERVE,
            name=effect.observe_with,
            arguments=dict(effect.observe_arguments),
            rationale=rationale,
        )

    def _wait_for_effect(self, effect, state: AgentState, rationale: str) -> CandidateAction:
        return CandidateAction(
            kind=ActionKind.WAIT,
            name="wait_for_effect_window",
            arguments={"seconds": effect.seconds_until_ready(state.virtual_time_seconds)},
            rationale=rationale,
        )

    def _protected_effect(self, state: AgentState):
        effects = [
            effect
            for effect in state.active_pending_effects()
            if effect.protect_attribution
        ]
        if not effects:
            return None
        return min(effects, key=lambda effect: (effect.expires_at, effect.ready_at))

    def _temporal_guard(self, selected: CandidateAction, state: AgentState) -> CandidateAction:
        if not self.capabilities.temporal_attribution:
            return selected
        now = state.virtual_time_seconds

        matching = [
            effect
            for effect in state.active_pending_effects()
            if effect.observe_with == selected.name and effect.is_premature(now)
        ]
        if matching:
            effect = min(matching, key=lambda item: item.ready_at)
            return self._wait_for_effect(
                effect,
                state,
                f"Runtime temporal guard: {selected.name} is premature; wait for the causal observation window.",
            )

        protected = self._protected_effect(state)
        if protected is None:
            return selected

        if selected.kind in (ActionKind.INTERVENE, ActionKind.STOP):
            if protected.is_ready(now):
                return self._observation_for_effect(
                    protected,
                    "Runtime attribution guard: observe the unresolved intervention effect before another intervention or stop.",
                )
            return self._wait_for_effect(
                protected,
                state,
                "Runtime attribution guard: wait for the unresolved intervention effect before another intervention or stop.",
            )

        if selected.kind == ActionKind.WAIT:
            if protected.is_ready(now):
                return self._observation_for_effect(
                    protected,
                    "Runtime attribution guard: the protected effect is ready; observe it before waiting longer.",
                )
            requested = float(selected.arguments.get("seconds", 0) or 0)
            until_ready = protected.seconds_until_ready(now)
            if requested <= 0.0 or requested > until_ready:
                return self._wait_for_effect(
                    protected,
                    state,
                    "Runtime attribution guard: cap WAIT at the first valid observation time.",
                )

        return selected

    def _schedule_temporal_effect(self, tool_spec, selected: CandidateAction, state: AgentState) -> None:
        if not self.capabilities.temporal_attribution:
            return
        contract = None if tool_spec is None else tool_spec.temporal_effect_contract
        if selected.kind != ActionKind.INTERVENE or contract is None:
            return
        effect = pending_effect_from_contract(
            selected.name,
            contract,
            state.virtual_time_seconds,
        )
        active = self.world_model.hypotheses(include_rejected=False)
        if active:
            prediction = max(active, key=lambda item: item.probability)
            effect.metadata["prediction_hypothesis"] = prediction.hypothesis_id
            effect.metadata["prediction_probability"] = prediction.probability
            effect.metadata["expected_outcome"] = effect.expected_for(prediction.hypothesis_id)
        state.schedule_effect(effect)

    def _evaluate_temporal_observation(self, selected: CandidateAction, result, state: AgentState):
        if not self.capabilities.temporal_attribution:
            return []
        evaluations = []
        now = state.virtual_time_seconds
        for effect in state.pending_effects:
            effect.refresh(now)
            if effect.observed or effect.expired or effect.observe_with != selected.name:
                continue
            if not effect.is_ready(now):
                continue
            if not isinstance(result, dict) or effect.observation_key not in result:
                continue
            observed = result[effect.observation_key]
            hypothesis_id = effect.metadata.get("prediction_hypothesis")
            expected = effect.expected_for(str(hypothesis_id)) if hypothesis_id else None
            matched = expected is None or observed == expected
            effect.observed = True
            effect.observed_value = observed
            effect.matched_prediction = matched
            evaluation = {
                "effect_id": effect.effect_id,
                "intervention": effect.intervention,
                "prediction_hypothesis": hypothesis_id,
                "expected": expected,
                "observed": observed,
                "matched_prediction": matched,
                "lag_seconds": now - effect.started_at,
                "within_window": effect.ready_at <= now <= effect.expires_at,
            }
            evaluations.append(evaluation)
            if hypothesis_id and expected is not None and self.capabilities.causal_updates:
                weight = effect.falsification_weight if matched else -effect.falsification_weight
                self.world_model.update_hypothesis(
                    str(hypothesis_id),
                    Evidence(
                        source=selected.name,
                        statement=(
                            f"Temporal effect {effect.effect_id}: expected {expected!r}, "
                            f"observed {observed!r} after {now - effect.started_at:.3f}s."
                        ),
                        weight=weight,
                        kind="temporal_intervention_outcome",
                        metadata=evaluation,
                    ),
                )
        return evaluations

    def _discover_after_mismatch(self, state: AgentState, assessment) -> list[str]:
        if not self.capabilities.open_world:
            return []
        discover = getattr(self.reasoner, "discover_hypotheses", None)
        if not callable(discover) or assessment is None or not assessment.escalate:
            return []

        unresolved_provisional = [
            hypothesis
            for hypothesis in self.world_model.hypotheses(include_rejected=False)
            if hypothesis.origin == "discovered" and not hypothesis.validated
        ]
        if unresolved_provisional:
            return []

        context = {
            "experiment_id": assessment.experiment_id,
            "outcome": assessment.outcome,
            "predictive_probability": assessment.predictive_probability,
            "surprisal": assessment.surprisal,
            "hard_mismatch": assessment.hard_mismatch,
            "recent_mismatches": self.world_model.snapshot().get("open_world", {}).get("recent_mismatches", []),
        }
        proposals = list(discover(state, self.world_model, context))
        added: list[str] = []
        for proposal in proposals:
            hypothesis = self.world_model.add_discovered_hypothesis(
                hypothesis_id=getattr(proposal, "hypothesis_id", ""),
                statement=getattr(proposal, "statement", ""),
                rationale=getattr(proposal, "rationale", ""),
                falsifiers=getattr(proposal, "falsifiers", []),
                experiment_predictions=getattr(proposal, "experiment_predictions", {}),
                initial_probability=self.mismatch_policy.discovered_initial_probability,
            )
            if hypothesis is not None:
                added.append(hypothesis.hypothesis_id)

        if added:
            state.scratch.setdefault("hypothesis_discovery_events", []).append(
                {
                    "step": state.step,
                    "trigger": context,
                    "hypotheses": list(added),
                }
            )
        return added

    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        state = AgentState(goal=goal, max_steps=max_steps)
        state.scratch["runtime_capabilities"] = self.capabilities.to_dict()
        self._sync_state_time(state)
        while not state.done and state.step < state.max_steps:
            self._sync_state_time(state)
            self._record_expired_effects(state)
            if self.goal_evaluator and self.goal_evaluator(state, self.world_model):
                state.done = True
                state.stop_reason = "goal_reached"
                break

            proposal_started = perf_counter()
            candidates = list(self.reasoner.propose(state, self.world_model))
            proposal_duration_ms = max(
                0.0, (perf_counter() - proposal_started) * 1000.0
            )
            proposal_method = getattr(self.reasoner, "hypothesis_proposals", None)
            if self.capabilities.causal_updates and callable(proposal_method):
                hypothesis_proposals = list(proposal_method(state, self.world_model))
                state.scratch["last_hypothesis_proposals"] = hypothesis_proposals
                self.world_model.sync_hypotheses(hypothesis_proposals)
            else:
                hypothesis_proposals = []
                state.scratch["last_hypothesis_proposals"] = []

            metadata_method = getattr(self.reasoner, "proposal_metadata", None)
            proposer_metadata: dict[str, Any] = {}
            if callable(metadata_method):
                try:
                    raw_metadata = metadata_method() or {}
                    proposer_metadata = (
                        dict(raw_metadata)
                        if isinstance(raw_metadata, dict)
                        else {"audit_error": "proposal_metadata returned a non-dict value"}
                    )
                except Exception as exc:
                    proposer_metadata = {
                        "audit_error": f"{type(exc).__name__}: {exc}",
                    }
            proposer_traces = state.scratch.setdefault("proposer_traces", [])
            attempt = 1 + sum(
                1
                for row in proposer_traces
                if int(row.get("step", -1)) == int(state.step)
            )
            proposer_trace = {
                "step": int(state.step),
                "attempt": int(attempt),
                "reasoner_class": type(self.reasoner).__name__,
                "duration_ms": proposal_duration_ms,
                "kind": proposer_metadata.get("kind", "deterministic"),
                "provider": proposer_metadata.get("provider"),
                "model": proposer_metadata.get("model"),
                "usage": dict(proposer_metadata.get("usage") or {}),
                "ok": proposer_metadata.get("ok", True),
                "error": proposer_metadata.get("error") or proposer_metadata.get("audit_error"),
                "structured_payload": proposer_metadata.get("structured_payload"),
                "hypothesis_proposals": [
                    _hypothesis_trace_payload(item)
                    for item in hypothesis_proposals
                ],
                "candidates": [
                    _candidate_trace_payload(candidate)
                    for candidate in candidates
                ],
            }
            proposer_traces.append(proposer_trace)
            state.scratch["last_proposer_trace"] = proposer_trace

            telemetry = getattr(self.tools, "telemetry", None)
            if telemetry is not None:
                telemetry.event(
                    "causalrag.proposer.submitted",
                    {
                        "causalrag.step": int(state.step),
                        "causalrag.proposer.attempt": int(attempt),
                        "causalrag.proposer.kind": proposer_trace["kind"],
                        "causalrag.proposer.provider": proposer_trace["provider"] or "",
                        "causalrag.proposer.model": proposer_trace["model"] or "",
                        "causalrag.proposer.duration_ms": proposal_duration_ms,
                        "causalrag.proposer.ok": bool(proposer_trace["ok"]),
                        "causalrag.proposer.candidate_count": len(candidates),
                        "causalrag.proposer.hypothesis_count": len(hypothesis_proposals),
                        "gen_ai.usage.input_tokens": proposer_trace["usage"].get("input_tokens", 0),
                        "gen_ai.usage.output_tokens": proposer_trace["usage"].get("output_tokens", 0),
                    },
                )

            if self.capabilities.causal_selection:
                ranked = rank_actions(
                    candidates,
                    world_model=self.world_model,
                    tools=self.tools,
                    capabilities=self.capabilities,
                )
                if ranked:
                    proposed_selected = ranked[0][0]
                    selected_score = ranked[0][1]
                    action_scores = [score for _action, score in ranked]
                else:
                    proposed_selected = CandidateAction(
                        kind=ActionKind.STOP,
                        name="stop",
                        rationale="No valid candidate actions were proposed.",
                    )
                    selected_score = None
                    action_scores = []
            else:
                proposed_selected = (
                    candidates[0]
                    if candidates
                    else CandidateAction(
                        kind=ActionKind.STOP,
                        name="stop",
                        rationale="No valid candidate actions were proposed.",
                    )
                )
                selected_score = None
                action_scores = []

            selected = self._temporal_guard(proposed_selected, state)
            if selected is not proposed_selected:
                selected_score = None

            uncertainty = self.reasoner.uncertainty(state, self.world_model)
            decision = DecisionRecord(
                step=state.step,
                uncertainty=uncertainty,
                candidates=candidates,
                selected=selected,
                beliefs_before=self.world_model.snapshot(),
                rationale=selected.rationale,
                action_scores=action_scores,
            )
            state.decisions.append(decision)

            # Optional human/control-plane gate. The runtime has already proposed,
            # scored, and temporally guarded an action, but no tool has executed yet.
            # A gate may block for external approval and may return one of the
            # candidate actions as an explicit override. Runtime temporal safety is
            # re-applied to overrides before execution.
            if self.decision_gate is not None:
                try:
                    override = self.decision_gate(state, self.world_model, decision)
                except DecisionGateReplan:
                    # The decision has not executed yet, so it is safe to discard
                    # this preview and recompute from the mutated world model.
                    state.decisions.pop()
                    state.scratch.setdefault("decision_gate_events", []).append(
                        {"step": state.step, "kind": "replan"}
                    )
                    continue
                if override is not None:
                    selected = self._temporal_guard(override, state)
                    selected_score = next(
                        (
                            score
                            for score in action_scores
                            if score.action_name == selected.name
                            and score.action_kind == selected.kind
                        ),
                        None,
                    )
                    decision = DecisionRecord(
                        step=decision.step,
                        uncertainty=decision.uncertainty,
                        candidates=decision.candidates,
                        selected=selected,
                        beliefs_before=decision.beliefs_before,
                        rationale=selected.rationale or decision.rationale,
                        action_scores=decision.action_scores,
                    )
                    state.decisions[-1] = decision
                    state.scratch.setdefault("decision_gate_events", []).append(
                        {
                            "step": state.step,
                            "selected": selected.name,
                            "kind": selected.kind.value,
                        }
                    )

            if selected.kind == ActionKind.STOP:
                answer = selected.arguments.get("answer")
                if answer is not None:
                    state.scratch["answer"] = answer
                state.done = True
                state.stop_reason = selected.rationale or "reasoner_stopped"
                break

            if selected.kind == ActionKind.WAIT:
                requested = float(selected.arguments.get("seconds", 0) or 0)
                if requested <= 0.0:
                    requested = state.next_effect_ready_in() or 0.0
                waited = float(self.time_driver.advance(requested))
                self._sync_state_time(state)
                self._record_expired_effects(state)
                result = {
                    "waited": waited,
                    "virtual_time_seconds": state.virtual_time_seconds,
                }
                tool_spec = None
            else:
                tool_spec = self.tools.get(selected.name)
                result = self.tools.execute(selected.name, selected.arguments)

            self._schedule_temporal_effect(tool_spec, selected, state)
            temporal_evaluations = self._evaluate_temporal_observation(selected, result, state)

            observation_metadata = {}
            if temporal_evaluations:
                observation_metadata["temporal_effects"] = temporal_evaluations
            observation = Observation(
                action_name=selected.name,
                result=result,
                metadata=observation_metadata,
            )
            state.observations.append(observation)

            experiment_update = None
            mismatch_assessment = None
            discovered_hypotheses: list[str] = []
            if tool_spec is not None and tool_spec.experiment_contract is not None:
                experiment_contract = expanded_experiment_contract(
                    tool_spec.experiment_contract,
                    self.world_model,
                )
                if self.capabilities.open_world:
                    mismatch_assessment = assess_model_mismatch(
                        experiment_contract,
                        self.world_model,
                        result,
                        policy=self.mismatch_policy,
                        metadata={"step": state.step, "action": selected.name},
                    )
                    if mismatch_assessment is not None:
                        observation.metadata["model_mismatch"] = {
                            "predictive_probability": mismatch_assessment.predictive_probability,
                            "surprisal": mismatch_assessment.surprisal,
                            "suspicious": mismatch_assessment.suspicious,
                            "hard_mismatch": mismatch_assessment.hard_mismatch,
                            "escalate": mismatch_assessment.escalate,
                            "mismatch_id": mismatch_assessment.mismatch_id,
                        }

                posterior_suppressed = bool(
                    mismatch_assessment is not None
                    and mismatch_assessment.suppress_closed_world_posterior
                )
                if self.capabilities.bayesian_updates and not posterior_suppressed:
                    experiment_update = apply_experiment_observation(
                        experiment_contract,
                        self.world_model,
                        result,
                        source=selected.name,
                        metadata={"step": state.step},
                    )
                    if experiment_update is not None and self.capabilities.open_world:
                        maybe_resolve_model_mismatch(self.world_model)
                elif posterior_suppressed:
                    discovered_hypotheses = self._discover_after_mismatch(
                        state,
                        mismatch_assessment,
                    )

            runtime_information_gain = (
                selected_score.information_gain
                if selected_score is not None
                else (0.0 if not self.capabilities.eig else selected.expected_information_gain)
            )
            information_source = (
                selected_score.information_source
                if selected_score is not None
                else (
                    "ablation_eig_disabled"
                    if not self.capabilities.eig
                    else "model_estimate"
                )
            )
            expected_effects = {
                "goal_gain": selected_score.goal_gain if selected_score is not None else selected.expected_goal_gain,
                "information_gain": runtime_information_gain,
                "information_source": information_source,
                "model_information_gain": selected.expected_information_gain,
                "tests_hypotheses": list(selected.tests_hypotheses),
                "falsification_target": selected.falsification_target,
                "virtual_time_seconds": state.virtual_time_seconds,
                "runtime_capabilities": self.capabilities.to_dict(),
            }
            if experiment_update is not None:
                expected_effects["experiment_id"] = experiment_update.experiment_id
                expected_effects["observed_outcome"] = experiment_update.outcome
                expected_effects["posterior"] = experiment_update.posterior
                expected_effects["predictive_probability"] = experiment_update.predictive_probability
                expected_effects["surprisal"] = experiment_update.surprisal
            if mismatch_assessment is not None:
                expected_effects["model_mismatch"] = {
                    "predictive_probability": mismatch_assessment.predictive_probability,
                    "surprisal": mismatch_assessment.surprisal,
                    "suspicious": mismatch_assessment.suspicious,
                    "hard_mismatch": mismatch_assessment.hard_mismatch,
                    "escalate": mismatch_assessment.escalate,
                    "posterior_suppressed": mismatch_assessment.suppress_closed_world_posterior,
                    "discovered_hypotheses": discovered_hypotheses,
                }
            if temporal_evaluations:
                expected_effects["temporal_effects"] = temporal_evaluations

            self.world_model.record_transition(
                Transition(
                    action=selected.name,
                    arguments=selected.arguments,
                    observation=result,
                    expected_effects=expected_effects,
                )
            )

            if self.capabilities.causal_updates and self.belief_updater:
                self.belief_updater(state, self.world_model, decision, observation)
            mismatch_escalated = bool(mismatch_assessment and mismatch_assessment.escalate)
            if (
                self.capabilities.causal_updates
                and self.hypothesis_updater
                and experiment_update is None
                and not temporal_evaluations
                and not mismatch_escalated
            ):
                self.hypothesis_updater(state, self.world_model, decision, observation)

            state.step += 1

        if not state.done:
            state.done = True
            state.stop_reason = "budget_exhausted"
        return state
