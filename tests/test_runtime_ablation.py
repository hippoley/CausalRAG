import pytest

from causalrag.agent import (
    ActionKind,
    CandidateAction,
    RuntimeCapabilities,
    TemporalEffectContract,
    create_agent,
)
from causalrag.benchmarks import (
    AblationArm,
    AblationEpisode,
    ScenarioManifest,
    run_ablation_matrix,
)
from causalrag.experiments import ExperimentContract, InterventionContract, OutcomeLikelihood
from causalrag.reasoning.hypothesis import HypothesisProposal
from causalrag.reasoning.policy import score_action
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "cause one", probability=0.5)
    world.upsert_hypothesis("H2", "cause two", probability=0.5)
    return world


def _experiment():
    return ExperimentContract(
        experiment_id="diagnostic",
        outcomes=[
            OutcomeLikelihood("a", {"H1": 0.9, "H2": 0.1}),
            OutcomeLikelihood("b", {"H1": 0.1, "H2": 0.9}),
        ],
    )


def test_eig_switch_changes_actual_policy_score_not_only_metadata():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(
            name="diagnose",
            description="diagnose",
            handler=lambda: {"outcome": "a"},
            experiment_contract=_experiment(),
        )
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
        expected_information_gain=0.99,
    )

    full = score_action(
        action,
        world_model=world,
        tools=tools,
        capabilities=RuntimeCapabilities(eig=True),
    )
    no_eig = score_action(
        action,
        world_model=world,
        tools=tools,
        capabilities=RuntimeCapabilities(eig=False),
    )

    assert full.information_source == "runtime_bayesian_eig"
    assert full.bayesian_information_gain is not None
    assert no_eig.information_source == "ablation_eig_disabled"
    assert no_eig.information_gain == 0.0


def test_evsi_switch_removes_sampling_decision_value_but_keeps_eig():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(
            name="diagnose",
            description="diagnose",
            handler=lambda: {"outcome": "a"},
            experiment_contract=_experiment(),
            cost=0.02,
        ),
        ToolSpec(
            name="fix_one",
            description="fix one",
            handler=lambda: {"ok": True},
            intervention_contract=InterventionContract(
                intervention_id="fix_one",
                utilities={"H1": 1.0, "H2": -1.0},
            ),
        ),
        ToolSpec(
            name="fix_two",
            description="fix two",
            handler=lambda: {"ok": True},
            intervention_contract=InterventionContract(
                intervention_id="fix_two",
                utilities={"H1": -1.0, "H2": 1.0},
            ),
        ),
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
    )

    with_evsi = score_action(
        action,
        world_model=world,
        tools=tools,
        capabilities=RuntimeCapabilities(evsi=True),
    )
    without_evsi = score_action(
        action,
        world_model=world,
        tools=tools,
        capabilities=RuntimeCapabilities(evsi=False),
    )

    assert with_evsi.information_source == "runtime_bayesian_eig"
    assert with_evsi.expected_value_of_sample_information is not None
    assert with_evsi.decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert without_evsi.information_source == "runtime_bayesian_eig"
    assert without_evsi.expected_value_of_sample_information is None
    assert without_evsi.decision_value is None


class TemporalReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(kind=ActionKind.INTERVENE, name="change_state")]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

    def uncertainty(self, state, world_model):
        return "delayed effect"

    def hypothesis_proposals(self, state, world_model):
        return []


def _temporal_agent(enabled):
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "change causes delayed cooling", probability=0.8)
    return create_agent(
        reasoner=TemporalReasoner(),
        world_model=world,
        capabilities=RuntimeCapabilities(temporal_attribution=enabled),
        tools=[
            ToolSpec(
                name="change_state",
                description="change state",
                handler=lambda: {"ok": True},
                metadata={"kind": "intervene"},
                temporal_effect_contract=TemporalEffectContract(
                    effect_id="cooling",
                    observe_with="read_state",
                    observation_key="state",
                    earliest_seconds=10,
                    latest_seconds=20,
                    expected_outcomes={"H1": "cool"},
                ),
            ),
            ToolSpec(
                name="read_state",
                description="read delayed state",
                handler=lambda: {"state": "cool"},
                metadata={"kind": "observe"},
            ),
        ],
    )


def test_temporal_switch_changes_execution_path():
    guarded = _temporal_agent(True).run("change and verify", max_steps=5)
    naive = _temporal_agent(False).run("change and verify", max_steps=5)

    guarded_actions = [decision.selected.name for decision in guarded.state.decisions]
    naive_actions = [decision.selected.name for decision in naive.state.decisions]

    assert "wait_for_effect_window" in guarded_actions
    assert "read_state" in guarded_actions
    assert guarded.state.step > naive.state.step
    assert "wait_for_effect_window" not in naive_actions
    assert "read_state" not in naive_actions


class OpenWorldReasoner:
    def __init__(self):
        self.discovery_calls = 0

    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(
                kind=ActionKind.OBSERVE,
                name="surprise",
                tests_hypotheses=["H1", "H2"],
            )]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

    def uncertainty(self, state, world_model):
        return "whether model class is incomplete"

    def hypothesis_proposals(self, state, world_model):
        return []

    def discover_hypotheses(self, state, world_model, mismatch_context):
        self.discovery_calls += 1
        return [HypothesisProposal(
            hypothesis_id="H3",
            statement="novel hidden mechanism",
            probability=0.99,
        )]


def _open_world_agent(enabled):
    world = _world()
    reasoner = OpenWorldReasoner()
    contract = ExperimentContract(
        experiment_id="hard_surprise",
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 0.999, "H2": 0.999}),
            OutcomeLikelihood("novel", {"H1": 0.001, "H2": 0.001}),
        ],
    )
    agent = create_agent(
        reasoner=reasoner,
        world_model=world,
        capabilities=RuntimeCapabilities(open_world=enabled),
        tools=[
            ToolSpec(
                name="surprise",
                description="hard model mismatch",
                handler=lambda: {"outcome": "novel"},
                experiment_contract=contract,
                metadata={"kind": "observe"},
            )
        ],
    )
    return agent, reasoner


def test_open_world_switch_controls_mismatch_discovery_path():
    enabled, enabled_reasoner = _open_world_agent(True)
    enabled_result = enabled.run("detect incompleteness", max_steps=3)

    disabled, disabled_reasoner = _open_world_agent(False)
    disabled_result = disabled.run("closed world only", max_steps=3)

    assert enabled_reasoner.discovery_calls == 1
    assert enabled_result.world_model.get_hypothesis("H3") is not None
    assert disabled_reasoner.discovery_calls == 0
    assert disabled_result.world_model.get_hypothesis("H3") is None


def test_vanilla_tool_loop_capabilities_are_exposed_in_run_result():
    class OneStep:
        def propose(self, state, world_model):
            return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

        def uncertainty(self, state, world_model):
            return None

    capabilities = RuntimeCapabilities.vanilla_tool_loop()
    result = create_agent(reasoner=OneStep(), capabilities=capabilities).run("finish")
    assert result.to_dict()["runtime_capabilities"] == capabilities.to_dict()
    assert result.to_dict()["runtime_capabilities"]["causal_selection"] is False


def test_ablation_matrix_pairs_same_frozen_scenario_across_model_lanes():
    scenarios = [
        ScenarioManifest(
            scenario_id="boptest:bestest_air:summer",
            environment="boptest",
            seed=7,
            horizon_steps=24,
            metadata={"electricity_price": "dynamic"},
        )
    ]
    arms = [
        AblationArm(
            arm_id="small+causal",
            proposer_family="small",
            provider="local",
            model="qwen3:8b",
            capabilities=RuntimeCapabilities.full(),
        ),
        AblationArm(
            arm_id="frontier+vanilla",
            proposer_family="frontier",
            provider="openai",
            model="frontier-model",
            capabilities=RuntimeCapabilities.vanilla_tool_loop(),
        ),
    ]

    def run_episode(arm, scenario):
        return AblationEpisode(
            arm_id=arm.arm_id,
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            metrics={"cost_tot": 1.0 if arm.arm_id == "small+causal" else 1.3},
        )

    report = run_ablation_matrix(arms=arms, scenarios=scenarios, run_episode=run_episode)
    assert len(report.episodes) == 2
    assert report.paired_metric_deltas("frontier+vanilla", "small+causal", "cost_tot") == pytest.approx([-0.3])
