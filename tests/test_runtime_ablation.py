import pytest

from causalrag.agent import ActionKind, CandidateAction, TemporalEffectContract
from causalrag.agent.features import RuntimeFeatureFlags
from causalrag.experiments import ExperimentContract, InterventionContract, OutcomeLikelihood
from causalrag.observability import InMemoryEventSink, create_observable_agent
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


def test_eig_switch_changes_actual_policy_information_source():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(name="diagnose", description="diagnose", handler=lambda: {"outcome": "a"}, experiment_contract=_experiment())
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
        expected_information_gain=0.01,
    )

    full = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(eig=True))
    no_eig = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(eig=False))

    assert full.information_source == "runtime_bayesian_eig"
    assert full.bayesian_information_gain is not None
    assert no_eig.information_source == "runtime_hypothesis_discrimination_eig_disabled"
    assert no_eig.bayesian_information_gain is None


def test_causal_runtime_off_uses_model_estimates_and_skips_runtime_bayes():
    world = _world()
    tools = ToolRegistry([
        ToolSpec(name="diagnose", description="diagnose", handler=lambda: {"outcome": "a"}, experiment_contract=_experiment())
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="diagnose",
        tests_hypotheses=["H1", "H2"],
        expected_information_gain=0.23,
        expected_goal_gain=0.12,
    )

    score = score_action(
        action,
        world_model=world,
        tools=tools,
        features=RuntimeFeatureFlags(causal_runtime=False),
    )
    assert score.information_source == "model_estimate_causal_runtime_disabled"
    assert score.information_gain == pytest.approx(0.23)
    assert score.bayesian_information_gain is None
    assert score.decision_value is None


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

    with_evsi = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(evsi=True))
    without_evsi = score_action(action, world_model=world, tools=tools, features=RuntimeFeatureFlags(evsi=False))

    assert with_evsi.information_source == "runtime_bayesian_eig"
    assert with_evsi.expected_value_of_sample_information is not None
    assert with_evsi.decision_value_source == "runtime_expected_decision_value_after_sampling"
    assert without_evsi.information_source == "runtime_bayesian_eig"
    assert without_evsi.expected_value_of_sample_information is None
    assert without_evsi.decision_value is None


class TemporalAblationReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(kind=ActionKind.INTERVENE, name="change_state")]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

    def uncertainty(self, state, world_model):
        return "delayed effect"

    def hypothesis_proposals(self, state, world_model):
        return []


def _temporal_agent(features):
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "intervention causes delayed cooling", probability=0.8)
    sink = InMemoryEventSink()
    agent = create_observable_agent(
        event_sink=sink,
        features=features,
        reasoner=TemporalAblationReasoner(),
        world_model=world,
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
    return agent, sink


def test_temporal_attribution_switch_changes_actual_execution_path():
    guarded, guarded_sink = _temporal_agent(RuntimeFeatureFlags(temporal_attribution=True))
    guarded_result = guarded.run("change and verify", max_steps=5)
    guarded_names = [event["name"] for event in guarded_sink.snapshot()]

    naive, naive_sink = _temporal_agent(RuntimeFeatureFlags(temporal_attribution=False))
    naive_result = naive.run("change and verify", max_steps=5)
    naive_names = [event["name"] for event in naive_sink.snapshot()]

    assert "wait.started" in guarded_names
    assert "attribution" in guarded_names
    assert guarded_result.state.step > naive_result.state.step
    assert "wait.started" not in naive_names
    assert "attribution" not in naive_names


class OpenWorldAblationReasoner:
    def __init__(self):
        self.discovery_calls = 0

    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(kind=ActionKind.OBSERVE, name="surprise", tests_hypotheses=["H1", "H2"])]
        return [CandidateAction(kind=ActionKind.STOP, name="stop", arguments={"answer": "done"})]

    def uncertainty(self, state, world_model):
        return "whether the model class is incomplete"

    def hypothesis_proposals(self, state, world_model):
        return []

    def discover_hypotheses(self, state, world_model, mismatch_context):
        self.discovery_calls += 1
        return [HypothesisProposal(hypothesis_id="H3", statement="novel hidden mechanism", probability=0.99)]


def _open_world_agent(enabled):
    world = _world()
    reasoner = OpenWorldAblationReasoner()
    sink = InMemoryEventSink()
    contract = ExperimentContract(
        experiment_id="hard_surprise",
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 0.999, "H2": 0.999}),
            OutcomeLikelihood("novel", {"H1": 0.001, "H2": 0.001}),
        ],
    )
    agent = create_observable_agent(
        event_sink=sink,
        features=RuntimeFeatureFlags(open_world_discovery=enabled),
        reasoner=reasoner,
        world_model=world,
        tools=[
            ToolSpec(
                name="surprise",
                description="emit a hard model mismatch",
                handler=lambda: {"outcome": "novel"},
                experiment_contract=contract,
                metadata={"kind": "observe"},
            )
        ],
    )
    return agent, sink, reasoner


def test_open_world_switch_controls_mismatch_and_discovery_path():
    enabled, enabled_sink, enabled_reasoner = _open_world_agent(True)
    enabled.run("detect model incompleteness", max_steps=3)
    enabled_names = [event["name"] for event in enabled_sink.snapshot()]

    disabled, disabled_sink, disabled_reasoner = _open_world_agent(False)
    disabled.run("closed world only", max_steps=3)
    disabled_names = [event["name"] for event in disabled_sink.snapshot()]

    assert enabled_reasoner.discovery_calls == 1
    assert "model_mismatch" in enabled_names
    assert "hypothesis.discovered" in enabled_names
    assert disabled_reasoner.discovery_calls == 0
    assert "model_mismatch" not in disabled_names
    assert "hypothesis.discovered" not in disabled_names
