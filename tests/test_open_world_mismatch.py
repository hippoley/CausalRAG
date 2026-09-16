import pytest

from causalrag.agent import ActionKind, CandidateAction, CausalAgentLoop
from causalrag.experiments import (
    ExperimentContract,
    ModelMismatchPolicy,
    OutcomeLikelihood,
    apply_experiment_observation,
    assess_model_mismatch,
    expanded_experiment_contract,
    maybe_resolve_model_mismatch,
)
from causalrag.reasoning.hypothesis import HypothesisProposal
from causalrag.reasoning.policy import score_action
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel


def _known_world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Known mechanism one", probability=0.5)
    world.upsert_hypothesis("H2", "Known mechanism two", probability=0.5)
    return world


def _sensor_a_contract():
    return ExperimentContract(
        experiment_id="sensor_a_test",
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 0.96, "H2": 0.94}),
            OutcomeLikelihood("novel_signature", {"H1": 0.04, "H2": 0.06}),
        ],
    )


def _sensor_b_contract():
    return ExperimentContract(
        experiment_id="sensor_b_test",
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 0.95, "H2": 0.96}),
            OutcomeLikelihood("novel_signature", {"H1": 0.05, "H2": 0.04}),
        ],
    )


def test_soft_model_mismatch_requires_distinct_experiments():
    world = _known_world()
    policy = ModelMismatchPolicy(
        soft_predictive_threshold=0.06,
        hard_predictive_threshold=0.005,
        min_distinct_experiments=2,
    )

    first = assess_model_mismatch(
        _sensor_a_contract(),
        world,
        {"outcome": "novel_signature"},
        policy=policy,
    )
    assert first is not None
    assert first.suspicious is True
    assert first.escalate is False
    assert world.model_mismatch_active is True

    second_same = assess_model_mismatch(
        _sensor_a_contract(),
        world,
        {"outcome": "novel_signature"},
        policy=policy,
    )
    assert second_same.escalate is False

    second_sensor = assess_model_mismatch(
        _sensor_b_contract(),
        world,
        {"outcome": "novel_signature"},
        policy=policy,
    )
    assert second_sensor.escalate is True
    assert second_sensor.suppress_closed_world_posterior is True
    assert len(world.unresolved_model_mismatches()) == 3


def test_hard_model_mismatch_can_escalate_from_one_near_impossible_outcome():
    world = _known_world()
    contract = ExperimentContract(
        experiment_id="hard_surprise",
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 0.999, "H2": 0.999}),
            OutcomeLikelihood("impossible_like", {"H1": 0.001, "H2": 0.001}),
        ],
    )
    assessment = assess_model_mismatch(
        contract,
        world,
        {"outcome": "impossible_like"},
        policy=ModelMismatchPolicy(),
    )
    assert assessment is not None
    assert assessment.hard_mismatch is True
    assert assessment.escalate is True


def test_discovered_hypothesis_prediction_expands_contract_and_requires_evidence_to_validate():
    world = _known_world()
    discovered = world.add_discovered_hypothesis(
        "H4",
        "Previously unmodeled sensor drift",
        experiment_predictions={
            "sensor_a_test": {
                "ordinary": 0.05,
                "novel_signature": 0.95,
            }
        },
        initial_probability=0.2,
    )
    assert discovered is not None
    assert discovered.origin == "discovered"
    assert discovered.validated is False
    assert discovered.probability == pytest.approx(0.2)

    expanded = expanded_experiment_contract(_sensor_a_contract(), world)
    assert set(expanded.hypothesis_ids()) == {"H1", "H2", "H4"}
    assert expanded.likelihood("novel_signature", "H4") == pytest.approx(0.95)

    update = apply_experiment_observation(
        expanded,
        world,
        {"outcome": "novel_signature"},
        source="sensor_a",
    )
    assert update is not None
    assert update.posterior["H4"] > 0.5
    assert world.get_hypothesis("H4").validated is True

    world.model_mismatch_active = True
    assert maybe_resolve_model_mismatch(world) is True
    assert world.model_mismatch_active is False


def test_policy_refuses_closed_world_bayes_when_declared_test_includes_unmodeled_hypothesis():
    world = _known_world()
    world.add_discovered_hypothesis("H4", "Novel mechanism without predictions")
    tools = ToolRegistry([
        ToolSpec(
            name="sensor_a",
            description="sensor a",
            handler=lambda: {"outcome": "ordinary"},
            experiment_contract=_sensor_a_contract(),
        )
    ])
    action = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="sensor_a",
        tests_hypotheses=["H1", "H4"],
        expected_information_gain=0.99,
    )
    score = score_action(action, world_model=world, tools=tools)
    assert score.information_source == "runtime_hypothesis_discrimination"
    assert score.bayesian_information_gain is None


class OpenWorldReasoner:
    def __init__(self):
        self.discovery_calls = 0

    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(
                kind=ActionKind.OBSERVE,
                name="sensor_a",
                tests_hypotheses=["H1", "H2"],
                rationale="First independent residual check.",
            )]
        if state.step == 1:
            return [CandidateAction(
                kind=ActionKind.OBSERVE,
                name="sensor_b",
                tests_hypotheses=["H1", "H2"],
                rationale="Second independent residual check.",
            )]
        if state.step == 2:
            return [CandidateAction(
                kind=ActionKind.OBSERVE,
                name="sensor_a",
                tests_hypotheses=["H1", "H2", "H4"],
                falsification_target="H4",
                rationale="Re-test a predicted signature against the discovered mechanism.",
            )]
        return [CandidateAction(
            kind=ActionKind.STOP,
            name="stop",
            arguments={"answer": "validated open-world hypothesis"},
        )]

    def hypothesis_proposals(self, state, world_model):
        return []

    def discover_hypotheses(self, state, world_model, mismatch_context):
        self.discovery_calls += 1
        return [HypothesisProposal(
            hypothesis_id="H4",
            statement="A previously unmodeled sensor drift mechanism causes the residual signature.",
            probability=0.99,
            rationale="Two independent measurements are unlikely under H1/H2.",
            falsifiers=["sensor A returns ordinary on repeat"],
            experiment_predictions={
                "sensor_a_test": {
                    "ordinary": 0.05,
                    "novel_signature": 0.95,
                },
                "sensor_b_test": {
                    "ordinary": 0.10,
                    "novel_signature": 0.90,
                },
            },
        )]

    def uncertainty(self, state, world_model):
        return "whether the known model class is incomplete"


def test_agent_loop_detects_none_of_the_above_discovers_and_validates_new_hypothesis():
    world = _known_world()
    reasoner = OpenWorldReasoner()
    tools = ToolRegistry([
        ToolSpec(
            name="sensor_a",
            description="independent diagnostic sensor A",
            handler=lambda: {"outcome": "novel_signature"},
            experiment_contract=_sensor_a_contract(),
            metadata={"kind": "observe"},
        ),
        ToolSpec(
            name="sensor_b",
            description="independent diagnostic sensor B",
            handler=lambda: {"outcome": "novel_signature"},
            experiment_contract=_sensor_b_contract(),
            metadata={"kind": "observe"},
        ),
    ])
    loop = CausalAgentLoop(
        reasoner=reasoner,
        tools=tools,
        world_model=world,
        mismatch_policy=ModelMismatchPolicy(
            soft_predictive_threshold=0.06,
            hard_predictive_threshold=0.005,
            min_distinct_experiments=2,
            discovered_initial_probability=0.2,
        ),
    )

    state = loop.run("diagnose a failure outside the known fault model", max_steps=5)

    assert state.done
    assert reasoner.discovery_calls == 1
    assert state.scratch["answer"] == "validated open-world hypothesis"
    assert state.scratch["hypothesis_discovery_events"][0]["hypotheses"] == ["H4"]

    discovered = world.get_hypothesis("H4")
    assert discovered is not None
    assert discovered.origin == "discovered"
    assert discovered.validated is True
    assert discovered.probability > 0.5
    assert world.model_mismatch_active is False

    # The second low-likelihood outcome triggers none-of-the-above rather than
    # forcing another closed-world posterior onto H1/H2.
    second_transition = world.transitions[1]
    assert second_transition.expected_effects["model_mismatch"]["escalate"] is True
    assert second_transition.expected_effects["model_mismatch"]["posterior_suppressed"] is True
    assert second_transition.expected_effects["model_mismatch"]["discovered_hypotheses"] == ["H4"]
    assert "posterior" not in second_transition.expected_effects

    # The next experiment uses H4's falsifiable prediction and validates it by
    # evidence rather than accepting the proposer's claimed 0.99 confidence.
    third_transition = world.transitions[2]
    assert third_transition.expected_effects["information_source"] == "runtime_bayesian_eig"
    assert third_transition.expected_effects["posterior"]["H4"] > 0.5
