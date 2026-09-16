"""No-key demonstration of model mismatch -> hypothesis discovery -> validation."""

from causalrag.agent import ActionKind, CandidateAction, CausalAgentLoop
from causalrag.experiments import ExperimentContract, ModelMismatchPolicy, OutcomeLikelihood
from causalrag.reasoning.hypothesis import HypothesisProposal
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel


def contract(experiment_id, h1_rare, h2_rare):
    return ExperimentContract(
        experiment_id=experiment_id,
        outcomes=[
            OutcomeLikelihood("ordinary", {"H1": 1.0 - h1_rare, "H2": 1.0 - h2_rare}),
            OutcomeLikelihood("novel_signature", {"H1": h1_rare, "H2": h2_rare}),
        ],
    )


class Reasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [CandidateAction(ActionKind.OBSERVE, "sensor_a", tests_hypotheses=["H1", "H2"])]
        if state.step == 1:
            return [CandidateAction(ActionKind.OBSERVE, "sensor_b", tests_hypotheses=["H1", "H2"])]
        if state.step == 2:
            return [CandidateAction(ActionKind.OBSERVE, "sensor_a", tests_hypotheses=["H1", "H2", "H4"])]
        return [CandidateAction(ActionKind.STOP, "stop", arguments={"answer": "H4 validated from residual evidence"})]

    def hypothesis_proposals(self, state, world_model):
        return []

    def discover_hypotheses(self, state, world_model, mismatch_context):
        return [HypothesisProposal(
            hypothesis_id="H4",
            statement="A previously unmodeled sensor drift mechanism explains the residual signature.",
            probability=0.99,  # deliberately ignored by runtime
            falsifiers=["sensor A returns ordinary on a repeat measurement"],
            experiment_predictions={
                "sensor_a_test": {"ordinary": 0.05, "novel_signature": 0.95},
                "sensor_b_test": {"ordinary": 0.10, "novel_signature": 0.90},
            },
        )]

    def uncertainty(self, state, world_model):
        return "whether the modeled fault class is incomplete"


world = CausalWorldModel()
world.upsert_hypothesis("H1", "Known fault one", probability=0.5)
world.upsert_hypothesis("H2", "Known fault two", probability=0.5)

tools = ToolRegistry([
    ToolSpec(
        "sensor_a",
        "Independent residual sensor A",
        handler=lambda: {"outcome": "novel_signature"},
        experiment_contract=contract("sensor_a_test", 0.04, 0.06),
        metadata={"kind": "observe"},
    ),
    ToolSpec(
        "sensor_b",
        "Independent residual sensor B",
        handler=lambda: {"outcome": "novel_signature"},
        experiment_contract=contract("sensor_b_test", 0.05, 0.04),
        metadata={"kind": "observe"},
    ),
])

loop = CausalAgentLoop(
    reasoner=Reasoner(),
    tools=tools,
    world_model=world,
    mismatch_policy=ModelMismatchPolicy(soft_predictive_threshold=0.06),
)
state = loop.run("diagnose a fault outside the current model class", max_steps=5)

print(state.scratch["answer"])
print("\nOpen-world trace:")
for index, transition in enumerate(world.transitions):
    mismatch = transition.expected_effects.get("model_mismatch")
    posterior = transition.expected_effects.get("posterior")
    print(f"  step={index} action={transition.action}")
    if mismatch:
        print(
            "    mismatch: "
            f"p={mismatch['predictive_probability']:.4f} "
            f"escalate={mismatch['escalate']} "
            f"posterior_suppressed={mismatch['posterior_suppressed']} "
            f"discovered={mismatch['discovered_hypotheses']}"
        )
    if posterior:
        print(f"    posterior={posterior}")

h4 = world.get_hypothesis("H4")
print("\nDiscovered hypothesis:")
print(f"  id={h4.hypothesis_id} origin={h4.origin} validated={h4.validated} p={h4.probability:.3f}")
print(f"  open_world={world.snapshot()['open_world']}")
