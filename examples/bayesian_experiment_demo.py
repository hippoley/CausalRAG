"""No-key demo: choose a modeled experiment by Bayesian EIG and update posterior.

Run:
    python examples/bayesian_experiment_demo.py
"""

from branchpoint import (
    ActionKind,
    CandidateAction,
    ExperimentContract,
    OutcomeLikelihood,
    ToolSpec,
    create_agent,
)
from branchpoint.world_model import CausalWorldModel


class DemoReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_room_temperature",
                    expected_information_gain=0.95,
                    rationale="Model thinks temperature might be useful.",
                ),
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="measure_filter_pressure",
                    expected_information_gain=0.02,
                    tests_hypotheses=["H1", "H2"],
                    rationale="Runtime owns a diagnostic outcome model for this test.",
                ),
            ]
        h1 = world_model.get_hypothesis("H1")
        h2 = world_model.get_hypothesis("H2")
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={
                    "answer": f"Posterior after pressure test: H1={h1.probability:.2f}, H2={h2.probability:.2f}."
                },
            )
        ]

    def uncertainty(self, state, world_model):
        return "filter restriction versus weak fan"


world = CausalWorldModel()
world.upsert_hypothesis("H1", "The HVAC filter is clogged", probability=0.5)
world.upsert_hypothesis("H2", "The supply fan is weak", probability=0.5)

pressure_contract = ExperimentContract(
    experiment_id="filter_pressure_test",
    description="Pressure drop discriminates filter restriction from fan weakness.",
    outcomes=[
        OutcomeLikelihood("high", {"H1": 0.9, "H2": 0.1}),
        OutcomeLikelihood("normal", {"H1": 0.1, "H2": 0.9}),
    ],
)

agent = create_agent(
    world_model=world,
    reasoner=DemoReasoner(),
    tools=[
        ToolSpec(
            name="read_room_temperature",
            description="Read ambient room temperature.",
            handler=lambda: {"temperature_c": 25.0},
            cost=0.01,
            metadata={"kind": "observe"},
        ),
        ToolSpec(
            name="measure_filter_pressure",
            description="Measure pressure drop across the HVAC filter.",
            handler=lambda: {"outcome": "normal", "pressure_pa": 12},
            cost=0.01,
            metadata={"kind": "observe"},
            experiment_contract=pressure_contract,
        ),
    ],
)

result = agent.run("Diagnose why airflow is low", max_steps=3)
payload = result.to_dict()
first = payload["decisions"][0]
selected = first["selected"]
selected_score = next(score for score in first["action_scores"] if score["action_name"] == selected["name"])

print(result.answer)
print("\nSelected first action:", selected["name"])
print("information_source:", selected_score["information_source"])
print("model_information_gain:", selected_score["model_information_gain"])
print("bayesian_information_gain:", round(selected_score["bayesian_information_gain"], 3))
print("posterior:", payload["transitions"][0]["expected_effects"]["posterior"])
