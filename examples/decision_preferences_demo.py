"""Show deployment-owned utilities changing observe-vs-intervene behavior.

Run:
    python examples/decision_preferences_demo.py
"""

from branchpoint import (
    ActionKind,
    CandidateAction,
    DecisionPreferences,
    ToolSpec,
    create_agent,
)
from branchpoint.experiments import (
    ExperimentContract,
    InterventionContract,
    OutcomeLikelihood,
)
from branchpoint.world_model import CausalWorldModel


class DemoReasoner:
    def propose(self, state, world_model):
        if state.observations:
            latest = state.observations[-1].result
            if isinstance(latest, dict) and "fixed" in latest:
                return [
                    CandidateAction(
                        kind=ActionKind.STOP,
                        name="stop",
                        arguments={"answer": "DecisionPreferences demo complete."},
                    )
                ]
        return [
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="diagnose",
                tests_hypotheses=["H1", "H2"],
            ),
            CandidateAction(kind=ActionKind.INTERVENE, name="fix_h1"),
            CandidateAction(kind=ActionKind.INTERVENE, name="fix_h2"),
        ]

    def uncertainty(self, state, world_model):
        return "H1 vs H2"


def build_world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "mechanism one", probability=0.70)
    world.upsert_hypothesis("H2", "mechanism two", probability=0.30)
    return world


def build_tools():
    diagnostic = ExperimentContract(
        experiment_id="diagnostic",
        outcomes=[
            OutcomeLikelihood("leans_h1", {"H1": 0.75, "H2": 0.25}),
            OutcomeLikelihood("leans_h2", {"H1": 0.25, "H2": 0.75}),
        ],
    )
    return [
        ToolSpec(
            name="diagnose",
            description="Run a diagnostic observation.",
            handler=lambda: {"outcome": "leans_h1"},
            cost=0.08,
            experiment_contract=diagnostic,
        ),
        ToolSpec(
            name="fix_h1",
            description="Apply intervention for H1.",
            handler=lambda: {"fixed": "H1"},
            cost=0.20,
            intervention_contract=InterventionContract(
                "fix_h1", {"H1": 1.0, "H2": 0.0}
            ),
        ),
        ToolSpec(
            name="fix_h2",
            description="Apply intervention for H2.",
            handler=lambda: {"fixed": "H2"},
            cost=0.20,
            intervention_contract=InterventionContract(
                "fix_h2", {"H1": 0.0, "H2": 1.0}
            ),
        ),
    ]


def main():
    preferences = DecisionPreferences(
        intervention_utilities={
            "fix_h1": {"H1": 1.0, "H2": -1.0},
            "fix_h2": {"H1": -1.0, "H2": 1.0},
        }
    )
    agent = create_agent(
        world_model=build_world(),
        tools=build_tools(),
        reasoner=DemoReasoner(),
        decision_preferences=preferences,
    )
    result = agent.run("Resolve the hidden mechanism with minimum expected loss.", max_steps=5)

    print(result.answer)
    print("selected actions:")
    for decision in result.state.decisions:
        score = next(
            value
            for value in decision.action_scores
            if value.action_name == decision.selected.name
        )
        print(
            f"  step={decision.step} action={decision.selected.name} "
            f"source={score.decision_value_source or score.information_source} "
            f"utility={score.total_utility:.3f}"
        )


if __name__ == "__main__":
    main()
