"""No-key demo: competing hypotheses -> diagnostic action -> falsification.

Run:
    python examples/hypothesis_falsification_demo.py
"""

from causalrag import (
    ActionKind,
    CandidateAction,
    Evidence,
    HypothesisProposal,
    ToolSpec,
    create_agent,
)


class HVACReasoner:
    """Small deterministic reasoner so the demo needs no model/API key."""

    def propose(self, state, world_model):
        if not state.observations:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_filter_pressure",
                    expected_information_gain=0.9,
                    tests_hypotheses=["H1", "H2"],
                    falsification_target="H1",
                    rationale=(
                        "Filter pressure is diagnostic: a normal reading counts "
                        "against the clogged-filter hypothesis."
                    ),
                )
            ]

        h1 = world_model.get_hypothesis("H1")
        h2 = world_model.get_hypothesis("H2")
        answer = (
            f"After the diagnostic observation: H1={h1.probability:.2f}, "
            f"H2={h2.probability:.2f}. Investigate the fan next."
        )
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": answer},
                rationale="The first diagnostic observation changed the hypothesis ranking.",
            )
        ]

    def uncertainty(self, state, world_model):
        return "Is low airflow caused by a clogged filter or an underperforming fan?"

    def hypothesis_proposals(self, state, world_model):
        return [
            HypothesisProposal(
                hypothesis_id="H1",
                statement="The HVAC filter is clogged.",
                probability=0.60,
                falsifiers=["Filter pressure drop is within the normal range."],
            ),
            HypothesisProposal(
                hypothesis_id="H2",
                statement="The supply fan is underperforming.",
                probability=0.40,
                falsifiers=["Fan RPM and delivered airflow are normal under load."],
            ),
        ]


def update_hypotheses(state, world_model, decision, observation):
    """Deterministic domain evaluator for this demo's sensor result."""
    pressure = float(observation.result["pressure_pa"])
    if pressure < 20:
        world_model.update_hypothesis(
            "H1",
            Evidence(
                source=observation.action_name,
                statement="Filter pressure drop is normal.",
                weight=-0.85,
                kind="falsifier",
            ),
        )
        world_model.update_hypothesis(
            "H2",
            Evidence(
                source=observation.action_name,
                statement="Normal filter pressure leaves a fan fault more plausible.",
                weight=0.35,
                kind="support",
            ),
        )


def main():
    agent = create_agent(
        reasoner=HVACReasoner(),
        hypothesis_updater=update_hypotheses,
        tools=[
            ToolSpec(
                name="read_filter_pressure",
                description="Read pressure drop across the HVAC filter.",
                handler=lambda: {"pressure_pa": 12, "status": "normal"},
                cost=0.01,
                risk=0.0,
                reversible=True,
                metadata={"kind": "observe"},
            )
        ],
    )

    result = agent.run("Diagnose why HVAC airflow is low", max_steps=3)
    payload = result.to_dict()

    print(payload["answer"])
    print("\nHypotheses:")
    for hypothesis in payload["hypotheses"]:
        print(
            f"  {hypothesis['id']}: p={hypothesis['probability']:.3f} "
            f"status={hypothesis['status']} — {hypothesis['statement']}"
        )

    first_action = payload["decisions"][0]["selected"]
    print("\nFirst diagnostic action:")
    print(f"  {first_action['name']}")
    print(f"  tests={first_action['tests_hypotheses']}")
    print(f"  falsification_target={first_action['falsification_target']}")


if __name__ == "__main__":
    main()
