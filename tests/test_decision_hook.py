from causalrag import ActionKind, CandidateAction, ToolSpec, create_agent


class TwoChoiceReasoner:
    def propose(self, state, world_model):
        if state.observations:
            return [
                CandidateAction(
                    kind=ActionKind.STOP,
                    name="stop",
                    arguments={"answer": "done"},
                    rationale="One observation is enough.",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="observe_a",
                expected_goal_gain=0.2,
                rationale="Proposer prefers A first.",
            ),
            CandidateAction(
                kind=ActionKind.OBSERVE,
                name="observe_b",
                expected_goal_gain=0.1,
                rationale="Alternative B.",
            ),
        ]

    def uncertainty(self, state, world_model):
        return "A versus B"


def test_decision_hook_overrides_execution_inside_canonical_agent_loop():
    calls = []

    def hook(state, world_model, decision):
        calls.append(
            {
                "step": state.step,
                "runtime_selected": decision.selected.name,
                "candidates": [candidate.name for candidate in decision.candidates],
            }
        )
        if state.step == 0:
            return next(
                candidate
                for candidate in decision.candidates
                if candidate.name == "observe_b"
            )
        return decision.selected

    agent = create_agent(
        reasoner=TwoChoiceReasoner(),
        decision_hook=hook,
        tools=[
            ToolSpec(
                name="observe_a",
                description="A",
                handler=lambda: {"source": "a"},
                metadata={"kind": "observe"},
            ),
            ToolSpec(
                name="observe_b",
                description="B",
                handler=lambda: {"source": "b"},
                metadata={"kind": "observe"},
            ),
        ],
    )

    result = agent.run("Test the decision hook.", max_steps=3)

    assert calls[0]["runtime_selected"] == "observe_a"
    assert result.state.decisions[0].selected.name == "observe_b"
    assert result.state.observations[0].action_name == "observe_b"
    assert result.state.observations[0].result == {"source": "b"}
    assert result.world_model.transitions[0].action == "observe_b"
