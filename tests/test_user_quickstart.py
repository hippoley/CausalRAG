from causalrag import ActionKind, CandidateAction, ToolSpec, create_agent


class LocalReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_sensor",
                    arguments={"room": "bedroom"},
                    expected_information_gain=0.9,
                    rationale="Read the local sensor before deciding.",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                rationale="Enough local evidence.",
                arguments={"answer": "Sensor checked; no remote model was required."},
            )
        ]

    def uncertainty(self, state, world_model):
        return "sensor state unknown" if state.step == 0 else None


def test_create_agent_runs_without_api_key_or_rag_dependencies():
    agent = create_agent(
        reasoner=LocalReasoner(),
        tools=[
            ToolSpec(
                name="read_sensor",
                description="Read a local room sensor.",
                handler=lambda room: {"room": room, "co2_ppm": 900},
                metadata={"kind": "observe"},
            )
        ],
    )

    result = agent.run("Check the bedroom before acting", max_steps=3)
    payload = result.to_dict()

    assert payload["steps"] == 2
    assert payload["observations"][0]["result"]["co2_ppm"] == 900
    assert payload["stop_reason"] == "Enough local evidence."
