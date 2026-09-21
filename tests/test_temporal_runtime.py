from branchpoint import (
    ActionKind,
    CandidateAction,
    CausalWorldModel,
    TemporalEffectContract,
    ToolSpec,
    create_agent,
)


class TemporalReasoner:
    def propose(self, state, world_model):
        if state.step == 0:
            return [
                CandidateAction(
                    kind=ActionKind.INTERVENE,
                    name="open_valve",
                    rationale="Test the leading mechanism by changing the actuator.",
                )
            ]
        if state.step in {1, 2}:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_flow",
                    rationale="Observe the delayed effect.",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "temporal effect evaluated"},
                rationale="Temporal effect has been evaluated.",
            )
        ]

    def uncertainty(self, state, world_model):
        return "whether opening the valve increases flow"


def _run(observed_status):
    world = CausalWorldModel()
    world.upsert_hypothesis(
        "H1",
        "The valve is restricting flow.",
        probability=0.8,
    )
    world.upsert_hypothesis(
        "H2",
        "The downstream duct is restricting flow.",
        probability=0.2,
    )

    tools = [
        ToolSpec(
            name="open_valve",
            description="Open the control valve.",
            handler=lambda: {"applied": True},
            metadata={"kind": "intervene"},
            temporal_effect_contract=TemporalEffectContract(
                effect_id="valve_to_flow",
                observe_with="read_flow",
                observation_key="status",
                earliest_seconds=5.0,
                latest_seconds=20.0,
                expected_outcomes={
                    "H1": "improved",
                    "H2": "unchanged",
                },
                falsification_weight=0.5,
            ),
        ),
        ToolSpec(
            name="read_flow",
            description="Read current flow response.",
            handler=lambda: {"status": observed_status},
            metadata={"kind": "observe"},
        ),
    ]

    agent = create_agent(
        reasoner=TemporalReasoner(),
        world_model=world,
        tools=tools,
    )
    return agent.run("Identify whether the valve causally controls flow.", max_steps=4)


def test_premature_observation_is_replaced_by_wait_and_prediction_match_strengthens_hypothesis():
    result = _run("improved")
    state = result.state

    assert [decision.selected.kind for decision in state.decisions] == [
        ActionKind.INTERVENE,
        ActionKind.WAIT,
        ActionKind.OBSERVE,
        ActionKind.STOP,
    ]
    assert state.virtual_time_seconds == 5.0
    assert state.observations[1].result["waited"] == 5.0

    effect = state.pending_effects[0]
    assert effect.observed is True
    assert effect.matched_prediction is True
    assert effect.observed_value == "improved"

    temporal = state.observations[2].metadata["temporal_effects"][0]
    assert temporal["expected"] == "improved"
    assert temporal["observed"] == "improved"
    assert temporal["lag_seconds"] == 5.0
    assert temporal["within_window"] is True

    assert result.world_model.get_hypothesis("H1").probability > 0.8


def test_temporal_prediction_error_falsifies_the_hypothesis_that_was_actually_tested():
    result = _run("unchanged")
    state = result.state

    effect = state.pending_effects[0]
    assert effect.observed is True
    assert effect.matched_prediction is False

    temporal = state.observations[2].metadata["temporal_effects"][0]
    assert temporal["matched_prediction"] is False
    assert temporal["prediction_hypothesis"] == "H1"

    h1 = result.world_model.get_hypothesis("H1")
    assert h1.probability < 0.8
    assert h1.conflicting_evidence[-1].kind == "temporal_intervention_outcome"
