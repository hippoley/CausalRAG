from branchpoint import (
    ActionKind,
    CandidateAction,
    CausalWorldModel,
    TemporalEffectContract,
    ToolSpec,
    VirtualTimeDriver,
    create_agent,
)


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "Actuator A controls the measured response.", probability=0.7)
    world.upsert_hypothesis("H2", "Another mechanism controls the response.", probability=0.3)
    return world


def _tools(clock, protect_attribution=True):
    intervention_times = []

    def intervene_a():
        intervention_times.append(("A", clock.now_seconds))
        return {"applied": "A"}

    def intervene_b():
        intervention_times.append(("B", clock.now_seconds))
        return {"applied": "B"}

    def read_response(sensor="main"):
        return {"status": "improved", "sensor": sensor, "at": clock.now_seconds}

    tools = [
        ToolSpec(
            name="intervene_a",
            description="Apply intervention A.",
            handler=intervene_a,
            metadata={"kind": "intervene"},
            temporal_effect_contract=TemporalEffectContract(
                effect_id="a_response",
                observe_with="read_response",
                observation_key="status",
                earliest_seconds=5.0,
                latest_seconds=10.0,
                expected_outcomes={"H1": "improved", "H2": "unchanged"},
                observe_arguments={"sensor": "main"},
                protect_attribution=protect_attribution,
            ),
        ),
        ToolSpec(
            name="intervene_b",
            description="Apply intervention B.",
            handler=intervene_b,
            metadata={"kind": "intervene"},
        ),
        ToolSpec(
            name="read_response",
            description="Read the delayed response.",
            handler=read_response,
            metadata={"kind": "observe"},
        ),
    ]
    return tools, intervention_times


class StackingReasoner:
    def propose(self, state, world_model):
        if not state.observations:
            return [CandidateAction(ActionKind.INTERVENE, "intervene_a")]
        if not any(o.action_name == "intervene_b" for o in state.observations):
            return [
                CandidateAction(
                    ActionKind.INTERVENE,
                    "intervene_b",
                    rationale="Try a second intervention immediately.",
                )
            ]
        return [CandidateAction(ActionKind.STOP, "stop", rationale="done")]

    def uncertainty(self, state, world_model):
        return "which intervention caused the response"


class OverwaitReasoner:
    def propose(self, state, world_model):
        if not state.observations:
            return [CandidateAction(ActionKind.INTERVENE, "intervene_a")]
        if not any(o.action_name == "read_response" for o in state.observations):
            return [
                CandidateAction(
                    ActionKind.WAIT,
                    "wait_far_too_long",
                    arguments={"seconds": 100.0},
                    rationale="Wait a long time before checking.",
                )
            ]
        return [CandidateAction(ActionKind.STOP, "stop", rationale="done")]

    def uncertainty(self, state, world_model):
        return "whether enough time has passed"


class MissWindowReasoner:
    def propose(self, state, world_model):
        if not state.observations:
            return [CandidateAction(ActionKind.INTERVENE, "intervene_a")]
        if state.virtual_time_seconds == 0.0:
            return [
                CandidateAction(
                    ActionKind.WAIT,
                    "wait_past_window",
                    arguments={"seconds": 20.0},
                )
            ]
        return [CandidateAction(ActionKind.STOP, "stop", rationale="window missed")]

    def uncertainty(self, state, world_model):
        return "whether the delayed effect can still be measured"


def test_second_intervention_is_deferred_until_first_effect_is_observed():
    clock = VirtualTimeDriver()
    tools, intervention_times = _tools(clock, protect_attribution=True)
    agent = create_agent(
        reasoner=StackingReasoner(),
        world_model=_world(),
        tools=tools,
        time_driver=clock,
    )

    result = agent.run("Preserve causal attribution while testing interventions.", max_steps=6)
    kinds = [decision.selected.kind for decision in result.state.decisions]
    names = [decision.selected.name for decision in result.state.decisions]

    assert kinds[:4] == [
        ActionKind.INTERVENE,
        ActionKind.WAIT,
        ActionKind.OBSERVE,
        ActionKind.INTERVENE,
    ]
    assert names[2] == "read_response"
    assert result.state.observations[2].result["sensor"] == "main"
    assert intervention_times == [("A", 0.0), ("B", 5.0)]
    assert result.state.pending_effects[0].observed is True


def test_excessive_wait_is_capped_at_first_valid_observation_time():
    clock = VirtualTimeDriver()
    tools, _ = _tools(clock, protect_attribution=True)
    agent = create_agent(
        reasoner=OverwaitReasoner(),
        world_model=_world(),
        tools=tools,
        time_driver=clock,
    )

    result = agent.run("Do not miss the effect observation window.", max_steps=5)

    assert result.state.observations[1].action_name == "wait_for_effect_window"
    assert result.state.observations[1].result["waited"] == 5.0
    assert result.state.observations[2].action_name == "read_response"
    assert result.state.virtual_time_seconds == 5.0
    assert result.state.pending_effects[0].expired is False
    assert result.state.pending_effects[0].observed is True


def test_missed_window_is_execution_failure_not_negative_causal_evidence():
    clock = VirtualTimeDriver()
    world = _world()
    h1_before = world.get_hypothesis("H1").probability
    tools, _ = _tools(clock, protect_attribution=False)
    agent = create_agent(
        reasoner=MissWindowReasoner(),
        world_model=world,
        tools=tools,
        time_driver=clock,
    )

    result = agent.run("Demonstrate a missed observation window.", max_steps=4)
    effect = result.state.pending_effects[0]
    events = result.state.scratch["temporal_events"]

    assert effect.expired is True
    assert effect.expiry_recorded is True
    assert events == [
        {
            "kind": "missed_observation_window",
            "effect_id": "a_response",
            "intervention": "intervene_a",
            "observe_with": "read_response",
            "ready_at": 5.0,
            "expires_at": 10.0,
            "detected_at": 20.0,
        }
    ]
    assert result.world_model.get_hypothesis("H1").probability == h1_before
    assert any(
        transition.action == "missed_observation_window"
        and transition.expected_effects["belief_update"] is False
        for transition in result.world_model.transitions
    )
