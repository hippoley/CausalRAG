import json

from causalrag.agent.actions import ActionKind, CandidateAction, DecisionRecord
from causalrag.agent.loop import CausalAgentLoop
from causalrag.agent.runtime import AgentRunResult, CausalAgent
from causalrag.agent.state import AgentState, Observation
from causalrag.reasoning.belief import LLMBeliefUpdater
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, *args, **kwargs):
        return json.dumps(self.payload)


class StopReasoner:
    def propose(self, state, world_model):
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "final causal answer"},
                rationale="enough evidence",
            )
        ]

    def uncertainty(self, state, world_model):
        return None


def test_reasoner_enforces_tool_metadata():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="dangerous_action",
                description="An irreversible action",
                handler=lambda: None,
                cost=0.4,
                risk=0.7,
                reversible=False,
                metadata={"kind": "intervene"},
            )
        ]
    )
    llm = FakeLLM(
        {
            "uncertainty": "outcome unknown",
            "candidates": [
                {
                    "kind": "observe",
                    "name": "dangerous_action",
                    "arguments": {},
                    "expected_goal_gain": 2.0,
                    "expected_information_gain": 0.0,
                    "cost": 0.0,
                    "risk": 0.0,
                    "irreversibility": 0.0,
                    "rationale": "try it",
                }
            ],
        }
    )
    reasoner = LLMCausalReasoner(llm, registry)
    action = list(reasoner.propose(AgentState(goal="test"), CausalWorldModel()))[0]

    assert action.kind == ActionKind.INTERVENE
    assert action.cost == 0.4
    assert action.risk == 0.7
    assert action.irreversibility == 1.0


def test_belief_updater_persists_defeasible_claim():
    llm = FakeLLM(
        {
            "updates": [
                {
                    "cause": "open window",
                    "effect": "indoor CO2 decreases",
                    "prior": 0.5,
                    "weight": 0.4,
                    "statement": "Observed CO2 falling after opening the window",
                    "kind": "intervention",
                    "mechanism": "increased air exchange",
                    "temporal_lag": "5 min",
                }
            ]
        }
    )
    updater = LLMBeliefUpdater(llm)
    world = CausalWorldModel()
    state = AgentState(goal="reduce CO2")
    action = CandidateAction(kind=ActionKind.INTERVENE, name="open_window")
    decision = DecisionRecord(
        step=0,
        uncertainty="ventilation effect unknown",
        candidates=[action],
        selected=action,
        beliefs_before=world.snapshot(),
    )
    observation = Observation(action_name="open_window", result={"co2_delta": -300})

    updater(state, world, decision, observation)
    belief = world.get("open window", "indoor CO2 decreases")

    assert belief is not None
    assert belief.probability > 0.5
    assert belief.mechanism == "increased air exchange"
    assert belief.temporal_lag == "5 min"


def test_direct_runtime_returns_serializable_final_answer():
    world = CausalWorldModel()
    loop = CausalAgentLoop(reasoner=StopReasoner(), world_model=world)
    agent = CausalAgent(loop=loop)

    result = agent.run("answer this", max_steps=2)
    payload = result.to_dict()

    assert isinstance(result, AgentRunResult)
    assert payload["answer"] == "final causal answer"
    assert payload["stop_reason"] == "enough evidence"
    assert payload["decisions"][0]["selected"]["kind"] == "stop"
    json.dumps(payload)
