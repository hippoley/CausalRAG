import json

from causalrag.agent.actions import ActionKind, CandidateAction, DecisionRecord
from causalrag.agent.loop import CausalAgentLoop
from causalrag.agent.state import AgentState, Observation
from causalrag.reasoning.hypothesis import HypothesisProposal, LLMHypothesisUpdater
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools.base import ToolRegistry, ToolSpec
from causalrag.world_model.models import CausalWorldModel


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, *args, **kwargs):
        return json.dumps(self.payload)


def test_reasoner_exposes_competing_hypotheses_and_test_metadata():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="read_filter_pressure",
                description="Read HVAC filter pressure drop",
                handler=lambda: {"pressure_pa": 82},
                metadata={"kind": "observe"},
            )
        ]
    )
    llm = FakeLLM(
        {
            "uncertainty": "airflow restriction source",
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": "The filter is clogged",
                    "probability": 0.55,
                    "rationale": "low airflow is compatible with restriction",
                    "falsifiers": ["normal filter pressure drop"],
                },
                {
                    "id": "H2",
                    "statement": "The fan is underperforming",
                    "probability": 0.45,
                    "rationale": "fan degradation can also lower airflow",
                    "falsifiers": ["normal fan RPM under load"],
                },
            ],
            "candidates": [
                {
                    "kind": "observe",
                    "name": "read_filter_pressure",
                    "arguments": {},
                    "expected_information_gain": 0.8,
                    "tests_hypotheses": ["H1", "H2"],
                    "falsification_target": "H1",
                    "rationale": "normal pressure would count against H1",
                }
            ],
        }
    )
    reasoner = LLMCausalReasoner(llm=llm, tools=registry)
    state = AgentState(goal="Find the airflow fault")
    world = CausalWorldModel()

    candidates = list(reasoner.propose(state, world))
    hypotheses = list(reasoner.hypothesis_proposals(state, world))

    assert [hypothesis.hypothesis_id for hypothesis in hypotheses] == ["H1", "H2"]
    assert candidates[0].tests_hypotheses == ["H1", "H2"]
    assert candidates[0].falsification_target == "H1"


def test_hypothesis_updater_records_falsifying_evidence():
    world = CausalWorldModel()
    world.upsert_hypothesis(
        "H1",
        "The filter is clogged",
        probability=0.7,
        falsifiers=["normal filter pressure drop"],
    )
    world.upsert_hypothesis("H2", "The fan is underperforming", probability=0.4)

    updater = LLMHypothesisUpdater(
        FakeLLM(
            {
                "updates": [
                    {
                        "id": "H1",
                        "weight": -0.8,
                        "statement": "Filter pressure drop is normal",
                        "kind": "falsifier",
                    },
                    {
                        "id": "H2",
                        "weight": 0.4,
                        "statement": "Normal filter pressure leaves fan fault more plausible",
                        "kind": "support",
                    },
                ]
            }
        )
    )
    selected = CandidateAction(
        kind=ActionKind.OBSERVE,
        name="read_filter_pressure",
        tests_hypotheses=["H1", "H2"],
        falsification_target="H1",
    )
    decision = DecisionRecord(
        step=0,
        uncertainty="airflow restriction source",
        candidates=[selected],
        selected=selected,
        beliefs_before=world.snapshot(),
    )
    observation = Observation(
        action_name="read_filter_pressure",
        result={"pressure_pa": 12, "status": "normal"},
    )

    updater(AgentState(goal="Find the airflow fault"), world, decision, observation)

    h1 = world.get_hypothesis("H1")
    h2 = world.get_hypothesis("H2")
    assert h1 is not None and h1.probability < 0.7
    assert h1.conflicting_evidence[-1].kind == "falsifier"
    assert h2 is not None and h2.probability > 0.4


class TwoStepReasoner:
    def __init__(self):
        self.calls = 0

    def propose(self, state, world_model):
        self.calls += 1
        if self.calls == 1:
            return [
                CandidateAction(
                    kind=ActionKind.OBSERVE,
                    name="read_sensor",
                    expected_information_gain=0.9,
                    tests_hypotheses=["H1", "H2"],
                    falsification_target="H1",
                )
            ]
        return [
            CandidateAction(
                kind=ActionKind.STOP,
                name="stop",
                arguments={"answer": "H2 remains more plausible"},
                rationale="diagnostic observation collected",
            )
        ]

    def uncertainty(self, state, world_model):
        return "which mechanism explains the outcome"

    def hypothesis_proposals(self, state, world_model):
        return [
            HypothesisProposal(
                hypothesis_id="H1",
                statement="Mechanism A explains the outcome",
                probability=0.5,
                falsifiers=["sensor reads normal"],
            ),
            HypothesisProposal(
                hypothesis_id="H2",
                statement="Mechanism B explains the outcome",
                probability=0.5,
                falsifiers=["sensor reads abnormal in A-specific way"],
            ),
        ]


def test_agent_loop_persists_hypotheses_before_action():
    world = CausalWorldModel()
    tools = ToolRegistry(
        [
            ToolSpec(
                name="read_sensor",
                description="Read a diagnostic sensor",
                handler=lambda: {"status": "normal"},
                metadata={"kind": "observe"},
            )
        ]
    )
    loop = CausalAgentLoop(
        reasoner=TwoStepReasoner(),
        tools=tools,
        world_model=world,
    )

    state = loop.run("diagnose mechanism", max_steps=3)

    assert state.done is True
    assert world.get_hypothesis("H1") is not None
    assert world.get_hypothesis("H2") is not None
    first_snapshot = state.decisions[0].beliefs_before
    assert {item["id"] for item in first_snapshot["hypotheses"]} == {"H1", "H2"}
    assert world.transitions[0].expected_effects["falsification_target"] == "H1"
