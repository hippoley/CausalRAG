from causalrag.experiments import ExperimentContract, OutcomeLikelihood
from causalrag.reasoning.llm import LLMCausalReasoner
from causalrag.tools import ToolRegistry, ToolSpec
from causalrag.world_model import CausalWorldModel
from causalrag.agent.state import AgentState


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, *args, **kwargs):
        return self.payload


def _tools():
    contract = ExperimentContract(
        experiment_id="sensor_test",
        outcomes=[
            OutcomeLikelihood("high", {"H1": 0.8, "H2": 0.2}),
            OutcomeLikelihood("low", {"H1": 0.2, "H2": 0.8}),
        ],
    )
    return ToolRegistry([
        ToolSpec(
            name="sensor",
            description="diagnostic sensor",
            handler=lambda: {"outcome": "high"},
            experiment_contract=contract,
            metadata={"kind": "observe"},
        )
    ])


def _world():
    world = CausalWorldModel()
    world.upsert_hypothesis("H1", "known one", probability=0.5)
    world.upsert_hypothesis("H2", "known two", probability=0.5)
    world.activate_model_mismatch()
    return world


def test_discovery_accepts_only_complete_normalized_predictions_for_known_experiments():
    llm = FakeLLM({
        "new_hypotheses": [
            {
                "id": "H3",
                "statement": "new mechanism with a testable signature",
                "rationale": "residual evidence",
                "falsifiers": ["sensor returns low"],
                "experiment_predictions": {
                    "sensor_test": {"high": 0.9, "low": 0.1},
                    "invented_test": {"x": 1.0},
                },
            },
            {
                "id": "H4",
                "statement": "bad incomplete prediction",
                "experiment_predictions": {
                    "sensor_test": {"high": 0.9},
                },
            },
            {
                "id": "H5",
                "statement": "bad non-normalized prediction",
                "experiment_predictions": {
                    "sensor_test": {"high": 0.9, "low": 0.9},
                },
            },
        ]
    })
    reasoner = LLMCausalReasoner(llm=llm, tools=_tools())
    proposals = list(reasoner.discover_hypotheses(
        AgentState(goal="diagnose", max_steps=4),
        _world(),
        {"predictive_probability": 0.01},
    ))

    assert [proposal.hypothesis_id for proposal in proposals] == ["H3"]
    assert proposals[0].experiment_predictions == {
        "sensor_test": {"high": 0.9, "low": 0.1}
    }


def test_discovery_rejects_existing_ids_and_hypotheses_without_testable_predictions():
    llm = FakeLLM({
        "new_hypotheses": [
            {
                "id": "H1",
                "statement": "attempt to overwrite known hypothesis",
                "experiment_predictions": {
                    "sensor_test": {"high": 0.9, "low": 0.1},
                },
            },
            {
                "id": "H3",
                "statement": "untestable story",
                "experiment_predictions": {},
            },
        ]
    })
    reasoner = LLMCausalReasoner(llm=llm, tools=_tools())
    proposals = list(reasoner.discover_hypotheses(
        AgentState(goal="diagnose", max_steps=4),
        _world(),
        {"predictive_probability": 0.01},
    ))
    assert proposals == []
