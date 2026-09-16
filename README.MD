# CausalRAG v0.2 — from retrieval to intervention

CausalRAG is evolving from a causal-graph-enhanced RAG pipeline into a **goal-directed agent runtime built around an explicit causal world model**.

The core loop is no longer just:

```text
query → retrieve → rerank → answer
```

It is:

```text
Goal
  ↓
Belief state
  ↓
Candidate actions
  ↓
Causal utility selection
  ↓
Observe / Retrieve / Ask / Intervene / Wait
  ↓
Observation
  ↓
Transition + belief update
  ↓
Repeat or Stop
```

The runtime is designed around three ideas:

1. **Proposal is not decision.** A model proposes actions; a separate policy scores goal gain, information gain, cost, risk, and irreversibility.
2. **Beliefs are explicit.** The system keeps a causal world model and a trace of action → observation transitions instead of hiding all state inside the model context.
3. **Retrieval is one capability, not the identity of the system.** RAG remains useful for gathering evidence, but agents can also observe, ask, intervene, wait, and stop.

> v0.2 is currently on `feature/causal-agent-v2` and PR #3 while install/CI checks are completed. The legacy `CausalRAGPipeline` remains available, but new applications should start with `create_agent()`.

## Install the latest v0.2 branch

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
git checkout feature/causal-agent-v2
pip install -e .
```

For the default OpenAI-backed reasoner:

```bash
export OPENAI_API_KEY=...
```

Python 3.8+ is supported by the runtime API. CI currently exercises Python 3.10 and 3.12.

## 30-second quickstart

```python
from causalrag import create_agent

agent = create_agent(
    documents=[
        "Opening a window increases air exchange and can lower indoor CO2.",
        "More occupants increase indoor CO2 when ventilation is unchanged.",
        "Outdoor particulate pollution can make opening windows undesirable.",
    ]
)

result = agent.run(
    "Find the safest leverage point for reducing indoor CO2 and explain what evidence should be checked first.",
    max_steps=6,
)

print(result.answer)
print(result.to_dict()["decisions"])
```

A fresh agent automatically gets a `retrieve_evidence` capability backed by the existing hybrid vector + causal graph retrieval stack. Extracted graph edges seed the explicit belief state, and later observations can revise those beliefs.

## CLI

Run directly against a text file or directory:

```bash
causalrag agent \
  --goal "What should I change to reduce indoor CO2, and what should I verify first?" \
  --input ./docs \
  --max-steps 8
```

Print the complete decision trace:

```bash
causalrag agent \
  --goal "Find the most plausible causal explanation" \
  --input ./docs \
  --json
```

Create a reusable index:

```bash
causalrag index --input ./docs --output ./index
```

Use that index later:

```bash
causalrag agent \
  --goal "Which variable is the highest-leverage intervention?" \
  --index ./index
```

## HTTP API

Start the standalone v0.2 server:

```bash
causalrag serve --host 0.0.0.0 --port 8000
```

Health check:

```http
GET /health
```

Run the agent:

```http
POST /agent/run
Content-Type: application/json

{
  "goal": "Find the safest leverage point for reducing indoor CO2",
  "max_steps": 8,
  "model": "gpt-4o-mini",
  "provider": "openai",
  "documents": [
    "Opening a window increases air exchange and can lower indoor CO2."
  ]
}
```

The response contains the final answer plus the decision trace, observations, current beliefs, and recorded transitions.

## Add real-world capabilities

Tools are explicit capabilities rather than framework-specific tool calls:

```python
from causalrag import ToolSpec, create_agent


def read_co2(room: str):
    # Replace with a sensor/API call.
    return {"room": room, "co2_ppm": 1180}


agent = create_agent(
    tools=[
        ToolSpec(
            name="read_co2",
            description="Read the current CO2 concentration in a room.",
            handler=read_co2,
            cost=0.01,
            risk=0.0,
            reversible=True,
            metadata={"kind": "observe"},
        )
    ]
)

result = agent.run("Determine whether the bedroom needs more ventilation")
```

The model may propose the capability, but the runtime chooses among candidates using causal utility:

```text
utility =
    expected goal gain
  + expected information gain
  - cost
  - risk
  - irreversibility
```

Tool metadata is enforced by the runtime: model output cannot lower a capability's declared cost/risk or relabel an irreversible intervention as an observation.

`WAIT` and `STOP` are first-class actions. This matters for causal reasoning: after an intervention, the correct next action may be to wait for its expected effect instead of immediately changing another variable.

## Core primitives

### Goal

The desired state or question the agent is trying to resolve.

### CausalWorldModel

An explicit store for causal beliefs, evidence, and observed transitions. The current implementation is intentionally lightweight so the backend can later be replaced by a Bayesian SCM, temporal causal model, simulator, neural world model, or hybrid system without changing the agent loop.

### CandidateAction

Every proposal declares:

```text
kind
name
arguments
expected_goal_gain
expected_information_gain
cost
risk
irreversibility
rationale
```

Supported action kinds:

```text
observe
retrieve
ask
intervene
wait
stop
```

### DecisionRecord

Each step stores the uncertainty, all candidate actions, the selected action, and the belief snapshot before action. Explanations therefore come from an actual decision record rather than being reconstructed after the fact.

### Transition

Executed actions and their observations are stored as transitions. This is the foundation for causal memory and future experiment-driven belief learning.

### Belief update

After a new observation, the v0.2 runtime can extract conservative, defeasible directional causal updates. Evidence can strengthen or weaken a belief, and retrieved claims are kept distinct from an actual intervention performed by the agent.

## Architecture

```text
┌─────────────────────────────────────────┐
│              CausalAgent                │
│ Goal → Decide → Act → Observe → Learn  │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│          Explicit World Model           │
│ beliefs · evidence · transitions        │
│ future: SCM · temporal · experiments   │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│             Capabilities                │
│ retrieval · APIs · sensors · humans     │
│ simulators · software · robots          │
└─────────────────────────────────────────┘
```

The LLM is a `Reasoner`: it proposes hypotheses/actions and explains them. It is **not** treated as the world model or the final action authority.

## Current repository layout

```text
causalrag/
├── agent/
│   ├── actions.py       # action kinds + decision records
│   ├── state.py         # persistent runtime state
│   ├── loop.py          # thin causal learning/action loop
│   └── runtime.py       # CausalAgent + create_agent
├── world_model/
│   └── models.py        # beliefs, evidence, transitions
├── reasoning/
│   ├── base.py          # model-agnostic Reasoner protocol
│   ├── llm.py           # schema-constrained LLM proposer
│   ├── belief.py        # observation → defeasible belief updates
│   └── policy.py        # causal utility selection
├── tools/
│   └── base.py          # capability registry
├── causal_graph/        # existing extraction + graph layer
├── retriever/           # existing vector/hybrid retrieval
├── reranker/            # causal reranking
├── interface/
│   └── agent_api.py     # standalone /health + /agent/run
└── pipeline.py          # legacy one-shot RAG compatibility
```

## Legacy one-shot RAG

Existing code can continue to use:

```python
from causalrag import create_pipeline

pipeline = create_pipeline()
pipeline.index([
    "Climate change causes rising sea levels.",
    "Rising sea levels increase coastal flooding."
])

result = pipeline.run("How can climate change lead to coastal flooding?")
print(result["answer"])
print(result["causal_paths"])
```

For new projects, prefer `create_agent()`.

## What v0.2 already changes

- explicit `CausalWorldModel`
- persistent `AgentState`
- `observe / retrieve / ask / intervene / wait / stop`
- schema-constrained model-backed action proposals
- runtime-enforced capability risk/cost/reversibility
- causal utility action selection
- explicit decision records
- action → observation transition memory
- graph → initial belief synchronization
- observation → defeasible belief updates
- built-in hybrid causal evidence retrieval
- Python `create_agent()` API
- CLI agent entry point
- standalone HTTP `/agent/run`
- legacy pipeline and dependency compatibility fixes
- install/compile/agent-test CI

## Next: from agent loop to causal learning loop

The immediate roadmap is intentionally not “add more tools”. The next layers are:

1. **Hypothesis + falsification engine** — represent competing explanations and actively look for evidence that separates them.
2. **Experiment contracts** — intervention, prediction, observation window, confounders, result, prediction error.
3. **HiddenWorld benchmark** — agents must discover an unknown SCM through limited observations/interventions and reach a goal under a budget.
4. **Causal regret + calibration metrics** — evaluate whether actions were informative and effective, not just whether final prose looked correct.
5. **Temporal effects and drift** — model delays, waiting, regime change, and non-stationary mechanisms.
6. **Protocol adapters** — capabilities can later be exposed through MCP/A2A without making those protocols part of the core ontology.

The long-term invariant is:

```text
experience → causal belief → leverage → action → new experience
```

## License

MIT. See `LICENSE`.
