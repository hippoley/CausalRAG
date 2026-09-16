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

The legacy `CausalRAGPipeline` remains available as an optional compatibility layer, but new applications should start with `create_agent()`.

## Install v0.2

The default install is the lightweight causal-agent runtime. It does **not** install Torch, sentence-transformers, or FAISS.

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
pip install -e .
```

Add capabilities only when you need them:

```bash
pip install -e ".[rag]"               # lightweight document + causal retrieval
pip install -e ".[api]"               # FastAPI / HTTP server
pip install -e ".[local-embeddings]"  # sentence-transformers + PyTorch
pip install -e ".[faiss]"             # FAISS vector backend
pip install -e ".[evaluation]"        # pandas + Ragas evaluation
pip install -e ".[full]"              # hosted-runtime capability layers
```

For the default OpenAI-backed reasoner and hosted embedding provider:

```bash
export OPENAI_API_KEY=...
```

The default reasoning model is `gpt-5.6-terra` and the default embedding model is `text-embedding-3-small`. The hosted RAG path uses NumPy in-memory cosine search by default, so installing `[rag]` does not imply a local ML runtime.

Python **3.10+** is required for the v0.2 agent runtime. Core CI exercises Python 3.10 and 3.12 independently from optional RAG/API checks.

## 30-second core quickstart

You can use the causal agent runtime without RAG or embeddings:

```python
from causalrag import ToolSpec, create_agent


def read_co2(room: str):
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

result = agent.run(
    "Determine whether the bedroom needs more ventilation and what to verify first",
    max_steps=6,
)

print(result.answer)
print(result.to_dict()["decisions"])
```

The model proposes actions; the runtime selects among them and records decision, observation and belief state separately.

## Add causal retrieval

Install the lightweight RAG capability layer:

```bash
pip install -e ".[rag]"
```

Then documents can become an evidence capability without installing Torch or FAISS:

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
```

By default, vector retrieval and causal-graph node normalization share one hosted OpenAI embedding provider using `text-embedding-3-small`, with NumPy cosine search as the vector backend.

### Run embeddings locally

If you want an offline/local embedding model:

```bash
pip install -e ".[rag,local-embeddings]"
```

```python
agent = create_agent(
    documents=my_documents,
    embedding_provider_name="local",
    embedding_model="all-MiniLM-L6-v2",
)
```

### Use FAISS explicitly

FAISS is also opt-in:

```bash
pip install -e ".[rag,faiss]"
```

```python
agent = create_agent(
    documents=my_documents,
    vector_backend="faiss",
)
```

You can also inject your own `EmbeddingProvider` into `create_agent(..., embedding_provider=...)` or `create_pipeline(...)`. This keeps the causal graph and vector retrieval on the same embedding space without tying the runtime to a specific embedding vendor or local framework.

When retrieval is enabled, the agent gets a `retrieve_evidence` capability backed by the hybrid vector + causal graph retrieval stack. Extracted graph edges seed the explicit belief state, and later observations can revise those beliefs.

## Bring your own reasoner

The runtime does not require a vendor-specific model contract. You can inject a custom reasoner directly:

```python
agent = create_agent(
    tools=my_tools,
    reasoner=my_reasoner,
    belief_updater=my_belief_updater,
)
```

This is useful for deterministic policies, local models, test doubles, domain-specific planners, or a future non-LLM causal policy. The core path can run without an API key when you provide your own reasoner.

## CLI

Core commands such as version/help do not load the RAG stack:

```bash
causalrag --version
causalrag --help
```

To run against a text file or directory using hosted embeddings:

```bash
causalrag agent \
  --goal "What should I change to reduce indoor CO2, and what should I verify first?" \
  --input ./docs \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --vector-backend memory \
  --max-steps 8
```

Use local embeddings instead:

```bash
causalrag agent \
  --goal "Find the most plausible causal explanation" \
  --input ./docs \
  --embedding-provider local \
  --embedding-model all-MiniLM-L6-v2 \
  --json
```

Create a reusable hosted-embedding index:

```bash
causalrag index \
  --input ./docs \
  --output ./index \
  --embedding-provider openai
```

Use that index later:

```bash
causalrag agent \
  --goal "Which variable is the highest-leverage intervention?" \
  --index ./index
```

## HTTP API

Install the API layer:

```bash
pip install -e ".[api]"
```

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
  "model": "gpt-5.6-terra",
  "provider": "openai",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "vector_backend": "memory"
}
```

If the request includes `documents`, install both optional layers:

```bash
pip install -e ".[api,rag]"
```

The response contains the final answer plus the decision trace, observations, current beliefs, and recorded transitions. `steps` is the number of decision rounds including the terminal `STOP`; `executed_actions` counts non-STOP actions actually executed against a tool or the world.

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
├── embeddings/          # hosted/local/custom provider abstraction
├── causal_graph/        # optional causal extraction + graph layer
├── retriever/           # optional vector/hybrid retrieval
├── reranker/            # optional causal reranking
├── interface/
│   └── agent_api.py     # optional /health + /agent/run
└── pipeline.py          # optional legacy one-shot RAG compatibility
```

## Legacy one-shot RAG

Install the RAG layer and existing code can continue to use:

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
- GPT-5.6 Terra + Responses API as the default OpenAI path
- runtime-enforced capability risk/cost/reversibility
- causal utility action selection
- explicit decision records
- action → observation transition memory
- graph → initial belief synchronization
- observation → defeasible belief updates
- built-in hybrid causal evidence retrieval as an optional capability
- shared `EmbeddingProvider` abstraction across vector and causal graph retrieval
- lightweight hosted `[rag]` install with no Torch / sentence-transformers / FAISS
- opt-in local embeddings and FAISS backends
- lightweight default core install
- lazy-loaded RAG/API/evaluation/visualization layers
- injectable custom reasoner / LLM / belief updater / embedding provider
- Python `create_agent()` API
- CLI agent entry point
- standalone HTTP `/agent/run`
- legacy pipeline and dependency compatibility fixes
- separate core / API / RAG CI gates

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