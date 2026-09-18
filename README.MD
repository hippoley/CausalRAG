> **[Open the verified Pages replay →](https://hippoley.github.io/CausalRAG/)** · static A/B evidence rebuilt from the current runtime by GitHub Actions. For the actual live tester, use `causalrag probe --open`.\n\n# CausalRAG v0.3 — from retrieval to causal decision

CausalRAG is a **causal decision runtime for agents operating under uncertainty**.

It started as causal-graph-enhanced RAG. In v0.3, retrieval is only one way to acquire evidence. The runtime now maintains explicit competing hypotheses, chooses what to observe or intervene on, updates beliefs from outcomes, and can value another experiment by whether it is expected to improve the eventual decision.

```text
Goal
  ↓
Belief state + competing hypotheses
  ↓
Candidate observations / interventions
  ↓
Runtime decision layer
  ├─ Bayesian information gain
  ├─ expected value of sample information (EVSI)
  ├─ intervention consequence utility
  └─ capability cost / risk / reversibility
  ↓
Observe / Retrieve / Ask / Intervene / Wait
  ↓
Outcome
  ↓
Posterior + world-model update
  ↓
Repeat or Stop
```

The central distinction is simple:

> **The model proposes. The runtime decides. The world corrects both.**

## Playable Causal Probe

The repository now has two deliberately different web surfaces:

| surface | what it is | what it can do |
| --- | --- | --- |
| GitHub Pages | static, deployment-generated verified replay | show same-world vanilla-vs-causal A/B traces and benchmark evidence from the deployed commit |
| Playable Causal Probe | live FastAPI control plane | external LLM proposer, session state, SSE trace, pre-execution human gate, candidate override, provisional hypotheses, natural-language operator messages, replanning, real tool execution, and same-world A/B |

GitHub Pages cannot host the live FastAPI/session backend, so the public Pages site no longer pretends to be the live tester.

Start the real probe locally:

```bash
pip install -e ".[api]"
causalrag probe --open
```

By default it serves on `127.0.0.1:8765`. For a remote/container host:

```bash
causalrag probe --host 0.0.0.0 --port 8765
```

The repository `Dockerfile` launches that same surface on port `8765`.

External proposer modes use server-side credentials:

```bash
export OPENAI_API_KEY=...       # frontier / OpenAI
export ANTHROPIC_API_KEY=...    # frontier / Anthropic
export LOCAL_LLM_URL=http://localhost:1234/v1
export CAUSALRAG_SMALL_MODEL=local-model
export CAUSALRAG_FRONTIER_MODEL=gpt-5.6-terra
```

The UI reports whether a provider is configured and exposes an explicit **Test** button for a real model call. Credentials are never sent to the browser.

### What happens in one live step

```text
external model / deterministic proposer
        ↓
structured hypotheses + candidate actions
        ↓
causal runtime validates and scores
        ↓
HUMAN GATE — no tool has executed yet
        ├─ approve runtime choice
        ├─ choose another runtime-valid candidate
        ├─ add a provisional hypothesis
        └─ send an operator message → replan
        ↓
tool executes
        ↓
observation → posterior / world-model update
        ↓
next gate
```

Human overrides still pass through runtime temporal safety, and candidates rejected by runtime validation cannot be forced through the UI. Human messages are treated as operator context, not automatically true evidence.

### Same-world A/B

The probe can run a vanilla tool loop and the causal control plane on the same frozen task inputs. It holds the hidden mechanism, seed, goal, tools, budgets, and proposer configuration fixed. Only runtime capabilities change.

For stochastic HiddenWorld comparisons, the A/B runner uses **action-indexed common random numbers**: the random variate is keyed by `seed + experiment_id + occurrence`, so the same experiment receives the same noise even if the two policies choose experiments in a different order.

For external models, identical prompts are memoized across arms. If both arms present the same prompt, only one provider call is made and the exact structured output is replayed into the other arm. Model calls separate only after the runtime states—and therefore prompts—actually diverge.

An LLM can suggest hypotheses and actions, but it does not own the belief state, capability economics, Bayesian update, or deployment consequence utility.

## Why this is not another tool-use loop

A normal agent loop can repeatedly do:

```text
LLM → tool → result → LLM
```

CausalRAG v0.3 adds explicit semantics around *why* another action is worth taking:

1. **Beliefs are explicit and defeasible.** Competing hypotheses live in a `CausalWorldModel`, not only in prompt context.
2. **Experiments have declared observation models.** `ExperimentContract` can represent `P(outcome | hypothesis, action)` and enables exact Bayesian posterior updates.
3. **Information and decision value are separated.** EIG asks how much uncertainty an experiment removes. EVSI asks whether the information is expected to change the best downstream decision enough to justify its cost.
4. **Interventions have consequence utilities.** `InterventionContract` evaluates actions under the current posterior rather than trusting model self-scores.
5. **Deployment owns preferences.** `DecisionPreferences` can change consequence utility without changing the reasoner, the tool implementation, or the causal world model.
6. **Behavior is benchmarked, not just prose.** HiddenWorld measures success, calibration, experiment cost, and causal regret over seeded stochastic episodes.

The legacy `CausalRAGPipeline` remains available as an optional compatibility layer.

## Install

Python **3.10+** is required.

The default install is the lightweight causal runtime. It does **not** install Torch, sentence-transformers, or FAISS.

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
pip install -e .
```

Add optional capability layers only when needed:

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

Core CI exercises Python 3.10 and 3.12 independently from the optional API and RAG gates.

## 30-second agent quickstart

```python
from causalrag import ToolSpec, create_agent


def read_co2(room: str):
    return {"room": room, "co2_ppm": 1180}


agent = create_agent(
    tools=[
        ToolSpec(
            name="read_co2",
            description="Read current CO2 concentration in a room.",
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

The runtime records candidate actions, scores, selected action, observations, beliefs, and transitions separately.

## Causal experiments: likelihoods instead of model guesses

When an observation has a known or estimated likelihood model, declare an `ExperimentContract`:

```python
from causalrag.experiments import ExperimentContract, OutcomeLikelihood

pressure_test = ExperimentContract(
    experiment_id="filter_pressure_test",
    outcomes=[
        OutcomeLikelihood("high",   {"H1": 0.90, "H2": 0.15}),
        OutcomeLikelihood("normal", {"H1": 0.10, "H2": 0.85}),
    ],
)
```

The runtime can then compute Bayesian expected information gain and update the posterior exactly when an outcome arrives. The LLM does not get a second chance to independently reinterpret the same evidence and double-count it.

Without a valid likelihood contract, the runtime falls back to weaker hypothesis-discrimination or model-estimated information signals rather than pretending it has Bayesian semantics.

## Observe or intervene? Use decision value

Information gain alone is not enough. An experiment can reduce uncertainty without changing the decision.

CausalRAG v0.3 can place experiments and interventions on the same expected-value axis when both contracts are available:

```text
best action now
vs.
expected best action after observing one more sample
```

For an experiment:

```text
EVSI =
  E_outcome [ best intervention value after posterior update ]
  - best intervention value now

net value of sampling = EVSI - experiment cost
```

For an intervention, expected utility is evaluated under the current posterior and its capability cost is subtracted. The runtime can therefore decide whether another observation is worth buying or whether it should act now.

The current implementation is **one-step value of information**, not a general POMDP solver.

## Deployment-owned consequence utility

Different deployments can value the same physical outcome differently. That preference should not live in the LLM prompt.

```python
from causalrag import DecisionPreferences, create_agent

preferences = DecisionPreferences(
    intervention_utilities={
        "repair_fan": {
            "H1": -1.0,
            "H2":  1.0,
            "H3": -1.0,
        },
        "clear_duct": {
            "H1": -1.0,
            "H2": -1.0,
            "H3":  1.0,
        },
    }
)

agent = create_agent(
    tools=my_tools,
    reasoner=my_reasoner,
    decision_preferences=preferences,
)
```

Ownership is intentionally separated:

```text
ExperimentContract      = what observations are likely under each hypothesis
InterventionContract    = default consequence semantics
DecisionPreferences     = how this deployment values intervention outcomes
ToolSpec                 = capability cost / risk / reversibility
Reasoner                 = proposes candidates
Runtime                  = performs arbitration
World                    = returns outcomes and updates evidence
```

`DecisionPreferences` creates effective runtime contracts without mutating the original tool objects.

## HiddenWorld: evaluate agency as behavior

`HiddenWorld` hides the true mechanism and gives the agent a finite diagnostic/intervention budget. The environment alone knows the truth; the agent receives priors, experiment contracts, capabilities, costs, and observations.

The first HVAC scenario contains three competing hidden mechanisms and noisy diagnostic tests. It is intentionally small: the purpose is to establish a reproducible behavioral benchmark contract before moving to hidden graph/SCM discovery.

A fixed stochastic suite can be run without an API key:

```bash
python examples/hidden_world_suite_demo.py
python examples/hidden_world_policy_comparison.py
python examples/hidden_world_risk_sweep.py
```

### Current fixed-suite baseline

The table below is **not a universal performance claim**. It is the measured result on the repository's fixed 30-episode suite: three hidden mechanisms × seeds `0..9`.

| policy | success | Brier ↓ | probes | cost | causal regret ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| greedy EIG | 0.800 | 0.340 | 2.10 | 0.346 | 0.296 |
| conservative EIG | 0.867 | 0.247 | 2.90 | 0.388 | 0.272 |
| decision value | 0.833 | 0.292 | **2.00** | **0.334** | **0.251** |
| random probe | 0.733 | 0.403 | 2.33 | 0.357 | 0.373 |
| cheapest probe | 0.633 | 0.390 | 2.73 | 0.344 | 0.461 |

The benchmark also exposed a useful non-monotonic utility frontier. On the same episodes, assigning increasing loss to a wrong intervention produced:

| wrong-action loss | success | Brier ↓ | probes | cost | causal regret ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.833 | 0.292 | 2.00 | 0.334 | 0.251 |
| 0.5 | 0.833 | 0.292 | 2.00 | 0.334 | 0.251 |
| 1.0 | **0.900** | **0.176** | 2.63 | 0.366 | **0.216** |
| 2.0 | 0.867 | 0.225 | 2.60 | 0.368 | 0.251 |

The useful result is not that `1.0` is a globally correct constant. It is that policy behavior can exhibit a phase change as consequence utility changes, and making a system "more risk averse" is not monotonically better.

## Run the no-key causal demos

```bash
python examples/hypothesis_falsification_demo.py
python examples/bayesian_experiment_demo.py
python examples/decision_preferences_demo.py
python examples/hidden_world_demo.py
python examples/hidden_world_suite_demo.py
python examples/hidden_world_policy_comparison.py
python examples/hidden_world_risk_sweep.py
```

These demos exercise the causal runtime without requiring a hosted reasoning model.

## Add causal retrieval

RAG remains useful as an evidence capability.

```bash
pip install -e ".[rag]"
```

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
    "Find the safest leverage point for reducing indoor CO2 and what evidence should be checked first.",
    max_steps=6,
)
```

By default, vector retrieval and causal-graph normalization share one hosted embedding provider with NumPy in-memory cosine search. Local sentence-transformers and FAISS remain opt-in.

### Local embeddings

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

### FAISS

```bash
pip install -e ".[rag,faiss]"
```

```python
agent = create_agent(
    documents=my_documents,
    vector_backend="faiss",
)
```

## Bring your own reasoner

The runtime does not require a vendor-specific model contract:

```python
agent = create_agent(
    tools=my_tools,
    reasoner=my_reasoner,
    belief_updater=my_belief_updater,
    decision_preferences=my_preferences,
)
```

This supports deterministic policies, local models, test doubles, domain-specific planners, or future RL/planning policies. The core path can run without an API key when a custom reasoner is supplied.

## CLI

```bash
causalrag --version
causalrag --help
```

Run an agent over documents:

```bash
causalrag agent \
  --goal "Which variable is the highest-leverage intervention and what should I verify first?" \
  --input ./docs \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --vector-backend memory \
  --max-steps 8
```

Build a reusable index:

```bash
causalrag index --input ./docs --output ./index
```

Legacy one-shot retrieval remains available:

```bash
causalrag query --index ./index --query "How can A lead to B?"
```

## HTTP API

```bash
pip install -e ".[api]"
causalrag serve --host 0.0.0.0 --port 8000
```

```http
GET /health
```

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

If the request includes documents, install both optional layers:

```bash
pip install -e ".[api,rag]"
```

The response contains the final answer, decision trace, observations, beliefs/hypotheses, and recorded transitions.

## Core primitives

### `CausalWorldModel`

Stores explicit causal beliefs, competing hypotheses, evidence, and action→observation transitions. The implementation is intentionally lightweight so future backends can include Bayesian SCMs, temporal models, simulators, neural world models, or hybrids.

### `ExperimentContract`

Declares discrete outcomes and their likelihood under competing hypotheses. Enables Bayesian EIG and exact posterior updates.

### `InterventionContract`

Declares consequence utility of an intervention under each causal hypothesis.

### `DecisionPreferences`

Deployment-owned overrides for consequence utility. The reasoner cannot rewrite them.

### `CandidateAction`

Supported action kinds:

```text
observe
retrieve
ask
intervene
wait
stop
```

### `DecisionRecord`

Stores uncertainty, all candidates, runtime score breakdown, selected action, and belief snapshot before action.

### `Transition`

Stores executed action → observation history, forming the base for future causal/experiment memory.

## Architecture

```text
┌──────────────────────────────────────────────┐
│                CausalAgent                   │
│ Goal → Propose → Decide → Act → Observe     │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│             Causal Decision Runtime          │
│ hypotheses · Bayes · EIG · EVSI · utility   │
│ capability cost/risk · DecisionPreferences  │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│             Explicit World Model             │
│ beliefs · evidence · transitions · outcomes │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│                 Environment                  │
│ retrieval · APIs · sensors · humans         │
│ simulators · software · robots              │
└──────────────────────────────────────────────┘
```

The LLM is a `Reasoner`: it can propose hypotheses/actions and explain them. It is not the world model or the final action authority.

## Repository layout

```text
causalrag/
├── agent/              # loop, state, actions, user-facing runtime
├── world_model/        # beliefs, hypotheses, evidence, transitions
├── reasoning/          # reasoner adapters, updates, runtime policy
├── experiments/        # experiment/intervention contracts, Bayes, EVSI, preferences
├── benchmarks/         # HiddenWorld, stochastic suites, policy comparisons
├── tools/              # capability registry and runtime metadata
├── embeddings/         # hosted/local/custom embedding abstraction
├── causal_graph/       # optional causal extraction + graph layer
├── retriever/          # optional vector/hybrid retrieval
├── reranker/           # optional causal reranking
├── interface/          # optional HTTP surfaces
└── pipeline.py         # legacy one-shot RAG compatibility
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
```

For new projects, prefer `create_agent()`.

## What v0.3 adds

- explicit competing hypotheses and falsification targets
- runtime-owned hypothesis discrimination
- `ExperimentContract` likelihood models
- Bayesian expected information gain
- exact posterior updates without LLM double-counting
- `InterventionContract` consequence utility
- one-step EVSI / observe-vs-intervene decision value
- deployment-owned `DecisionPreferences`
- runtime enforcement of capability cost/risk/reversibility
- deterministic and seeded stochastic HiddenWorld episodes
- Brier calibration, probe cost, and causal-regret metrics
- fixed-world policy tournament and utility-sensitivity sweeps
- no-key causal decision demos
- optional RAG compatibility rather than RAG as the core ontology

See [`CHANGELOG.md`](CHANGELOG.md) for the release summary.

## Next

The next layers should extend the same contracts rather than replace them:

1. **Temporal effects and observation windows** — action lag, persistence, and first-class `WAIT` semantics.
2. **Causal/experiment memory** — learn from `(state, action, next_state)` and reuse intervention outcomes across similar contexts.
3. **Hidden graph / SCM discovery** — move HiddenWorld from discrete mechanism identification to unknown structure and latent variables.
4. **Finite-horizon value of information** — replace one-step EVSI with multi-step planning when the extra complexity is justified.
5. **Drift and regime change** — detect when previously learned mechanisms stop predicting outcomes.
6. **Protocol adapters** — MCP/A2A/API/sensor adapters remain replaceable capability transports rather than core ontology.

The intended invariant is:

```text
experience → causal belief → decision value → action → new experience
```

## License

MIT. See `LICENSE`.
