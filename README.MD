<div align="center">

# Branchpoint

### Your model can propose. Your software still has to decide.

**A decision runtime for agents that need bounded choices, explicit uncertainty, and a defensible execution boundary.**

![Python](https://img.shields.io/badge/python-3.10%2B-111?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-111?style=flat-square)
![Status](https://img.shields.io/badge/status-beta-111?style=flat-square)

Branchpoint turns messy agent state into inspectable next actions: **choose a branch, score its value, gate its risk, execute it, read the world, update state, repeat.**

`typed actions` · `probabilistic decisions` · `confidence gates` · `tool routing` · `verification` · `human approval` · `counterfactuals`

[Verified examples](https://hippoley.github.io/Branchpoint/examples.html) · [Capability packs](docs/capability-packs.md) · [Decision recipes](docs/decision-recipes.md) · [Contributing](CONTRIBUTING.md)

</div>

<p align="center">
  <img src="docs/assets/branchpoint-hero.svg" alt="Branchpoint: state enters a branch, the runtime chooses, world evidence updates the next state" width="100%">
</p>

---

## The idea in one screen

Generative models are excellent **proposers**. Production software repeatedly needs something narrower:

```text
choose one option
score a risk
gate an action
decide whether evidence is enough
decide whether to continue
decide whether to ask a human
```

Those are bounded decisions inside a workflow. They should not disappear inside prose.

Branchpoint keeps them in code, with explicit state, probabilities, costs, risk, timing, and evidence.

> **State in. Decision out. World feedback becomes the next state.**

---

## A receipt, not a claim

One deterministic CI test runs the **same frozen world twice**:

```text
same hidden mechanism
same seed
same goal
same tools
same budget
same proposer
```

Only the control layer changes.

```text
plain tool loop        → fails the task
full Branchpoint loop  → succeeds
                         ↑
                 first divergence recorded
```

The repository tests the exact branch where the trajectories separate. It does not compare two unrelated prompts and call that an evaluation.

Run it:

```bash
pytest -q tests/test_playable_probe_api.py::test_probe_compare_api_runs_same_world_causal_and_vanilla_arms
```

---

## The 20-second version

Most agent loops look like this:

```text
model → tool → model → tool → answer
```

Branchpoint inserts a control layer between **what a model proposes** and **what the system actually does**:

```text
state
  ↓
competing hypotheses
  ↓
candidate actions
  ↓
┌────────────────────────────────────────────┐
│  score · probability · cost · risk · time │
└────────────────────────────────────────────┘
  ↓
choose / wait / verify / ask / intervene / stop
  ↓
world observation
  ↓
belief update
  ↓
next branch
```

If the world disagrees, the state changes.

If the current hypothesis set cannot explain the evidence, the hypothesis set can change too.

That is the point.

---

## Try it

### 1. Open the live workbench

```bash
pip install -e ".[api]"
branchpoint probe --open
```

Then open:

```text
/            product entrance
/demo        three focused failure-mode examples
/workbench   live agent history + human steering
/research    full score / trace / counterfactual console
```

No model key is required for deterministic examples.

### 2. Run one agent from Python

```python
from branchpoint import ToolSpec, create_agent


def read_co2(room: str):
    return {"room": room, "co2_ppm": 1180}


agent = create_agent(
    tools=[
        ToolSpec(
            name="read_co2",
            description="Read the current CO₂ concentration in a room.",
            handler=read_co2,
            cost=0.01,
            risk=0.0,
            reversible=True,
            metadata={"kind": "observe"},
        )
    ]
)

result = agent.run(
    "Decide what to verify before changing the ventilation state.",
    max_steps=6,
)

print(result.answer)
print(result.to_dict()["decisions"])
```

The answer is only one artifact. The runtime also records the candidates, scores, selected action, observations, belief changes, transitions, and trace.

---

## Where this fits

Branchpoint is useful anywhere an agent has to make **small, consequential decisions inside otherwise deterministic software**:

| decision | example |
| --- | --- |
| **choose** | which tool, skill, route, document, browser action, or recovery path |
| **score** | risk, urgency, relevance, quality, expected utility |
| **gate** | auto-execute, verify, ask a human, wait, or stop |
| **verify** | did the action actually cause the observed result? |
| **revise** | does the current hypothesis set still explain the world? |

Typical surfaces include:

`tool selection` · `semantic routing` · `browser actions` · `incident diagnosis` · `device control` · `workflow triage` · `code repair` · `context selection` · `agent evaluation`

The point is not to replace your model.

The point is to stop making one model response carry **proposal, policy, uncertainty, execution authority, and verification** all at once.

---

## Why Branchpoint exists

A fluent model response is not the same thing as a good software decision.

Production agents repeatedly face bounded decisions such as:

- Which tool should run next?
- Is the current evidence strong enough to act?
- Should the workflow continue, retry, wait, ask a human, or stop?
- Which hypothesis is worth testing first?
- Is an observation actually attributable to the action that preceded it?
- Did the world just invalidate the model we were using?
- Is another probe worth its latency, cost, or risk?

Branchpoint makes those decisions explicit.

| Ordinary loop | Branchpoint |
| --- | --- |
| State lives mostly in prompt history | Beliefs and hypotheses are first-class runtime state |
| Model output directly drives the next tool | Model proposes; runtime validates and arbitrates |
| Tool results become more text | Observations can update explicit probabilities |
| Every observation is treated as usable | Temporal attribution can reject stale evidence |
| Hypothesis set is fixed implicitly | Model mismatch can trigger hypothesis discovery |
| Human approval is bolted on | Human gates are part of the execution protocol |
| Debugging ends at logs | Every step can be inspected and forked |
| Evaluation asks “did it answer?” | Same-world runs measure trajectory differences |

---

## Three failure modes you can play

Branchpoint ships with three small environments because each exposes a different failure that ordinary tool loops hide.

### 01 · The most useful tool is not the cheapest tool

Several explanations are plausible. Probe budget is limited.

The runtime has to decide which observation is most likely to change the downstream action enough to justify its cost.

**What you see:** candidate scores, information value, selected probe, posterior movement.

### 02 · A real number can still be bad evidence

An actuator changes state, but the physical effect arrives later.

Reading too early returns a perfectly real measurement that should **not** be attributed to the intervention.

**What you see:** wait actions, temporal guards, valid observation windows.

### 03 · The correct explanation is missing

All known hypotheses assign very low probability to what the world keeps returning.

The correct move is not to keep redistributing probability among bad explanations.

**What you see:** model mismatch, a newly discovered hypothesis, structural world-model change.

---

## The live workbench

The workbench is built around **Agent History**, not a chat transcript.

Each row is one correction loop:

```text
BEFORE
H2 · 44%
   ↓
ACTION
airflow_test
   ↓
WORLD
window flow present, cross-room pressure ≈ 0
   ↓
AFTER
H2 · 78%
```

When the structure changes, the UI says so explicitly:

```text
WORLD MODEL CHANGED
new hypothesis: H4
```

### Manual mode

Pause before every execution.

Useful for debugging, teaching, and inspecting every branch.

### Safe Auto

Low-risk evidence actions can continue automatically.

Branchpoint pauses when the next branch deserves attention, for example:

- an intervention is about to execute;
- risk or irreversibility rises;
- runtime arbitration diverges from the proposer;
- a new hypothesis appears;
- model mismatch is detected;
- a human starts steering the session.

The goal is not “full autonomy.”

The goal is **attention at the right branch**.

---

## A decision is inspectable

A runtime decision keeps the pieces separate:

```text
prior state
candidate actions
runtime-valid candidates
score provenance
selected action
human override, if any
world observation
posterior state
structural hypothesis changes
telemetry
counterfactual branch
```

That separation matters because it lets you ask useful questions after the run:

- Did the model propose a bad action?
- Did the runtime score it incorrectly?
- Was the observation noisy?
- Did the belief update overreact?
- Did a human override improve or harm the trajectory?
- What would have happened if we chose candidate #2 instead?

---

## Capability packs

The workbench does not hard-code one domain.

A capability pack defines the pieces needed for one decision environment:

```text
world
+ tools
+ hypotheses
+ experiment contracts
+ intervention contracts
+ defaults
+ metrics
```

The frontend reads available packs from `/api/config`.

Add a new pack to the registry and it appears in the workbench without rewriting the UI.

Current packs include:

- hidden-world ventilation diagnosis;
- delayed-effect temporal attribution;
- open-world model mismatch.

This is the extension point for incident response, operations, device control, deployment diagnosis, support routing, evaluation workflows, and domain-specific agent systems.

---

## External models

Branchpoint can use a deterministic proposer, a small local model, or a hosted model.

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export LOCAL_LLM_URL=http://localhost:1234/v1

export BRANCHPOINT_SMALL_MODEL=local-model
export BRANCHPOINT_FRONTIER_MODEL=<your-model>
```

The proposer can suggest hypotheses and candidate actions.

It does **not** own:

- the canonical belief state;
- runtime validity;
- temporal guards;
- Bayesian updates backed by declared likelihoods;
- capability costs;
- intervention utility;
- human approval.

Those stay in application code.

---

## Declaring evidence

When you know how an experiment behaves under competing hypotheses, declare it.

```python
from branchpoint.experiments import ExperimentContract, OutcomeLikelihood

pressure_test = ExperimentContract(
    experiment_id="filter_pressure_test",
    outcomes=[
        OutcomeLikelihood("high",   {"H1": 0.90, "H2": 0.15}),
        OutcomeLikelihood("normal", {"H1": 0.10, "H2": 0.85}),
    ],
)
```

Now the runtime can compute the expected information gain before the probe and the exact posterior after the outcome.

No second model call is required to “reinterpret” the same evidence.

---

## Human steering is state, not magic

At a live gate you can:

```text
approve the runtime choice
choose another runtime-valid candidate
add a provisional hypothesis
send operator context
request a replan
```

A human message does not automatically become evidence.

A human-selected action does not bypass runtime validity.

That boundary is deliberate.

---

## Counterfactuals

Completed steps can be replayed from a frozen state with another candidate.

The fork rebuilds the same environment, reapplies runtime guards, follows the alternate branch, and reports the first future divergence.

This makes “what if we had chosen the other tool?” a runtime operation instead of a storytelling exercise.

---

## Same-world evaluation

Branchpoint includes paired runs where both policies receive the same:

```text
hidden mechanism
seed
goal
tools
budget
proposer configuration
```

Only the runtime capability profile changes.

That lets tests identify **where the trajectories first diverge**, not merely whether the final text is different.

CI covers deterministic and seeded environments, temporal attribution, open-world discovery, human gates, counterfactual forks, proposer audit, API surfaces, Docker smoke tests, and verified example replays.

---

## Project layout

```text
branchpoint/
├── agent/          decision loop, runtime arbitration, temporal effects
├── experiments/    likelihoods, information value, model mismatch
├── world_model/    hypotheses, beliefs, evidence, transitions
├── tools/          typed tool registry
├── reasoning/      proposer and hypothesis generation
├── probe/          sessions, examples, A/B, counterfactuals
├── observability/  trace + OpenTelemetry
├── interface/      FastAPI surfaces
└── templates/      examples, workbench, research console
```

---

## Install

Python 3.10+.

```bash
pip install -e .
```

Optional surfaces:

```bash
pip install -e ".[api]"               # FastAPI workbench
pip install -e ".[observability]"     # OTLP export
pip install -e ".[retrieval]"         # document evidence adapters
pip install -e ".[local-embeddings]"  # local embedding models
pip install -e ".[faiss]"             # FAISS vector backend
pip install -e ".[evaluation]"        # evaluation extras
pip install -e ".[full]"              # hosted capability layers
```

---

## Design principle

> **The model proposes. The runtime decides. The world corrects both.**

Branchpoint is deliberately opinionated about three things:

1. **A probability is more useful than confident prose when software must branch.**
2. **Execution authority belongs to the runtime, not to the proposer.**
3. **Reality must be able to change both the probabilities and the hypothesis set.**

If that resonates, try the examples, break a capability pack, add a strange environment, or open an issue with a failure mode you want the runtime to survive.

---

## Contributing

Small, falsifiable contributions are preferred over large speculative rewrites.

Good first contributions include:

- a new capability pack with a hidden-world test;
- a tool contract with explicit cost / risk / reversibility;
- a failure case where a normal loop takes the wrong branch;
- a new same-world comparison;
- a better visualization of one decision;
- a counterexample that breaks an existing guard.

See `CONTRIBUTING.md`.

---

## License

MIT.
