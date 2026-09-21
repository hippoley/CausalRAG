# Capability Packs

A **capability pack** turns a domain into a decision environment that Branchpoint can run, inspect, evaluate, and replay.

It is intentionally smaller than an application and more explicit than a prompt.

A good pack defines enough structure for the runtime to answer:

> Given this state, which branch is worth taking next, and what evidence would justify changing that choice?

## Contract

A pack should provide:

```text
world state
+ competing hypotheses
+ typed candidate actions
+ tools
+ experiment contracts
+ intervention contracts
+ costs / risks / reversibility
+ timing constraints
+ defaults
+ metrics
```

The frontend discovers registered packs through `/api/config`.

That means a pack should not require custom UI code.

## Packs included in the repository

| pack | what it makes explicit |
| --- | --- |
| `browser_action_guard` | proposal order vs canonical risk / irreversibility at the execution boundary |
| `tool_routing` | private read vs fresh public information vs state-changing work |
| `incident_triage` | weak generic evidence vs a diagnostic that can change the intervention |
| `hvac_hidden_world` | information value under a limited probe budget |
| `temporal_delayed_effect` | attribution windows for delayed effects |
| `open_world_mismatch` | structural hypothesis discovery when the current model stops fitting |

All six use the same session, trace, human-gate, comparison, and Workbench surfaces.

The first three are good templates for application developers. The last three are good templates for deeper runtime behavior.

## Register a pack without editing Branchpoint

Capability packs are registry-backed. A complete runnable external pack is included at:

```text
examples/custom_capability_pack.py
```

Run it directly:

```bash
python examples/custom_capability_pack.py
```

It registers `queue_incident_demo`, runs a real episode through the normal runtime, and then unregisters itself.

Inspect the same application-owned pack through the CLI registry:

```bash
branchpoint packs \
  --load-pack examples/custom_capability_pack.py:register_pack
```

Or load it into the real Workbench:

```bash
branchpoint probe \
  --load-pack examples/custom_capability_pack.py:register_pack \
  --open
```

The external file exposes a normal registration function:

```python
def register_pack():
    return register_probe_scenario(
        ProbeScenarioSpec(
            scenario_id="queue_incident_demo",
            label="Queue incident demo",
            description="An application-owned pack registered at runtime.",
            hidden_hypotheses=("H1", "H2"),
            outcome_modes=("deterministic", "stochastic"),
            recommended_test="External capability-pack registration",
            default_hidden_hypothesis="H2",
            default_outcome_mode="deterministic",
            default_goal="Diagnose why jobs are backing up.",
            builder=build_pack,
        ),
        replace=True,
    )
```

After registration:

- `ProbeRunConfig(scenario="queue_incident_demo", ...)` validates against the new spec;
- `/api/config` includes the new pack;
- the Live Workbench scenario selector discovers it automatically;
- sessions, Human Gate, trace, export, comparison, and evaluation reuse the existing runtime surface.

The builder must return `ProbeScenarioRuntime`. Branchpoint deliberately does not require custom frontend code for each domain.

## 1. World state

The world model contains explicit, defeasible state.

At minimum, a pack should expose hypotheses with probabilities:

```json
{
  "hypotheses": [
    {
      "id": "H1",
      "statement": "The sensor is drifting",
      "probability": 0.30
    },
    {
      "id": "H2",
      "statement": "There is no effective cross-room airflow",
      "probability": 0.45
    },
    {
      "id": "H3",
      "statement": "The exhaust path is restricted",
      "probability": 0.25
    }
  ]
}
```

A pack may add richer state, but the important property is that uncertainty is inspectable.

## 2. Typed actions

Candidate actions should communicate what kind of branch they represent.

Typical kinds are:

```text
observe
retrieve
ask
wait
intervene
stop
```

Each action should carry enough metadata for runtime arbitration:

```python
CandidateAction(
    name="cross_room_pressure_test",
    kind=ActionKind.OBSERVE,
    rationale="Distinguish local window flow from effective room ventilation.",
    tests_hypotheses=["H2", "H3"],
    expected_information_gain=0.4,
)
```

The proposer may suggest actions.

The runtime owns whether those actions are valid.

## 3. Tools

Tools are executable boundaries, not prose affordances.

Prefer tools that declare:

- cost;
- risk;
- reversibility;
- side effects;
- timing behavior;
- the action kind they support.

A tool result should become an observation before it affects the world model.

```python
ToolSpec(
    name="cross_room_pressure_test",
    description="Read cross-room pressure without changing device state.",
    handler=read_pressure,
    cost=0.03,
    risk=0.0,
    reversible=True,
    metadata={"kind": "observe"},
    experiment_contract=pressure_test,
)
```

**Tool metadata is canonical.** A proposer cannot make a dangerous tool safe by reporting a lower cost or risk. During scoring, the runtime reconciles candidate metadata against the registered tool boundary; a non-reversible tool remains non-reversible regardless of what the proposer says.

## 4. Experiment contracts

When an observation has a known or estimated likelihood model, declare it.

```python
from branchpoint.experiments import ExperimentContract, OutcomeLikelihood

pressure_test = ExperimentContract(
    experiment_id="pressure_test",
    outcomes=[
        OutcomeLikelihood(
            "crossflow_present",
            {"H1": 0.20, "H2": 0.10, "H3": 0.75},
        ),
        OutcomeLikelihood(
            "crossflow_absent",
            {"H1": 0.15, "H2": 0.80, "H3": 0.20},
        ),
    ],
)
```

Now the runtime can compute probability-weighted information value before the experiment and exact posterior updates after the outcome.

## 5. Intervention contracts

An intervention changes the world rather than merely observing it.

A pack should make that distinction explicit because intervention utility may depend on:

- expected benefit;
- cost;
- risk;
- reversibility;
- current belief state.

Do not score an intervention only by asking a language model whether it “seems useful.”

## 6. Timing

Some actions create effects that arrive later.

If timing matters, define a temporal effect contract.

The runtime should be able to answer:

- When is an observation eligible for attribution?
- Is this read too early?
- Is this effect stale?
- Can two pending effects overlap?

A pack that ignores timing should not claim causal attribution for delayed systems.

## 7. Human approval

Human approval is a runtime branch, not a special comment channel.

A pack can rely on Branchpoint's Human Gate protocol to:

- approve the runtime-selected candidate;
- select another runtime-valid candidate;
- add a provisional hypothesis;
- provide operator context;
- request a replan.

Operator text is context.

It is not automatically evidence.

## 8. Safe Auto

Safe Auto lets low-risk evidence-gathering branches continue while pausing at meaningful decision boundaries.

A pack should classify actions well enough that the runtime can distinguish:

- evidence collection;
- waiting;
- reversible verification;
- interventions;
- elevated-risk actions.

The UI can then keep attention for the branches that deserve it.

## 9. Model mismatch

A fixed hypothesis set is often wrong.

A strong pack should define what “the current model no longer explains the world” means.

Examples:

- repeated low predictive probability;
- high surprisal;
- a diagnostic outcome impossible under all known hypotheses.

Model mismatch can trigger hypothesis discovery.

That structural change should appear in the episode ledger.

## 10. Metrics

A pack needs a behavioral success criterion.

Useful metrics include:

- mechanism identification;
- task success;
- posterior calibration;
- unnecessary probes;
- total cost;
- harmful interventions;
- time to valid evidence;
- causal regret;
- human interventions;
- first trajectory divergence.

Prefer metrics that inspect behavior, not generated prose.

## Minimal pack checklist

Before registering a pack, verify:

- [ ] The world has at least two plausible hypotheses.
- [ ] There is at least one branch where choosing poorly matters.
- [ ] Tools declare cost / risk / reversibility where relevant.
- [ ] Observations have a clear path into state updates.
- [ ] Timing rules are explicit if effects can be delayed.
- [ ] Success is machine-checkable.
- [ ] A deterministic test reproduces one meaningful failure.
- [ ] The pack appears in `/api/config`.
- [ ] The workbench can launch it without custom frontend code.

## Good pack ideas

Branchpoint is especially useful when the next step is a bounded software decision under uncertainty:

- incident triage;
- deployment diagnosis;
- device control;
- home automation;
- operations workflows;
- support routing;
- verification pipelines;
- browser agents;
- data-quality investigation;
- experiment selection.

The interesting question is always the same:

> What should the system do next, what evidence would change that choice, and when should a human take over?
