# Contributing to CausalRAG

CausalRAG is easiest to improve with **small, falsifiable additions**.

The best contribution is usually not “add a new abstraction.” It is one of these:

1. a failure case where a normal agent loop takes the wrong branch;
2. a capability pack that makes that failure reproducible;
3. a counterexample that breaks an existing runtime guard;
4. a clearer visualization of one decision boundary.

## The contribution contract

A useful change should answer four questions:

- **What state was the agent in?**
- **What choices were available?**
- **What did the world return?**
- **What behavior should change because of that evidence?**

If the answer cannot be tested, the contribution is probably too vague.

## Good first contributions

### Add a failure case

Create a minimal environment where at least two actions look plausible and one ordinary policy fails.

Include:

- hidden mechanism or frozen state;
- candidate actions;
- tool costs / risks / reversibility;
- expected observation behavior;
- success criterion;
- one test that reproduces the failure.

### Add a capability pack

A pack should define:

```text
world
+ tools
+ hypotheses
+ experiment contracts
+ intervention contracts
+ defaults
+ metrics
```

The workbench reads packs from the runtime registry. A new pack should appear without special-case frontend code.

### Break a guard

Counterexamples are welcome.

If temporal attribution, model mismatch, score arbitration, Safe Auto, or counterfactual replay behaves incorrectly, open an issue with a minimal trace or add a regression test.

## Development

```bash
python -m pip install -e ".[api,dev]"
pytest -q
```

Run the live surfaces:

```bash
branchpoint probe --open
```

Useful paths:

```text
/            product entrance
/demo        focused examples
/workbench   live session UI
/research    deep runtime console
```

## Pull requests

Keep PRs narrow.

A strong PR normally includes:

- one clearly stated behavior change;
- tests that fail before the change and pass after it;
- no unrelated refactor;
- screenshots only when the UI behavior changed;
- a short note on truth boundaries if the UI is replaying generated artifacts rather than live execution.

## Style

Prefer:

- explicit state over prompt-only state;
- deterministic tests over prose claims;
- typed tool contracts over implicit conventions;
- runtime-owned policy over model self-scoring;
- observability that explains where trajectories diverged.

Avoid:

- hiding uncertainty inside generated text;
- silently treating operator messages as evidence;
- bypassing runtime validity during human override;
- frontend-only simulations presented as backend behavior.

## Before opening a PR

```bash
pytest -q
python -m compileall branchpoint
```

If you changed the probe surfaces, also run the browser and verify one full trajectory manually.

## Discussion

If your idea changes the runtime contract, open an issue first.

If your idea is a concrete failure mode with a reproducible environment, a PR is usually the fastest way to discuss it.


## A 10-minute first contribution

You do not need to understand the whole runtime.

1. Run one no-key example.
2. Change one state, risk, cost, or observation.
3. See whether the selected action changes.
4. If the behavior surprises you, capture the smallest reproducible case.

Good contributions often start as: **“I expected branch A, but the runtime chose branch B because…”**

That is more useful than a broad feature proposal.
