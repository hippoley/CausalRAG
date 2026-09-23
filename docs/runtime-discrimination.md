# Runtime hypothesis discrimination

Branchpoint's proposer can propose candidate actions, but it should not be able to decide which action wins merely by assigning itself a large `expected_information_gain`.

This layer introduces a runtime-owned discrimination baseline whenever an action explicitly declares which hypotheses it tests.

## Decision boundary

```text
LLM / custom reasoner
  ↓ proposes
CandidateAction
  ↓
Runtime policy
  ├─ goal-gain bound
  ├─ hypothesis discrimination
  ├─ capability cost
  ├─ capability risk
  └─ irreversibility
  ↓ selects
Action
```

When `tests_hypotheses` is empty, the runtime keeps the old model-estimated information gain as a compatibility fallback.

When `tests_hypotheses` is present, runtime state becomes authoritative for the information component.

## Discrimination baseline

The current score uses three observable quantities.

### 1. Hypothesis coverage

Let active hypothesis credence mass be:

```text
M = sum(p_i)
```

For an action testing a subset `T`:

```text
coverage = sum(p_i for i in T) / M
```

An action that tests only a low-credence side hypothesis should generally be less valuable than one covering most of the live explanatory mass.

### 2. Ambiguity among tested hypotheses

For two or more tested hypotheses, their credences are normalized within the tested set and the runtime computes normalized entropy:

```text
q_i = p_i / sum(p_j in T)

ambiguity = H(q) / log(|T|)
```

The multi-hypothesis discrimination component is:

```text
discrimination = coverage * ambiguity
```

This is high when an action covers important hypotheses that are still difficult to distinguish.

### 3. Falsification leverage

A single observation can still be valuable when it directly challenges a dominant explanation.

If `falsification_target` names an active tested hypothesis:

```text
falsification_leverage = coverage * target_credence
```

The runtime information score is:

```text
runtime_information = max(discrimination, falsification_leverage)
```

bounded to `[0, 1]`.

## Why this is not Bayesian VOI

The policy does not yet know outcome likelihoods such as:

```text
P(observation | H1, action)
P(observation | H2, action)
```

Without those likelihood models, exact expected posterior entropy reduction is not available.

So this score should be read as:

> a deterministic prior on how diagnostically targeted an action is, given the agent's explicit hypothesis state.

It should **not** be described as mathematically exact expected information gain.

A future experiment-contract layer can add predicted observation distributions and replace this baseline with true expected information gain / value of information.

## Policy hardening

Model-proposed heuristics are bounded:

```text
0 <= expected_goal_gain <= 1
0 <= expected_information_gain <= 1
```

Tool cost, risk and reversibility continue to come from runtime capability metadata rather than model output.

If an action claims it tests hypotheses but those IDs are unknown or rejected, its runtime information score is `0`; it does not fall back to the model's self-score.

This prevents a proposal like:

```text
expected_information_gain = 100
 tests_hypotheses = ["UNKNOWN"]
```

from bypassing the runtime policy.

## Traceability

Every `DecisionRecord` now contains `action_scores`.

For each candidate the trace records:

```text
action_name
action_kind
total_utility
goal_gain
information_gain
information_source
model_information_gain
discrimination_score
cost
risk
irreversibility
```

For an explicit hypothesis test:

```text
information_source = runtime_hypothesis_discrimination
```

Otherwise:

```text
information_source = model_estimate
```

The selected transition stores the same information source and both runtime and model information values, so evaluation can later compare model self-assessment with observed experiment usefulness.

## Runnable example

```bash
python examples/hypothesis_falsification_demo.py
```

The example deliberately has the proposer rate a generic temperature reading as more informative than the diagnostic filter-pressure measurement. The runtime should override that ranking because the pressure measurement explicitly discriminates H1/H2 and can falsify H1.

That is the architectural point of this layer:

```text
model proposes what might matter
runtime decides what evidence structure says matters
world answers
runtime updates belief
```
