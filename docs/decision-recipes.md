# Decision recipes

Branchpoint is most useful when a workflow already knows **what kinds of actions are allowed**, but still needs help deciding **which branch to take next**.

This page collects small patterns that map well to a bounded decision runtime.

---

## 1. Tool selection

Use this when several tools could satisfy the request, but they differ in cost, risk, or information value.

```python
from branchpoint import ActionKind, CandidateAction

candidates = [
    CandidateAction(
        kind=ActionKind.RETRIEVE,
        name="search_logs",
        expected_goal_gain=0.55,
        expected_information_gain=0.72,
        cost=0.04,
        risk=0.01,
        rationale="Cheap evidence before touching production.",
    ),
    CandidateAction(
        kind=ActionKind.INTERVENE,
        name="restart_service",
        expected_goal_gain=0.90,
        expected_information_gain=0.05,
        cost=0.10,
        risk=0.35,
        irreversibility=0.10,
        rationale="May fix the symptom but destroys diagnostic state.",
    ),
]
```

The proposer can suggest candidates. The runtime still owns the execution boundary.

Good for:

- tool routing;
- skill selection;
- recovery actions;
- incident response;
- browser action selection.

---

## 2. Confidence gate

A useful system often has three outcomes rather than two:

```text
high confidence   → continue automatically
middle confidence → verify / ask another question
low confidence    → hand to a human
```

In Branchpoint this is represented as runtime policy, not as prose such as “I am fairly confident, so I will continue.”

The distinction matters because the threshold can depend on cost, risk, and reversibility.

---

## 3. Semantic routing

A router should not need to generate a paragraph to decide where work belongs.

```text
request state
    ↓
candidate routes
    ↓
probabilities + runtime constraints
    ↓
route / abstain / escalate
```

Typical routes include a support queue, installed skill, specialist agent, local vs hosted capability, fast vs expensive model, or human review.

Keep the routing result separate from the downstream response.

---

## 4. Stop / retry / replan

Agents often waste tokens because “continue” is the default. Treat continuation as a decision:

```text
progressing?
evidence changed?
same tool failing again?
goal already satisfied?
another probe worth its cost?
```

The allowed branches might be `continue / retry / change strategy / wait / ask human / stop`.

This is especially useful for long tool loops where another model call is expensive.

---

## 5. Browser and computer actions

A browser agent should not treat every clickable element as equally acceptable.

Before execution, separate `proposed action / runtime validity / risk / reversibility / expected information / expected goal gain`.

Low-risk observation can run automatically. Irreversible submission, deletion, purchase, permission change, or external communication can stop at the execution boundary.

---

## 6. Context selection

Context is also a bounded decision problem.

```text
candidate context items
    ↓
relevance / novelty / contradiction / cost
    ↓
include / defer / discard
```

The important property is **consumer-owned policy**: a semantic score is evidence, not authority.

That makes it possible to compare different scorers without rewriting the workflow.

---

## 7. Evidence before intervention

When an intervention is expensive or destructive, buy information first.

```text
H1: actuator failure
H2: sensor drift
H3: downstream blockage
```

Possible branches are `observe actuator state / cross-check sensor / measure pressure / replace actuator`.

A good runtime can prefer a diagnostic probe when its expected downstream value exceeds its cost.

That is different from choosing the tool whose description sounds most relevant.

---

## 8. Temporal attribution

A measurement can be real and still be invalid evidence.

For delayed systems: `intervene → wait for the causal window → observe → attribute`.

Reading too early should not update the belief state as though the intervention caused the result.

This matters in infrastructure, device control, physical systems, deployment rollouts, asynchronous jobs, and user-behavior experiments.

---

## 9. Open-world correction

Sometimes the correct explanation is not among the current candidates.

If observations remain surprising under every known hypothesis, the runtime should be able to say: **the model is incomplete**.

Then a new hypothesis can enter as provisional, gather evidence, and become validated or rejected.

This is a structural update, not just another probability tweak.

---

## 10. Human escalation

A human message should change operator context, not silently become ground truth.

A human-selected action should still pass runtime validity.

That yields a cleaner contract: `human steers → runtime guards → world verifies → state updates`.

---

## A useful integration test

For any new recipe, freeze `world / seed / goal / tools / budget / proposer`, then compare two policies and record the **first branch where they diverge**.

If the proposed decision layer does not improve the trajectory under controlled inputs, it has not earned its complexity.

---

## Next

- [Capability packs](capability-packs.md)
- [Experiment contracts](v0.3-experiment-contracts.md)
- [Hypothesis falsification](v0.3-hypothesis-falsification.md)
- [Runtime discrimination](runtime-discrimination.md)
