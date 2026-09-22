# Production execution contract

Branchpoint is a decision and execution-control runtime. It is not a distributed transaction system, an IAM provider, or a guarantee that arbitrary external APIs execute exactly once.

Use this contract when deciding whether a workflow is ready to cross from model output into a real side effect.

## What Branchpoint can enforce today

### 1. Proposal is not execution authority

A proposer (LLM, Jev, deterministic policy, or human-selected candidate) can nominate an action. Canonical tool risk, irreversibility, information value, temporal guards, authorization, and human gates remain runtime-owned.

### 2. Principal authority is checked at consequence time

`ToolSpec.required_permissions` comes from application code. `AuthorizationContext` comes from the current authenticated principal/session. Missing, incomplete, or expired authority fails closed before the tool handler runs.

Do not encode permissions only in prompts.

### 3. Effect identity is durable

`SQLiteExecutionLedger` binds an `effect_id` to the canonical tool + arguments envelope. Reusing the same id for another effect is rejected.

### 4. Completed effects are replayed, not re-executed

If a receipt is already `succeeded`, Branchpoint returns the stored JSON observation instead of calling the handler again.

### 5. Unknown crash windows fail closed

If the process dies while an effect is `in_flight`, Branchpoint does not guess whether the world changed. A retry is blocked until an authoritative reconciliation records the real outcome.

### 6. Downstream idempotency can be propagated

For APIs that accept idempotency keys, set `ToolSpec.idempotency_key_argument`. Branchpoint passes the same effect id into the external call.

## What your deployment must still provide

- a durable database appropriate to your availability requirements (SQLite is the reference single-node implementation);
- an authoritative identity / permission source and a fresh `AuthorizationContext` or resolver;
- downstream idempotency for high-value external APIs whenever available;
- reconciliation/read-back for side effects whose outcome can become unknown after a crash;
- secret management, TLS, network policy, audit retention, backup and disaster recovery;
- load, chaos, and domain-specific failure testing;
- a release/pinning strategy for models and policy versions.

## The crash-window rule

Never infer `failed` from “I did not receive a success response.”

For an external effect:

```text
claim effect id
      ↓
fresh authorization
      ↓
external side effect
      ↓
durable success receipt
```

If the process disappears between the last two steps, the correct state is **unknown**, not retry.

Recover with:

```text
authoritative read-back / reconciliation
      ↓
mark succeeded or failed
      ↓
continue
```

## Reference implementation

Run the no-key reference:

```bash
python examples/production_execution_boundary.py
```

It demonstrates one integrated path:

```text
model proposal: submit_form
        ↓
Branchpoint runtime: inspect first
        ↓
world: session expired
        ↓
authorized reauthentication
        ↓
authorized + idempotent submit
        ↓
durable receipt
        ↓
same effect id replayed without a second external submit
```

It also attempts the same external action as a guest principal and proves the handler is never reached.

## Readiness tiers

**Good fit now:** semantic routing, browser actions with verification, incident/workflow triage, human-approved operations, device commands with authoritative state read-back, internal automation with idempotent APIs.

**Needs additional infrastructure:** payments, production deploys, access-control changes, physical security, safety-critical devices, or any workflow where the external system cannot be queried or made idempotent.

Branchpoint should make those requirements explicit rather than hide them behind model confidence.
