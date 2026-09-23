# Durable execution receipts

Branchpoint separates deciding that an action should run from proving whether an
effect has already run.

The execution ledger binds one `effect_id` to the canonical hash of a tool
name and JSON-compatible arguments. Reusing the same id for different semantic
content fails closed.

## Backends

`SQLiteExecutionLedger` is the zero-infrastructure default for local and
single-node processes.

`PostgresExecutionLedger` provides the same receipt contract for deployments
where multiple Branchpoint processes need to coordinate claims:

```bash
pip install "branchpoint[postgres]"
```

```python
from branchpoint import PostgresExecutionLedger, ToolRegistry

ledger = PostgresExecutionLedger(
    "postgresql://branchpoint:secret@db.internal/branchpoint"
)
registry = ToolRegistry(tools, execution_ledger=ledger)
```

The PostgreSQL claim path uses the receipt table's primary key and
`ON CONFLICT DO NOTHING`. Concurrent processes claiming the same effect id
converge on one new `in_flight` receipt; the other claimant sees the existing
receipt and does not receive execution authority.

## Crash-window rule

Neither backend claims exactly-once delivery for an arbitrary external side
effect. If a process dies after the external system accepted the action but
before Branchpoint records completion, the receipt remains `in_flight`.

A later attempt does not guess and does not automatically replay the effect. It
fails closed until an authoritative check calls `reconcile(...)` with the
actual external outcome.

For operations that support downstream idempotency, set
`ToolSpec.idempotency_key_argument`; Branchpoint propagates the effect id to
the external call in addition to maintaining its own durable receipt.

## Storage contract

Runtime code depends on the `ExecutionLedger` protocol rather than either
database implementation. A backend must implement `get`, `claim`,
`complete`, `fail`, and `reconcile` with the same effect-identity and
fail-closed semantics.

CI runs the SQLite contract in the normal suite and a live PostgreSQL 16 service
for cross-connection claim, replay, identity-conflict, failure, and
reconciliation tests.


## Async execution

`ToolRegistry.execute_async(...)` uses the same authorization, canonical
effect identity, receipt, replay, failure, idempotency, and reconciliation
contract as `execute(...)`.

Async handlers are awaited directly. Synchronous handlers are executed in a
worker thread, and synchronous receipt-store operations are also moved off the
event loop.

Calling the synchronous `execute(...)` API with an async handler fails before
any durable receipt is claimed. Applications therefore do not get a stranded
`in_flight` receipt merely because they chose the wrong execution API.

Explicit downstream idempotency keys are also validated before claiming a
receipt. If the caller supplies a value that conflicts with Branchpoint's
`effect_id`, the call fails before the ledger is mutated.

## Completion persistence failure

The external handler and the receipt completion write are separate failure
domains.

If the handler itself raises, Branchpoint records a `failed` receipt.

If the handler returns successfully but persisting `succeeded` fails,
Branchpoint leaves the receipt `in_flight`. At that point the external effect
may already have happened, so marking it failed would be false certainty. A
later retry fails closed and requires authoritative reconciliation before the
effect can move to a terminal state.

This is the same crash-window rule used for process death after an external
side effect and before durable acknowledgement.
