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
