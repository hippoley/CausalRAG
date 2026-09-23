# Framework-neutral execution gateway

Branchpoint can protect a tool boundary even when the calling agent framework is
not written in Python.

The HTTP gateway exposes two operations:

- `POST /v1/preview`: inspect current server-owned policy without executing;
- `POST /v1/execute`: re-evaluate policy and cross the durable execution
  boundary.

The client sends a tool name, JSON arguments, and an effect id. It cannot send
its own permissions, canonical risk, reversibility, or policy.

## Run the reference gateway

```bash
pip install -e ".[api]"
uvicorn examples.execution_gateway_app:app --reload
```

The reference app uses SQLite. A multi-instance deployment can provide
`PostgresExecutionLedger` instead.

The `x-principal` / `x-permissions` headers in the reference app are **demo-only**. Do not trust identity headers sent directly by an Internet client. In production, resolve identity from a verified session/JWT, mTLS, or headers injected by an authenticated proxy you control.

## Preview

```bash
curl -s http://127.0.0.1:8000/v1/preview \
  -H 'content-type: application/json' \
  -H 'x-principal: support:alice' \
  -H 'x-permissions: order.read' \
  -d '{
    "tool_name": "read_order",
    "arguments": {"order_id": "O-1"}
  }'
```

The response contains a Branchpoint decision and a canonical `proposal_hash`:

```json
{
  "decision": {
    "outcome": "allow",
    "tool_name": "read_order",
    "proposal_hash": "...",
    "reason_code": "auto_execute",
    "risk": 0.0,
    "reversible": true
  },
  "executes_tool": false
}
```

A high-risk or irreversible action returns `require_human` instead.

## Execute

Use a unique, stable `effect_id` for one semantic side effect:

```bash
curl -s http://127.0.0.1:8000/v1/execute \
  -H 'content-type: application/json' \
  -H 'x-principal: support:alice' \
  -H 'x-permissions: order.read' \
  -d '{
    "tool_name": "read_order",
    "arguments": {"order_id": "O-1"},
    "effect_id": "support-ticket-42:read-order"
  }'
```

HTTP execution always requires a configured Branchpoint execution ledger.
Repeating the exact request with the same effect id replays the stored result
instead of invoking the handler again. Reusing that effect id for another tool
or another argument payload returns an identity conflict.

## Trusted human approval

A high-risk HTTP action cannot be executed with a client-controlled
`human_approved: true` flag.

The application supplies an `approval_validator` callback when it creates the
gateway. For a `require_human` decision, the client must:

1. obtain a preview;
2. keep the returned `proposal_hash`;
3. complete the application's trusted review flow;
4. send the same proposal hash, one stable effect id, and the opaque approval
   artifact to `/v1/execute`.

The gateway passes the request, current decision, effect id, and opaque token to
the application's validator. A production validator should normally verify a
signed or one-time approval artifact that binds at least:

```text
principal / reviewer
tool
proposal_hash
effect_id
expiry
```

The reference app uses a deliberately simple token only so the zero-key demo is
easy to inspect.

## TOCTOU protection

For human-approved calls, `expected_proposal_hash` is mandatory. If the tool
or arguments change after preview, execution returns `proposal_changed`
before claiming a receipt.

Authority is not frozen by preview either. The server resolves the current
principal again on execute, and `ToolRegistry.execute_async(...)` checks
authorization again immediately before the receipt/external-effect boundary.

## Why the principal is not in JSON

The gateway's `principal_resolver(Request)` is application-owned code. It can
read a verified session, mTLS identity, JWT claims, reverse-proxy headers, or any
other trusted identity source.

Permissions submitted by the model or caller body are rejected as extra fields.
This keeps model output separate from execution authority.

## Integration shape

Any harness that can make HTTP requests can use the same flow:

```text
agent / orchestrator
       ↓
 proposed tool call
       ↓
 POST /v1/preview
       ↓
 allow ────────────────┐
 require_human → review│
 deny → stop           │
                       ↓
                POST /v1/execute
                       ↓
            authorization re-check
                       ↓
              durable receipt claim
                       ↓
                application handler
                       ↓
               succeeded receipt
                       ↓
                 tool observation
```

That makes the execution boundary independent of OpenAI Agents, LangGraph, a
browser agent, an MCP client, or a custom orchestrator. Those systems can keep
their own planner and run loop while Branchpoint owns the consequence boundary.
