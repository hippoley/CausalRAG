# MCP execution boundary

Branchpoint can expose its consequence boundary as a native MCP v2 server.

This integration is tested against the current MCP Python SDK v2 line and the
2026-07-28 protocol behavior.

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
pip install -e ".[mcp]"
python examples/mcp_execution_boundary.py
```

With no transport argument the MCP SDK runs over stdio. The same MCPServer can
also be hosted with the SDK's Streamable HTTP transport.

## What the model can see

The server exposes two tools:

```text
branchpoint_preview(tool_name, arguments)
branchpoint_execute(tool_name, arguments, effect_id)
```

Identity and approval are deliberately absent from the model-facing schemas.

They are injected as MCP resolver dependencies:

```text
model arguments
     │
     ├── tool_name
     ├── arguments
     └── effect_id
             │
             ▼
      hidden Resolve(...)
       ├── principal
       └── approval
             │
             ▼
       Branchpoint gate
             │
             ▼
      durable execution
```

A model cannot grant itself permissions or set `approval=true` because those
parameters are not in the tool input schema.

## Human approval uses MCP itself

For low-risk authorized calls, the approval resolver returns immediately and no
extra round trip occurs.

For a high-risk or irreversible call, the resolver returns an MCP `Elicit`
request. On the current 2026-07-28 protocol, the SDK carries this as native
multi-round-trip input-required state. The tool body does not cross the
Branchpoint execution boundary until the host answers the elicitation.

The confirmation question includes the tool name, canonical risk,
reversibility, effect id, and reason code. Branchpoint does not copy tool
argument values into the default approval prompt.

The MCP request-state machinery binds the resumed call to the tool and argument
payload, while Branchpoint independently binds the effect id to the canonical
tool + arguments receipt. A changed semantic effect therefore cannot silently
reuse a completed receipt.

## Identity is application-owned

`create_branchpoint_mcp_server(...)` requires a
`principal_resolver(ctx)`.

Do **not** treat `ctx.headers` as authenticated identity. The MCP SDK itself
documents request headers as client-supplied input. A production resolver
should consume identity already verified by the hosting application, for
example:

- an authenticated OAuth/session context;
- verified JWT claims;
- mTLS identity;
- a trusted reverse-proxy/auth middleware result;
- a local stdio identity chosen by the parent application.

The resolver must return a Branchpoint `AuthorizationContext` or `None`.
Anything else fails closed.

## Durable execution

MCP execution requires a configured Branchpoint `ExecutionLedger`.

For local/single-process use:

```python
SQLiteExecutionLedger("./effects.sqlite3")
```

For coordinated multi-instance execution:

```python
PostgresExecutionLedger(os.environ["DATABASE_URL"])
```

The caller supplies one stable `effect_id` for one semantic side effect.
Exact replay returns the stored result. Reusing the same id for another tool or
different arguments fails closed.

The actual handler still runs through `ToolRegistry.execute_async(...)`, so
authorization is checked at consequence time and downstream idempotency keys are
propagated when declared by the ToolSpec.

## Request-state keys in deployed MCP servers

The factory defaults to the MCP SDK's ephemeral request-state security because
that is appropriate for a single-process/local server and for the zero-key
example.

For a multi-replica or restart-resilient MCP deployment, provide explicit
`request_state_security` configured so every replica that may resume a
multi-round-trip call uses compatible persistent keys and the same server
identity. Keep this separate from the Branchpoint execution ledger: MCP request
state protects the protocol round trip; Branchpoint receipts protect external
effect identity.

## Tested protocol path

CI installs the MCP extra separately from Branchpoint core and uses the real SDK
in-memory transport:

```text
Client(MCPServer)
      ↓
tools/list
      ↓
hidden principal absent from schema
      ↓
tools/call
      ↓
Resolve(principal)
      ↓
Branchpoint gate
      ↓
low risk ───────────────→ execute
high risk → Elicit(user) → retry/resume → execute
deny ───────────────────→ no receipt
      ↓
ToolRegistry.execute_async
      ↓
durable receipt
```

The tests also prove exact replay runs the external handler once, rejected human
approval creates no receipt, missing permission does not become a human
override, and one effect id cannot be rebound to changed arguments.
