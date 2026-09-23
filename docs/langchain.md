# LangChain / LangGraph execution middleware

Branchpoint can sit at LangChain's tool-execution boundary without replacing the
agent graph.

This integration is tested against the LangChain **1.4.x** API.

Until the first public Branchpoint package release, install from this repository:

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
pip install -e ".[langchain]"
```

## The boundary

LangChain continues to own:

- model calls and tool-call generation;
- agent and LangGraph state;
- checkpointing and resume;
- its native human-in-the-loop interrupt flow.

Branchpoint owns, immediately before an approved canonical tool executes:

- current-principal authorization;
- durable effect identity;
- downstream idempotency;
- execution receipt claim / replay / reconciliation;
- sync or async ToolSpec execution.

The LangChain tool call id becomes the durable identity for effectful tools:

```text
langchain:<tool-name>:<tool-call-id>
```

Reusing the same call id with changed arguments therefore fails with an effect
identity conflict instead of silently executing a different side effect.

## Recommended setup

Use one canonical `ToolSpec` implementation. Do not maintain a second
side-effect handler only for LangChain.

```python
from langchain.agents import create_agent

from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import (
    langchain_branchpoint_stack,
    langchain_tool_schema,
)


def charge_card(amount, idempotency_key):
    return external_processor.charge(
        amount=amount,
        idempotency_key=idempotency_key,
    )


charge = ToolSpec(
    "charge_card",
    "Charge a card.",
    charge_card,
    risk=0.8,
    reversible=False,
    require_durable_receipt=True,
    idempotency_key_argument="idempotency_key",
    required_permissions=("payments.charge",),
)

registry = ToolRegistry(
    [charge],
    execution_ledger=SQLiteExecutionLedger("./effects.sqlite3"),
)

tool = langchain_tool_schema(
    charge,
    args_schema={
        "type": "object",
        "properties": {"amount": {"type": "number"}},
        "required": ["amount"],
        "additionalProperties": False,
    },
)

principal = AuthorizationContext.from_permissions(
    "billing:alice",
    ["payments.charge"],
)

hitl, branchpoint = langchain_branchpoint_stack(
    registry,
    auto_approve_max_risk=0.1,
    authorization_context=principal,
)

agent = create_agent(
    model,
    tools=[tool],
    middleware=[hitl, branchpoint],
    checkpointer=checkpointer,
)
```

The middleware order is intentional. LangChain's native HITL layer is outermost,
so a high-risk call pauses before the Branchpoint execution middleware is
entered. After approval and graph resume, Branchpoint rechecks current authority
and only then crosses the durable execution boundary.

## Schema-only tools fail closed

`langchain_tool_schema(...)` creates a normal LangChain `StructuredTool` for
model discovery and argument validation, but its direct handler deliberately
raises.

That means removing or misconfiguring the Branchpoint execution middleware does
not turn a governed side effect into a raw LangChain tool execution path.

The real implementation remains the `ToolSpec.handler`.

## Human review policy

`langchain_human_in_the_loop(...)` converts canonical risk/reversibility into
LangChain's native `HumanInTheLoopMiddleware` config.

Low-risk reversible tools remain auto-approved. Tools above
`auto_approve_max_risk`, tools with invalid risk metadata, and irreversible
tools require review by default.

Branchpoint limits those generated review choices to:

```text
approve
reject
```

It does not enable `edit` for canonical side effects by default. Editing
arguments changes the semantic effect and should form a new execution intent,
not mutate an already-reviewed durable call identity.

LangChain HITL still requires the application's normal durable checkpointer and
resume flow. Branchpoint does not replace that state machine.

## Authorization is checked after resume

Human approval is not permission elevation.

An `authorization_resolver` can derive the current Branchpoint principal from
the LangGraph runtime context:

```python
def resolve_authority(runtime_context, tool_name, arguments, call_id):
    return runtime_context.current_principal
```

The resolver is called at execution time. If the principal changed, expired, or
lost a required permission while the graph was paused, execution is denied even
if a human previously approved the action.

## Receipt evidence stays outside model content

A successful Branchpoint tool call returns an ordinary LangChain `ToolMessage`.

The model sees the normal tool result in `ToolMessage.content`. Branchpoint
execution evidence is attached under:

```python
tool_message.artifact["branchpoint"]
```

The evidence includes the execution schema version, tool/call identity, durable
effect id, replay flag, receipt status/hash, and authorization policy id. It
does not include principal identity, tool arguments, or the raw result by
default.

## Unknown tools

The default is `unknown_tool_policy="deny"`.

If a model emits a tool that is not represented by a canonical Branchpoint
`ToolSpec`, the middleware fails closed instead of forwarding it to a
LangChain handler.

Applications that intentionally mix governed and ungoverned tools may opt into
`unknown_tool_policy="passthrough"`. That is an explicit application choice,
not the default.

## Sync and async

Both LangChain middleware hooks are implemented:

- `wrap_tool_call(...)` uses Branchpoint `ToolRegistry.execute(...)`;
- `awrap_tool_call(...)` uses `ToolRegistry.execute_async(...)`.

Async ToolSpec handlers are awaited directly. Synchronous handlers can also run
through the async path without blocking the event loop.

## What this does not claim

The adapter does not make every LangChain tool safe, and it does not infer
correct risk values for your application.

It provides a concrete place where model-generated tool intent is forced
through application-owned execution policy and durable effect accounting before
the real handler runs.
