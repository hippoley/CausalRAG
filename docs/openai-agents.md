# OpenAI Agents SDK approval adapter

Branchpoint can sit on the OpenAI Agents SDK pre-execution approval boundary
without replacing the SDK's agent loop.

This integration is tested against the `openai-agents 0.22.x` API. Until the first public PyPI release, install it from this repository as shown below.

```bash
git clone https://github.com/hippoley/CausalRAG.git
cd CausalRAG
pip install -e ".[openai-agents]"
```

## What it owns

The adapter decides whether a pending tool call can be automatically approved,
must be rejected, or must remain paused for a human.

The OpenAI Agents SDK still owns model calls, handoffs, run state, and execution
of approved function tools.

Branchpoint applies canonical application-owned metadata from `ToolSpec`:

- required principal permissions;
- risk;
- reversibility;
- whether the tool requires Branchpoint's durable execution boundary.

Model-supplied arguments cannot grant permissions or lower canonical risk.

## Minimal integration

```python
from agents import Agent, Runner, function_tool

from branchpoint import AuthorizationContext, ToolSpec
from branchpoint.integrations import OpenAIAgentsApprovalAdapter


adapter = OpenAIAgentsApprovalAdapter(
    [
        ToolSpec(
            "lookup_order",
            "Read an order",
            lambda order_id: None,
            risk=0.0,
            required_permissions=("order.read",),
        ),
        ToolSpec(
            "cancel_order",
            "Cancel an order",
            lambda order_id: None,
            risk=0.7,
            reversible=False,
            required_permissions=("order.cancel",),
        ),
    ],
    auto_approve_max_risk=0.1,
)


@function_tool(
    needs_approval=adapter.needs_approval("lookup_order")
)
def lookup_order(order_id: str) -> str:
    return f"order:{order_id}"


@function_tool(
    needs_approval=adapter.needs_approval("cancel_order")
)
def cancel_order(order_id: str) -> str:
    return f"cancelled:{order_id}"


agent = Agent(
    name="support",
    instructions="Help with orders.",
    tools=[lookup_order, cancel_order],
)

principal = AuthorizationContext.from_permissions(
    "support:alice",
    ["order.read", "order.cancel"],
)

result = await Runner.run(agent, "Cancel O-42 if appropriate.")
resolution = adapter.resolve(
    result,
    authorization_context=principal,
)

if resolution.pending:
    # Render these approval items in your own human-review UI.
    # Approve/reject them on resolution.state using the SDK RunState API.
    ...
else:
    # Branchpoint resolved every interruption as allow or deny.
    result = await Runner.run(agent, resolution.state)
```

For a low-risk authorized function tool, the `needs_approval` callback returns
false and the SDK can execute without pausing. Calls that Branchpoint would deny
or escalate return true, so execution pauses before the tool handler.

If the run does pause, `resolve(...)` re-evaluates current authority and:

- calls `RunState.approve()` for `allow`;
- calls `RunState.reject(..., rejection_message=...)` for `deny`;
- leaves `require_human` interruptions untouched.

The method does not automatically resume the run. The application keeps control
of persistence, human review, and when to call `Runner.run(agent, state)`.

## Fail-closed behavior

The adapter leaves the call paused when the tool name is missing, the tool is
not in the canonical Branchpoint registry, or arguments are missing, malformed,
non-object JSON, or contain non-standard JSON constants.

Missing/expired authority is a denial, not a human-risk escalation.

Irreversible tools and tools above `auto_approve_max_risk` require human
approval by default.

A ToolSpec with `require_durable_receipt=True` is denied through this
approval-only adapter. Letting the Agents SDK directly invoke that handler would
bypass Branchpoint's durable receipt/idempotency boundary. Use a Branchpoint
execution wrapper for those side effects instead.

## Nested agents and handoffs

The Agents SDK surfaces approval interruptions from nested agent tools and
handoffs on the outer run state. Pass those interruptions through the same
adapter; the canonical ToolSpec remains the policy source.

## Compatibility contract

CI installs the OpenAI Agents integration extra separately from Branchpoint
core, then verifies:

- the real SDK `ToolApprovalItem` exposes the tool/call/argument surface used
  by the adapter;
- `RunState.reject` supports an explicit rejection message;
- `function_tool` accepts the generated `needs_approval` callback.

This keeps SDK churn from silently breaking the adapter while leaving ordinary
Branchpoint installations framework-independent.


## Execute durable tools through Branchpoint

For side effects that require Branchpoint receipts, do not decorate the
application handler directly with the SDK. Bind the existing canonical
`ToolSpec` through `adapter.function_tool(...)` instead:

```python
from branchpoint import (
    AuthorizationContext,
    SQLiteExecutionLedger,
    ToolRegistry,
    ToolSpec,
)
from branchpoint.integrations import OpenAIAgentsApprovalAdapter


ledger = SQLiteExecutionLedger("branchpoint-effects.sqlite3")

def charge_card(amount, idempotency_key):
    return provider.charge(
        amount=amount,
        idempotency_key=idempotency_key,
    )

registry = ToolRegistry(
    [
        ToolSpec(
            "charge_card",
            "Charge a card",
            charge_card,
            risk=0.7,
            reversible=False,
            require_durable_receipt=True,
            idempotency_key_argument="idempotency_key",
            required_permissions=("payments.charge",),
        )
    ],
    execution_ledger=ledger,
)

principal = AuthorizationContext.from_permissions(
    "billing:alice",
    ["payments.charge"],
)

adapter = OpenAIAgentsApprovalAdapter(
    registry,
    auto_approve_max_risk=0.1,
    authorization_context=principal,
)

charge_tool = adapter.function_tool(
    "charge_card",
    params_json_schema={
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
        },
        "required": ["amount"],
        "additionalProperties": False,
    },
)
```

`charge_tool` is a real OpenAI Agents `FunctionTool`. The SDK still owns the
run and the approval interruption, but its `on_invoke_tool` crosses
`ToolRegistry.execute(...)` immediately before the application handler.

For a durable tool, the SDK's `tool_call_id` becomes part of a stable effect
identity:

```text
openai-agents:<qualified-tool-name>:<tool-call-id>
```

That means a resumed or replayed SDK invocation with the same call id returns
the stored Branchpoint receipt instead of performing the external side effect
again. If `idempotency_key_argument` is configured, the same effect id is also
propagated to the downstream provider.

Authorization is evaluated again at execution time. An approval that was valid
earlier does not freeze authority: if the principal loses permission before the
tool body runs, Branchpoint denies before claiming a receipt or invoking the
external handler.

Binding fails immediately if a durable ToolSpec has no execution ledger. The
first integration version also requires synchronous ToolSpec handlers; the SDK
wrapper moves them off the event loop with `asyncio.to_thread`.

A no-key executable example is included:

```bash
python examples/openai_agents_durable_tool.py
```

It invokes the same SDK FunctionTool twice with one call id and demonstrates
that the external handler runs once while the second invocation replays the
stored result.


## SDK-only receipt evidence

A Branchpoint-bound FunctionTool attaches execution metadata to the Agents SDK
tool output as `custom_data["branchpoint"]`:

```json
{
  "schema_version": "branchpoint.openai-agents.execution.v1",
  "execution_boundary": "branchpoint",
  "tool_name": "charge_card",
  "call_id": "call-...",
  "durable": true,
  "effect_id": "openai-agents:charge_card:call-...",
  "replayed": false,
  "receipt_status": "succeeded",
  "effect_hash": "...",
  "authorization_rechecked": true,
  "authorization_policy_id": "branchpoint.capability.v1"
}
```

On a replay of the same successful SDK call id, `replayed` becomes `true`.
The receipt result still comes from Branchpoint's ledger and the application
handler is not invoked again.

This metadata is SDK-only. The Agents SDK stores it on `ToolCallOutputItem`
rather than inside the raw tool result replayed to the model. Branchpoint's
compatibility test explicitly verifies that `to_input_item()` does not include
`custom_data`.

The audit payload intentionally omits tool arguments, tool output content, and
principal identity. Those remain application-owned data rather than default
trace content.
