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
