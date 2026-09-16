"""Minimal v0.2 causal-agent example.

Set OPENAI_API_KEY, then run:
    python examples/agent_quickstart.py
"""

from causalrag import create_agent


documents = [
    "Opening a window increases air exchange and can lower indoor CO2.",
    "More occupants increase indoor CO2 when ventilation is unchanged.",
    "Outdoor particulate pollution can make opening windows undesirable.",
]

agent = create_agent(documents=documents)
result = agent.run(
    "Explain the most likely leverage points for reducing indoor CO2, and identify what evidence you would check before acting.",
    max_steps=6,
)

print(result.answer)
print("\n--- decision trace ---")
for decision in result.state.decisions:
    print(
        decision.step,
        decision.selected.kind.value,
        decision.selected.name,
        decision.rationale,
    )
