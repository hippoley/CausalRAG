"""Route an ambiguous request through an explicit decision runtime.

Run:
    python examples/tool_routing_demo.py
"""

from branchpoint.probe import ProbeRunConfig, run_probe_episode


def main():
    run = run_probe_episode(
        ProbeRunConfig(
            scenario="tool_routing",
            hidden_hypothesis="H3",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )

    print("success:", run["metrics"]["success"])
    print("route:", run["metrics"]["selected_intervention"])
    print("trajectory:")
    for decision in run["decisions"]:
        selected = decision["selected"]
        print(f"  {decision['step']}: {selected['kind']} · {selected['name']}")


if __name__ == "__main__":
    main()
