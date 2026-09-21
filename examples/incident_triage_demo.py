"""Choose diagnostic evidence before changing production.

Run:
    python examples/incident_triage_demo.py
"""

from branchpoint.probe import ProbeRunConfig, run_probe_episode


def main():
    run = run_probe_episode(
        ProbeRunConfig(
            scenario="incident_triage",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )

    print("success:", run["metrics"]["success"])
    print("intervention:", run["metrics"]["selected_intervention"])
    print("trajectory:")
    for decision in run["decisions"]:
        selected = decision["selected"]
        print(f"  {decision['step']}: {selected['kind']} · {selected['name']}")


if __name__ == "__main__":
    main()
