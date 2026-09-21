"""Show proposal order diverging from execution authority.

Run:
    python examples/browser_action_guard_demo.py
"""

from branchpoint.probe import ProbeRunConfig, run_probe_comparison


def main():
    report = run_probe_comparison(
        ProbeRunConfig(
            scenario="browser_action_guard",
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )

    vanilla = report["vanilla"]
    guarded = report["causal"]
    divergence = report["first_divergence"]

    print("same world:", report["comparison"]["same_world_inputs"])
    print("first divergence:", divergence["step"])
    print(
        "proposal-order loop:",
        vanilla["decisions"][0]["selected"]["name"],
        "-> success=",
        vanilla["metrics"]["success"],
    )
    print(
        "decision runtime:",
        guarded["decisions"][0]["selected"]["name"],
        "->",
        guarded["metrics"]["selected_intervention"],
        "-> success=",
        guarded["metrics"]["success"],
    )


if __name__ == "__main__":
    main()
