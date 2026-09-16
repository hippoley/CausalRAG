"""No-key HiddenWorld benchmark demo.

Run:
    python examples/hidden_world_demo.py
"""

from causalrag.benchmarks import build_hvac_hidden_world, run_hidden_world


def main():
    for hidden in ("H1", "H2", "H3"):
        metrics, result = run_hidden_world(build_hvac_hidden_world(hidden))
        print(f"\n=== hidden={hidden} ===")
        print(result.answer)
        print("metrics:", metrics.to_dict())
        print("trace:")
        for decision in result.state.decisions:
            selected_score = next(
                (
                    score
                    for score in decision.action_scores
                    if score.action_name == decision.selected.name
                ),
                None,
            )
            info = (
                f" info={selected_score.information_gain:.3f}"
                f" source={selected_score.information_source}"
                if selected_score is not None
                else ""
            )
            print(
                f"  step={decision.step} action={decision.selected.name}{info}"
            )


if __name__ == "__main__":
    main()
