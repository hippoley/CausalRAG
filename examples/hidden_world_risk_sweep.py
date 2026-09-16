"""Sweep wrong-intervention loss on the same stochastic HiddenWorld episodes.

Run:
    python examples/hidden_world_risk_sweep.py
"""

import json

from causalrag.benchmarks import compare_risk_sensitivity


def main():
    losses = (0.0, 0.5, 1.0, 2.0)
    report, metrics = compare_risk_sensitivity(
        wrong_action_losses=losses,
        seeds=range(10),
    )

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print("\nfrontier summary:")
    for policy_id, policy_report in report.reports.items():
        print(
            f"  {policy_id}: "
            f"success={policy_report.success_rate:.3f} "
            f"brier={policy_report.mean_brier_score:.3f} "
            f"probes={policy_report.mean_probes:.2f} "
            f"cost={policy_report.mean_total_cost:.3f} "
            f"regret={policy_report.mean_causal_regret:.3f}"
        )

    print("\nfailed episodes by loss policy:")
    for policy_id, rows in metrics.items():
        failed = [row for row in rows if not row.success]
        print(f"  {policy_id}: {len(failed)}/{len(rows)}")


if __name__ == "__main__":
    main()
