"""Compare no-key HiddenWorld policies on the same stochastic episodes.

Run:
    python examples/hidden_world_policy_comparison.py
"""

import json

from causalrag.benchmarks import compare_hidden_world_policies


def main():
    report, metrics = compare_hidden_world_policies(seeds=range(10))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print("\nfailed episodes by policy:")
    for policy_id, rows in metrics.items():
        failed = [row for row in rows if not row.success]
        print(f"  {policy_id}: {len(failed)}/{len(rows)}")


if __name__ == "__main__":
    main()
