"""No-key stochastic HiddenWorld suite demo.

Run:
    python examples/hidden_world_suite_demo.py
"""

import json

from causalrag.benchmarks import run_hidden_world_suite


def main():
    report, episodes = run_hidden_world_suite(seeds=range(10))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    failures = [row.to_dict() for row in episodes if not row.success]
    print(f"\nfailures={len(failures)}")
    for row in failures[:5]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
