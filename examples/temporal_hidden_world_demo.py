import json

from causalrag.benchmarks import compare_temporal_guard


if __name__ == "__main__":
    report = compare_temporal_guard()
    print(json.dumps(report, indent=2, sort_keys=True))
