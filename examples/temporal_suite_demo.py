import json

from branchpoint.benchmarks import compare_temporal_suite


if __name__ == "__main__":
    print(json.dumps(compare_temporal_suite(), indent=2, sort_keys=True))
