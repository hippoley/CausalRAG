# BOPTEST evidence protocol

Branchpoint's BOPTEST integration separates **connectivity** from **evidence**.

The existing live smoke proves that the project can talk to an independent
BOPTEST service. The experiment protocol adds a stricter requirement: every
controller comparison must be reproducible from a frozen manifest.

## Frozen manifest

A `BOPTESTScenarioManifest` records:

- testcase;
- explicit simulation start and warmup;
- communication step and evaluation horizon;
- scenario uncertainty / pricing settings and seed when used;
- the actuator surface a controller is allowed to touch;
- the measurement subset recorded in the artifact;
- KPI names that must exist before an episode is accepted;
- a protocol version and deterministic SHA-256 manifest hash.

V1 deliberately does not use BOPTEST's named `time_period` shortcut. Start,
warmup, and horizon remain explicit so a named testcase preset cannot silently
change the evaluation window.

## Baseline first

`no_op_controller` sends no overrides. BOPTEST's embedded/reference controller
remains in charge. Run it with:

```bash
python examples/boptest_protocol_run.py \
  examples/boptest_bestest_air_baseline.json \
  --output boptest-baseline.json
```

The output contains the exact manifest, manifest hash, service version, input
and measurement metadata, initial state, per-step controls/observations, and
final KPIs.

## Fair-comparison rule

A controller may return only controls declared in `controlled_inputs`.
Branchpoint rejects undeclared or unknown controls before they reach
`/advance`. A comparison should reuse the same manifest across controller
conditions.

This infrastructure does **not** establish that Branchpoint improves BOPTEST
outcomes. A performance claim requires repeated episodes with frozen manifests,
explicit baselines, retained failures, and statistical summaries over the
resulting KPI artifacts.


## Paired controller comparisons

Use `run_boptest_comparison` when comparing controllers. Every controller is
run in a fresh BOPTEST worker against the same manifest.

```python
from branchpoint.benchmarks import (
    BOPTESTControllerSpec,
    constant_controller,
    no_op_controller,
    run_boptest_comparison,
)

report = run_boptest_comparison(
    [manifest],
    [
        BOPTESTControllerSpec("embedded", no_op_controller),
        BOPTESTControllerSpec(
            "constant-heat",
            constant_controller({"oveHea_u": 0.25, "oveHea_activate": 1}),
        ),
    ],
    reference_controller_id="embedded",
)
```

The report stores each complete episode plus raw KPI deltas
(`controller - reference`). It does not label a controller better or worse,
because KPI direction and trade-offs belong to the experiment protocol.

By default the comparison also requires one BOPTEST service version across the
entire suite. If the public service changes version between arms or manifests,
the run is invalidated instead of silently mixing environments.

Aggregate output currently reports count, mean, minimum, and maximum paired
delta. That is descriptive evidence only. Confidence intervals and stronger
claims require a repeated seeded protocol defined before final tuning.


## Repeated seeded evidence

For uncertainty-aware comparisons, expand one base manifest into explicit seeded
manifests and bootstrap the **paired** KPI deltas.

```python
from branchpoint.benchmarks import (
    bootstrap_paired_kpi_intervals,
    expand_seeded_manifests,
)

manifests = expand_seeded_manifests(base_manifest, [11, 12, 13, 14, 15])
comparison = run_boptest_comparison(
    manifests,
    controllers,
    reference_controller_id="embedded",
)
uncertainty = bootstrap_paired_kpi_intervals(
    comparison,
    confidence=0.95,
    resamples=5000,
    seed=20260923,
)
```

The bootstrap resamples manifest-level paired deltas, not unpaired controller
outcomes. The artifact records the reference controller, manifest hashes,
confidence level, resample count, master seed, and a deterministic derived seed
for each controller/KPI series.

If fewer than the configured minimum number of pairs are available, the
interval is recorded as `insufficient_pairs` with no fabricated bounds.

The runtime deliberately does not convert an interval into a winner,
"significance" label, or product claim. KPI direction, multiple-comparison
policy, seed selection, and minimum sample size belong to the benchmark
protocol and should be frozen before final tuning.


## Seeded evidence and uncertainty

Use `expand_seeded_manifests` to create a predeclared set of otherwise-identical
manifests whose BOPTEST scenario seed differs. After running the paired
comparison, `bootstrap_paired_kpi_intervals` computes a manifest-level paired
percentile bootstrap interval over `controller - reference` KPI deltas.

```python
from branchpoint.benchmarks import (
    bootstrap_paired_kpi_intervals,
    expand_seeded_manifests,
)

manifests = expand_seeded_manifests(base_manifest, [11, 12, 13, 14, 15])
comparison = run_boptest_comparison(
    manifests,
    controllers,
    reference_controller_id="embedded",
)
uncertainty = bootstrap_paired_kpi_intervals(
    comparison,
    confidence=0.95,
    resamples=5000,
    seed=20260923,
)
```

The bootstrap artifact records the reference controller, exact manifest hashes,
master seed, per-controller/KPI derived bootstrap seed, sample count and interval.
If fewer than the configured minimum number of paired episodes exist, the
interval is omitted and marked `insufficient_pairs`.

This layer intentionally does not produce significance labels, rankings, or a
winner. KPI direction, multiplicity corrections, minimum sample size and claim
thresholds must be fixed by the study protocol before interpreting the results.
