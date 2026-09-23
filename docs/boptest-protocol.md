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


## Isolating the runtime from the proposer

The first Branchpoint-vs-baseline experiment deliberately avoids changing the
proposal generator between arms.

`temperature_band_controller_specs` creates two controllers from one shared
BESTEST Air proposal provider:

- `proposal-order`: executes the first proposed control unchanged;
- `branchpoint`: sends the same ordered candidates through Branchpoint
  arbitration before returning BOPTEST controls.

The default BESTEST Air signal names are
`zon_reaTRooAir_y`, `con_oveTSetHea_u`, and `con_oveTSetCoo_u`.
The proposer is intentionally transparent: outside the configured temperature
band it proposes an explicit setpoint override first and leaving the embedded
controller in charge second. Branchpoint may preserve or reverse that order
based on the canonical cost/risk/reversibility metadata supplied with the
candidate.

Every step records controller metadata into the episode trajectory, including
the proposer-first action, selected action, whether Branchpoint changed the
order, and the runtime scores used for arbitration.

A live single-seed comparison can be run with:

```bash
python examples/boptest_branchpoint_vs_proposer.py \
  --seed 11 \
  --hours 6 \
  --output boptest_branchpoint_vs_proposer.json
```

This is an ablation of **decision arbitration**, not a claim that the shared
temperature-band proposer is an optimal controller. The next evidence step is
to predeclare several seeds and compare paired KPI deltas with the bootstrap
procedure above.


## Preregistered BESTEST Air regime study

The first external arbitration study is checked in before execution as
`examples/boptest_bestest_air_arbitration_study.json`.

It intentionally does **not** use forecast-uncertainty seeds as repeated
episodes. The temperature-band proposer only consumes the current zone
temperature, not BOPTEST forecasts, so changing forecast seeds would not create
a meaningful independent environment variation for this controller.

Instead, the plan evaluates the same two controller arms over five distinct
BESTEST Air operating regimes. The selected day is the center day of each
official two-week BOPTEST period:

| Regime | Official period | Study day |
| --- | --- | ---: |
| Peak heating | day 334–348 | 341 |
| Typical heating | day 44–58 | 51 |
| Peak cooling | day 282–296 | 289 |
| Typical cooling | day 146–160 | 153 |
| Mixed heating/cooling | day 14–28 | 21 |

Each paired episode uses a seven-day warmup, a 15-minute control step, and a
24-hour evaluation horizon. The proposal policy and runtime policy parameters
are frozen in the study JSON before the live run.

The checked-in plan currently represents a **high-risk override ablation**:
the shared proposer puts a temperature-setpoint override first when the room is
outside the comfort band, while Branchpoint is allowed to reject that proposal
using the canonical risk/cost metadata. This tests the effect of arbitration
under an explicit risk assumption; it is not evidence that the assigned risk
value is physically calibrated.

Run the frozen study locally:

```bash
python examples/boptest_branchpoint_regime_study.py \
  examples/boptest_bestest_air_arbitration_study.json \
  --output boptest_bestest_air_arbitration_study_result.json
```

Or manually run the GitHub Actions workflow `boptest-evidence`. It executes
the same checked-in plan against the public BOPTEST service and uploads the
complete JSON artifact.

The bootstrap interval over the five regimes is recorded as a descriptive
robustness summary only. These regimes are purposively selected benchmark
conditions rather than a random sample from a population, so the interval must
not be presented as an inferential population confidence interval.
