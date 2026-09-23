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
