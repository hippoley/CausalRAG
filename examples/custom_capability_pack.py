"""Register and run a capability pack without editing Branchpoint core.

Run:
    python examples/custom_capability_pack.py
"""

from branchpoint.benchmarks.hidden_world import (
    HiddenWorldEnvironment,
    HiddenWorldReasoner,
    HiddenWorldScenario,
)
from branchpoint.experiments import ExperimentContract, OutcomeLikelihood
from branchpoint.probe import (
    ProbeRunConfig,
    ProbeScenarioRuntime,
    ProbeScenarioSpec,
    available_probe_config,
    register_probe_scenario,
    run_probe_episode,
    unregister_probe_scenario,
)


PACK_ID = "queue_incident_demo"


def build_queue_scenario(hidden_hypothesis: str) -> HiddenWorldScenario:
    return HiddenWorldScenario(
        scenario_id="queue_incident_demo_v1",
        hypotheses={
            "H1": "The upstream API is rate-limiting requests.",
            "H2": "The worker pool is saturated.",
        },
        hidden_hypothesis=hidden_hypothesis,
        experiments={
            "inspect_http_status": ExperimentContract(
                experiment_id="queue_http_status",
                description="Inspect recent upstream HTTP status distribution.",
                outcomes=[
                    OutcomeLikelihood("many_429s", {"H1": 0.92, "H2": 0.10}),
                    OutcomeLikelihood("mostly_2xx", {"H1": 0.08, "H2": 0.90}),
                ],
            ),
            "inspect_worker_queue": ExperimentContract(
                experiment_id="queue_worker_depth",
                description="Inspect worker queue depth and execution lag.",
                outcomes=[
                    OutcomeLikelihood("saturated", {"H1": 0.08, "H2": 0.94}),
                    OutcomeLikelihood("healthy", {"H1": 0.92, "H2": 0.06}),
                ],
            ),
        },
        interventions={
            "backoff_upstream_calls": "H1",
            "scale_worker_pool": "H2",
        },
        experiment_costs={
            "inspect_http_status": 0.03,
            "inspect_worker_queue": 0.02,
        },
        intervention_costs={
            "backoff_upstream_calls": 0.08,
            "scale_worker_pool": 0.12,
        },
    )


def build_pack(config) -> ProbeScenarioRuntime:
    scenario = build_queue_scenario(config.hidden_hypothesis)
    environment = HiddenWorldEnvironment(
        scenario,
        outcome_mode=config.outcome_mode,
        seed=int(config.seed),
        outcome_coupling=config.stochastic_coupling,
    )
    return ProbeScenarioRuntime(
        scenario_id=PACK_ID,
        environment=environment,
        world_model=environment.world_model(),
        tools=environment.tools(),
        default_reasoner=HiddenWorldReasoner(
            scenario,
            confidence_threshold=float(config.confidence_threshold),
            max_probes=int(config.max_probes),
        ),
        goal=(
            "Diagnose why jobs are backing up and choose the intervention "
            "that matches the hidden mechanism."
        ),
    )


def register_pack():
    """Entry point that applications/CLI can call before starting the probe."""
    return register_probe_scenario(
        ProbeScenarioSpec(
            scenario_id=PACK_ID,
            label="Queue incident demo",
            description="An application-owned pack registered at runtime.",
            hidden_hypotheses=("H1", "H2"),
            outcome_modes=("deterministic", "stochastic"),
            recommended_test="External capability-pack registration",
            default_hidden_hypothesis="H2",
            default_outcome_mode="deterministic",
            default_goal=(
                "Diagnose why jobs are backing up and choose the intervention "
                "that matches the hidden mechanism."
            ),
            builder=build_pack,
        ),
        replace=True,
    )


def main():
    register_pack()

    try:
        ids = {row["id"] for row in available_probe_config()["scenarios"]}
        assert PACK_ID in ids

        episode = run_probe_episode(
            ProbeRunConfig(
                scenario=PACK_ID,
                hidden_hypothesis="H2",
                outcome_mode="deterministic",
                proposer_family="deterministic",
            )
        )

        print("registered:", PACK_ID in ids)
        print("success:", episode["metrics"]["success"])
        print("intervention:", episode["metrics"]["selected_intervention"])
        print("trajectory:")
        for decision in episode["decisions"]:
            selected = decision["selected"]
            print(
                f"  {decision['step']}: "
                f"{selected['kind']} · {selected['name']}"
            )
    finally:
        unregister_probe_scenario(PACK_ID)


if __name__ == "__main__":
    main()
