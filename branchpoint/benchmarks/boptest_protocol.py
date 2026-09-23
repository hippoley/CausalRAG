from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from branchpoint.environments import BOPTESTClient, BOPTESTProtocolError


PROTOCOL_VERSION = "branchpoint.boptest.v1"


class BOPTESTExperimentError(RuntimeError):
    """Base error for reproducible BOPTEST experiment execution."""


class BOPTESTManifestError(BOPTESTExperimentError):
    """The frozen experiment manifest is internally inconsistent."""


class BOPTESTControlError(BOPTESTExperimentError):
    """A controller attempted to use a control outside the frozen surface."""


@dataclass(frozen=True)
class BOPTESTScenarioManifest:
    """Frozen external-environment protocol for one BOPTEST episode."""

    testcase: str
    start_time: float = 0.0
    warmup_period: float = 86400.0
    step_seconds: float = 300.0
    horizon_seconds: float = 3600.0
    electricity_price: Optional[str] = None
    temperature_uncertainty: Optional[str] = None
    solar_uncertainty: Optional[str] = None
    seed: Optional[int] = None
    controlled_inputs: Tuple[str, ...] = ()
    measurement_points: Tuple[str, ...] = ()
    required_kpis: Tuple[str, ...] = ()
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        testcase = str(self.testcase).strip()
        if not testcase:
            raise BOPTESTManifestError("testcase must be non-empty")
        object.__setattr__(self, "testcase", testcase)

        for name, value in (
            ("start_time", self.start_time),
            ("warmup_period", self.warmup_period),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise BOPTESTManifestError(f"{name} must be finite and non-negative")

        for name, value in (
            ("step_seconds", self.step_seconds),
            ("horizon_seconds", self.horizon_seconds),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise BOPTESTManifestError(f"{name} must be finite and positive")

        steps = float(self.horizon_seconds) / float(self.step_seconds)
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
            raise BOPTESTManifestError(
                "horizon_seconds must be an integer multiple of step_seconds"
            )

        for field_name in (
            "controlled_inputs",
            "measurement_points",
            "required_kpis",
        ):
            raw = getattr(self, field_name)
            normalized = tuple(str(item).strip() for item in raw if str(item).strip())
            if len(set(normalized)) != len(normalized):
                raise BOPTESTManifestError(f"{field_name} contains duplicate names")
            object.__setattr__(self, field_name, normalized)

        if str(self.protocol_version) != PROTOCOL_VERSION:
            raise BOPTESTManifestError(
                f"unsupported protocol_version {self.protocol_version!r}"
            )

    @property
    def steps(self) -> int:
        return int(round(float(self.horizon_seconds) / float(self.step_seconds)))

    def scenario_settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {}
        if self.electricity_price is not None:
            settings["electricity_price"] = str(self.electricity_price)
        if self.temperature_uncertainty is not None:
            settings["temperature_uncertainty"] = str(self.temperature_uncertainty)
        if self.solar_uncertainty is not None:
            settings["solar_uncertainty"] = str(self.solar_uncertainty)
        if self.seed is not None:
            settings["seed"] = int(self.seed)
        return settings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "testcase": self.testcase,
            "start_time": float(self.start_time),
            "warmup_period": float(self.warmup_period),
            "step_seconds": float(self.step_seconds),
            "horizon_seconds": float(self.horizon_seconds),
            "steps": self.steps,
            "scenario": self.scenario_settings(),
            "controlled_inputs": list(self.controlled_inputs),
            "measurement_points": list(self.measurement_points),
            "required_kpis": list(self.required_kpis),
        }

    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BOPTESTControlContext:
    step_index: int
    elapsed_seconds: float
    observation: Mapping[str, Any]
    input_metadata: Mapping[str, Any]
    measurement_metadata: Mapping[str, Any]
    manifest: BOPTESTScenarioManifest


Controller = Callable[[BOPTESTControlContext], Mapping[str, Any]]


@dataclass(frozen=True)
class BOPTESTStepRecord:
    step_index: int
    elapsed_seconds: float
    controls: Mapping[str, Any]
    observation: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "elapsed_seconds": float(self.elapsed_seconds),
            "controls": dict(self.controls),
            "observation": dict(self.observation),
        }


@dataclass(frozen=True)
class BOPTESTEpisodeResult:
    manifest: BOPTESTScenarioManifest
    controller_id: str
    service_version: Any
    testcase_name: Any
    input_metadata: Mapping[str, Any]
    measurement_metadata: Mapping[str, Any]
    initial_observation: Mapping[str, Any]
    trajectory: Tuple[BOPTESTStepRecord, ...]
    kpis: Mapping[str, float]
    scenario_state: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.manifest_hash,
            "controller_id": self.controller_id,
            "service_version": self.service_version,
            "testcase_name": self.testcase_name,
            "input_metadata": dict(self.input_metadata),
            "measurement_metadata": dict(self.measurement_metadata),
            "initial_observation": dict(self.initial_observation),
            "trajectory": [row.to_dict() for row in self.trajectory],
            "kpis": dict(self.kpis),
            "scenario_state": dict(self.scenario_state),
        }


def no_op_controller(_context: BOPTESTControlContext) -> Mapping[str, Any]:
    """Leave the embedded/reference controller in charge for this step."""
    return {}


def _wait_until_running(
    client: BOPTESTClient,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        status = client.status()
        value = status.get("status") if isinstance(status, Mapping) else status
        normalized = str(value).strip().lower()
        if normalized == "running":
            return
        if normalized != "queued":
            raise BOPTESTProtocolError(f"Unexpected BOPTEST worker status: {status!r}")
        if time.monotonic() >= deadline:
            raise BOPTESTProtocolError(
                f"BOPTEST testcase remained queued for more than {timeout_seconds:g}s"
            )
        sleep(float(poll_seconds))


def _known_control_keys(input_metadata: Mapping[str, Any]) -> set[str]:
    keys = {str(name) for name in input_metadata}
    for name in tuple(keys):
        if name.endswith("_u"):
            keys.add(name[:-2] + "_activate")
    return keys


def _validate_control_surface(
    manifest: BOPTESTScenarioManifest,
    input_metadata: Mapping[str, Any],
) -> None:
    known = _known_control_keys(input_metadata)
    missing = [name for name in manifest.controlled_inputs if name not in known]
    if missing:
        raise BOPTESTManifestError(
            "manifest controlled_inputs are not exposed by BOPTEST: "
            + ", ".join(sorted(missing))
        )


def _validate_controls(
    controls: Mapping[str, Any],
    *,
    manifest: BOPTESTScenarioManifest,
    input_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {str(key): value for key, value in dict(controls).items()}
    known = _known_control_keys(input_metadata)
    unknown = sorted(set(result) - known)
    if unknown:
        raise BOPTESTControlError(
            "controller returned unknown BOPTEST control(s): " + ", ".join(unknown)
        )

    allowed = set(manifest.controlled_inputs)
    for name in manifest.controlled_inputs:
        if name.endswith("_u"):
            allowed.add(name[:-2] + "_activate")
    outside_manifest = sorted(set(result) - allowed)
    if outside_manifest:
        raise BOPTESTControlError(
            "controller escaped the frozen controlled_inputs surface: "
            + ", ".join(outside_manifest)
        )

    for key, value in result.items():
        if key.endswith("_activate") and value not in {0, 1, 0.0, 1.0, False, True}:
            raise BOPTESTControlError(f"{key} must be 0/1 when supplied")

    return result


def _select_observation(
    observation: Mapping[str, Any],
    measurement_points: Sequence[str],
) -> Dict[str, Any]:
    data = dict(observation)
    if not measurement_points:
        return data
    missing = [name for name in measurement_points if name not in data]
    if missing:
        raise BOPTESTExperimentError(
            "requested measurement point(s) missing from observation: "
            + ", ".join(sorted(missing))
        )
    selected = {"time": data["time"]} if "time" in data else {}
    selected.update({name: data[name] for name in measurement_points})
    return selected


def run_boptest_episode(
    manifest: BOPTESTScenarioManifest,
    controller: Controller = no_op_controller,
    *,
    controller_id: str = "embedded-baseline",
    client: Optional[BOPTESTClient] = None,
    queue_timeout_seconds: float = 120.0,
    poll_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> BOPTESTEpisodeResult:
    """Run one frozen BOPTEST episode and return a replayable result artifact."""

    controller_id = str(controller_id).strip()
    if not controller_id:
        raise ValueError("controller_id must be non-empty")
    if float(queue_timeout_seconds) <= 0 or float(poll_seconds) <= 0:
        raise ValueError("queue_timeout_seconds and poll_seconds must be positive")

    client = client or BOPTESTClient()
    allocated = False

    try:
        service_version = client.version()
        client.select_testcase(manifest.testcase)
        allocated = True
        _wait_until_running(
            client,
            timeout_seconds=queue_timeout_seconds,
            poll_seconds=poll_seconds,
            sleep=sleep,
        )

        testcase_name = client.name()
        input_metadata = client.inputs()
        measurement_metadata = client.measurements()
        _validate_control_surface(manifest, input_metadata)

        client.set_step(manifest.step_seconds)
        scenario_settings = manifest.scenario_settings()
        scenario_state: Mapping[str, Any] = {}
        if scenario_settings:
            scenario_state = client.set_scenario(**scenario_settings)

        initial = client.initialize(
            start_time=manifest.start_time,
            warmup_period=manifest.warmup_period,
        )
        observation = _select_observation(initial, manifest.measurement_points)

        trajectory = []
        for step_index in range(manifest.steps):
            context = BOPTESTControlContext(
                step_index=step_index,
                elapsed_seconds=step_index * float(manifest.step_seconds),
                observation=observation,
                input_metadata=input_metadata,
                measurement_metadata=measurement_metadata,
                manifest=manifest,
            )
            proposed = controller(context)
            if not isinstance(proposed, Mapping):
                raise BOPTESTControlError("controller must return a mapping of controls")
            controls = _validate_controls(
                proposed,
                manifest=manifest,
                input_metadata=input_metadata,
            )
            advanced = client.advance(controls)
            observation = _select_observation(advanced, manifest.measurement_points)
            trajectory.append(
                BOPTESTStepRecord(
                    step_index=step_index,
                    elapsed_seconds=(step_index + 1) * float(manifest.step_seconds),
                    controls=controls,
                    observation=observation,
                )
            )

        kpis = client.kpi()
        missing_kpis = sorted(set(manifest.required_kpis) - set(kpis))
        if missing_kpis:
            raise BOPTESTExperimentError(
                "required KPI(s) missing from BOPTEST result: "
                + ", ".join(missing_kpis)
            )

        return BOPTESTEpisodeResult(
            manifest=manifest,
            controller_id=controller_id,
            service_version=service_version,
            testcase_name=testcase_name,
            input_metadata=input_metadata,
            measurement_metadata=measurement_metadata,
            initial_observation=initial,
            trajectory=tuple(trajectory),
            kpis=kpis,
            scenario_state=scenario_state,
        )
    finally:
        if allocated and client.testid:
            client.stop()
