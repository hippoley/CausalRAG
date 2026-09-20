"""Live smoke test against the independent public BOPTEST service."""

from __future__ import annotations

import json
import time
from pathlib import Path

from branchpoint.environments import BOPTESTClient, BOPTESTProtocolError


result = {
    "base_url": "https://api.boptest.net",
    "testcase": "bestest_air",
}
client = BOPTESTClient(timeout=60.0)
result["service_version"] = client.version()
print("service_version=", json.dumps(result["service_version"], sort_keys=True))
result["testcases"] = client.testcases()
print("testcases_type=", type(result["testcases"]).__name__)

try:
    testid = client.select_testcase(result["testcase"])
    result["testid"] = testid
    print("testid=", testid)

    statuses = []
    for _ in range(30):
        status = client.status()
        statuses.append(status)
        print("status=", json.dumps(status, sort_keys=True, default=str))
        value = status.get("status") if isinstance(status, dict) else status
        if str(value).lower() == "running":
            break
        if str(value).lower() not in {"queued", "running"}:
            raise BOPTESTProtocolError(f"Unexpected BOPTEST worker status: {status!r}")
        time.sleep(2.0)
    else:
        raise BOPTESTProtocolError("BOPTEST testcase remained queued for too long")
    result["statuses"] = statuses

    measurements = client.measurements()
    inputs = client.inputs()
    result["measurement_count"] = len(measurements)
    result["input_count"] = len(inputs)
    result["measurement_names"] = sorted(measurements)[:25]
    result["input_names"] = sorted(inputs)[:25]
    assert measurements
    assert inputs

    initial = client.initialize(start_time=0, warmup_period=86400)
    result["initial_point_count"] = len(initial)
    step = client.get_step()
    result["control_step_seconds"] = step
    after = client.advance()
    result["advanced_point_count"] = len(after)
    result["advanced_time"] = after.get("time")
    assert after

    kpis = client.kpi()
    result["kpi"] = kpis
    print("kpi=", json.dumps(kpis, sort_keys=True))
    for key in ("ener_tot", "cost_tot", "tdis_tot", "idis_tot", "time_rat"):
        assert key in kpis, f"expected BOPTEST KPI missing: {key}"
finally:
    if client.testid:
        client.stop()
        result["stopped"] = True
    Path("boptest_live_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
