import pytest

from causalrag.environments import BOPTESTClient, BOPTESTProtocolError


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.responses = {}

    def add(self, method, suffix, response):
        self.responses[(method.upper(), suffix)] = response

    def __call__(self, method, url, payload, timeout):
        self.calls.append((method, url, payload, timeout))
        for (expected_method, suffix), response in self.responses.items():
            if method == expected_method and url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected BOPTEST call: {method} {url} {payload}")


def envelope(payload, status=200, message=""):
    return {"status": status, "message": message, "payload": payload}


def test_full_boptest_control_surface_and_kpis_are_mapped_to_documented_endpoints():
    transport = FakeTransport()
    transport.add("GET", "/version", {"version": "0.9.0"})
    transport.add("GET", "/testcases", ["bestest_air", "bestest_hydronic"])
    transport.add("POST", "/testcases/bestest_air/select", {"testid": "abc-123"})
    transport.add("GET", "/status/abc-123", {"status": "Running"})
    transport.add("GET", "/name/abc-123", envelope({"name": "BESTEST Air"}))
    transport.add("GET", "/measurements/abc-123", envelope({"TRooAir_y": {"Unit": "K"}}))
    transport.add("GET", "/inputs/abc-123", envelope({"oveHea_u": {"Unit": "1"}}))
    transport.add("GET", "/forecast_points/abc-123", envelope({"TDryBul": {"Unit": "K"}}))
    transport.add("GET", "/step/abc-123", envelope(300.0))
    transport.add("PUT", "/step/abc-123", envelope("600"))
    transport.add("PUT", "/initialize/abc-123", envelope({"time": 0.0, "TRooAir_y": 294.0}))
    transport.add("GET", "/scenario/abc-123", envelope({"electricity_price": "constant", "time_period": None}))
    transport.add("PUT", "/scenario/abc-123", envelope({"seed": 7, "temperature_uncertainty": "medium"}))
    transport.add("PUT", "/forecast/abc-123", envelope({"time": [0, 300], "TDryBul": [280, 281]}))
    transport.add("POST", "/advance/abc-123", envelope({"time": 600.0, "TRooAir_y": 295.0}))
    transport.add("PUT", "/results/abc-123", envelope({"time": [0, 600], "TRooAir_y": [294, 295]}))
    transport.add("GET", "/kpi/abc-123", envelope({"ener_tot": 1.2, "tdis_tot": 0.4, "cost_tot": 0.3}))
    transport.add("PUT", "/stop/abc-123", envelope({"stopped": True}))

    client = BOPTESTClient(base_url="https://example.invalid", transport=transport)
    assert client.version() == {"version": "0.9.0"}
    assert client.testcases() == ["bestest_air", "bestest_hydronic"]
    assert client.select_testcase("bestest_air") == "abc-123"
    assert client.status() == {"status": "Running"}
    assert client.name() == {"name": "BESTEST Air"}
    assert "TRooAir_y" in client.measurements()
    assert "oveHea_u" in client.inputs()
    assert "TDryBul" in client.forecast_points()
    assert client.get_step() == pytest.approx(300.0)
    assert client.set_step(600) == pytest.approx(600.0)
    assert client.initialize(0, 86400)["TRooAir_y"] == pytest.approx(294.0)
    assert client.get_scenario()["electricity_price"] == "constant"
    assert client.set_scenario(temperature_uncertainty="medium", seed=7)["seed"] == 7
    assert client.forecast(["TDryBul"], horizon=600, interval=300)["TDryBul"] == [280, 281]
    assert client.advance({"oveHea_u": 0.5, "oveHea_activate": 1})["time"] == pytest.approx(600.0)
    assert client.results(["TRooAir_y"], start_time=0, final_time=600)["TRooAir_y"] == [294, 295]
    assert client.kpi() == {"ener_tot": 1.2, "tdis_tot": 0.4, "cost_tot": 0.3}
    assert client.stop() == {"stopped": True}
    assert client.testid is None

    advance = [call for call in transport.calls if call[0] == "POST" and "/advance/" in call[1]][0]
    assert advance[2] == {"oveHea_u": 0.5, "oveHea_activate": 1}


def test_instance_calls_require_selected_testcase_but_service_calls_do_not():
    transport = FakeTransport()
    transport.add("GET", "/version", {"version": "0.9.0"})
    client = BOPTESTClient(transport=transport)
    assert client.version() == {"version": "0.9.0"}
    with pytest.raises(BOPTESTProtocolError, match="No active BOPTEST testcase"):
        client.measurements()


def test_application_error_is_not_silently_returned_as_payload():
    transport = FakeTransport()
    transport.add("GET", "/kpi/bad", envelope({}, status=400, message="bad input"))
    client = BOPTESTClient(transport=transport, testid="bad")
    with pytest.raises(BOPTESTProtocolError, match="status=400"):
        client.kpi()


def test_selection_requires_testid_and_scenario_requires_explicit_setting():
    transport = FakeTransport()
    transport.add("POST", "/testcases/demo/select", {})
    client = BOPTESTClient(transport=transport)
    with pytest.raises(BOPTESTProtocolError, match="no testid"):
        client.select_testcase("demo")

    client.testid = "abc"
    with pytest.raises(ValueError, match="at least one setting"):
        client.set_scenario()


def test_context_manager_stops_allocated_worker():
    transport = FakeTransport()
    transport.add("POST", "/testcases/demo/select", {"testid": "ctx"})
    transport.add("PUT", "/stop/ctx", envelope({"stopped": True}))

    with BOPTESTClient(transport=transport) as client:
        client.select_testcase("demo")
        assert client.testid == "ctx"
    assert client.testid is None
