from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
from urllib import error, parse, request


class BOPTESTError(RuntimeError):
    """Base error for BOPTEST environment interactions."""


class BOPTESTHTTPError(BOPTESTError):
    """Network/HTTP failure while calling BOPTEST."""


class BOPTESTProtocolError(BOPTESTError):
    """BOPTEST returned a malformed or unsuccessful application response."""


Transport = Callable[[str, str, Optional[Mapping[str, Any]], float], Any]


def _urllib_transport(
    method: str,
    url: str,
    payload: Optional[Mapping[str, Any]],
    timeout: float,
) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(dict(payload)).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise BOPTESTHTTPError(
            f"BOPTEST HTTP {exc.code} for {method.upper()} {url}: {details}"
        ) from exc
    except error.URLError as exc:
        raise BOPTESTHTTPError(
            f"BOPTEST connection failed for {method.upper()} {url}: {exc.reason}"
        ) from exc
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BOPTESTProtocolError(
            f"BOPTEST returned non-JSON response for {method.upper()} {url}"
        ) from exc


@dataclass(frozen=True)
class BOPTESTResponse:
    payload: Any
    status: Optional[int] = None
    message: str = ""


class BOPTESTClient:
    """Dependency-free client for the official BOPTEST REST API.

    Supports both the public service (``https://api.boptest.net``) and local
    deployments. A custom transport can be injected for deterministic tests,
    record/replay, or enterprise networking layers.
    """

    def __init__(
        self,
        base_url: str = "https://api.boptest.net",
        *,
        timeout: float = 30.0,
        transport: Optional[Transport] = None,
        testid: Optional[str] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must be non-empty")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.transport = transport or _urllib_transport
        self.testid = str(testid) if testid else None

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> BOPTESTResponse:
        url = f"{self.base_url}/{str(path).lstrip('/')}"
        raw = self.transport(method.upper(), url, payload, self.timeout)

        # Service-level responses such as {"status": "Running"} are direct
        # domain objects. Testcase operation envelopes are identified by the
        # documented `payload` field and then validate their numeric HTTP-like
        # application status. This prevents conflating worker state with the
        # envelope's success code.
        if not isinstance(raw, Mapping):
            return BOPTESTResponse(payload=raw)
        if "payload" not in raw:
            return BOPTESTResponse(payload=dict(raw))

        status = raw.get("status")
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            raise BOPTESTProtocolError(
                f"BOPTEST response envelope has invalid status: {status!r}"
            )
        message = str(raw.get("message") or "")
        if status_int != 200:
            raise BOPTESTProtocolError(
                f"BOPTEST application error status={status_int}: {message}"
            )
        return BOPTESTResponse(
            payload=raw.get("payload"),
            status=status_int,
            message=message,
        )

    def _instance_path(self, operation: str) -> str:
        if not self.testid:
            raise BOPTESTProtocolError(
                "No active BOPTEST testcase. Call select_testcase() first or pass testid."
            )
        return f"{operation}/{parse.quote(str(self.testid), safe='')}"

    def version(self) -> Any:
        return self._request("GET", "version").payload

    def testcases(self) -> Any:
        return self._request("GET", "testcases").payload

    def select_testcase(self, testcase_name: str) -> str:
        name = str(testcase_name).strip()
        if not name:
            raise ValueError("testcase_name must be non-empty")
        response = self._request(
            "POST",
            f"testcases/{parse.quote(name, safe='')}/select",
        )
        data = response.payload
        if not isinstance(data, Mapping) or not data.get("testid"):
            raise BOPTESTProtocolError("BOPTEST testcase selection returned no testid")
        self.testid = str(data["testid"])
        return self.testid

    def status(self) -> Any:
        return self._request("GET", self._instance_path("status")).payload

    def stop(self) -> Any:
        response = self._request("PUT", self._instance_path("stop"))
        self.testid = None
        return response.payload

    def name(self) -> Any:
        return self._request("GET", self._instance_path("name")).payload

    def measurements(self) -> Dict[str, Any]:
        payload = self._request("GET", self._instance_path("measurements")).payload
        return dict(payload or {})

    def inputs(self) -> Dict[str, Any]:
        payload = self._request("GET", self._instance_path("inputs")).payload
        return dict(payload or {})

    def forecast_points(self) -> Dict[str, Any]:
        payload = self._request("GET", self._instance_path("forecast_points")).payload
        return dict(payload or {})

    def get_step(self) -> float:
        payload = self._request("GET", self._instance_path("step")).payload
        try:
            return float(payload)
        except (TypeError, ValueError) as exc:
            raise BOPTESTProtocolError(f"Invalid BOPTEST step: {payload!r}") from exc

    def set_step(self, seconds: float) -> float:
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("step must be positive")
        payload = self._request(
            "PUT",
            self._instance_path("step"),
            {"step": seconds},
        ).payload
        try:
            return float(payload)
        except (TypeError, ValueError) as exc:
            raise BOPTESTProtocolError(f"Invalid BOPTEST step response: {payload!r}") from exc

    def initialize(self, start_time: float, warmup_period: float) -> Dict[str, Any]:
        if float(start_time) < 0 or float(warmup_period) < 0:
            raise ValueError("start_time and warmup_period must be non-negative")
        payload = self._request(
            "PUT",
            self._instance_path("initialize"),
            {
                "start_time": float(start_time),
                "warmup_period": float(warmup_period),
            },
        ).payload
        return dict(payload or {})

    def get_scenario(self) -> Dict[str, Any]:
        payload = self._request("GET", self._instance_path("scenario")).payload
        return dict(payload or {})

    def set_scenario(
        self,
        *,
        electricity_price: Optional[str] = None,
        time_period: Optional[str] = None,
        temperature_uncertainty: Optional[str] = None,
        solar_uncertainty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if electricity_price is not None:
            payload["electricity_price"] = str(electricity_price)
        if time_period is not None:
            payload["time_period"] = str(time_period)
        if temperature_uncertainty is not None:
            payload["temperature_uncertainty"] = str(temperature_uncertainty)
        if solar_uncertainty is not None:
            payload["solar_uncertainty"] = str(solar_uncertainty)
        if seed is not None:
            payload["seed"] = int(seed)
        if not payload:
            raise ValueError("set_scenario requires at least one setting")
        result = self._request("PUT", self._instance_path("scenario"), payload).payload
        return dict(result or {})

    def forecast(
        self,
        point_names: Iterable[str],
        *,
        horizon: float,
        interval: float,
    ) -> Dict[str, Any]:
        names = [str(name) for name in point_names if str(name).strip()]
        if not names:
            raise ValueError("point_names must be non-empty")
        if float(horizon) <= 0 or float(interval) <= 0:
            raise ValueError("horizon and interval must be positive")
        payload = self._request(
            "PUT",
            self._instance_path("forecast"),
            {
                "point_names": names,
                "horizon": float(horizon),
                "interval": float(interval),
            },
        ).payload
        return dict(payload or {})

    def advance(self, controls: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload = self._request(
            "POST",
            self._instance_path("advance"),
            dict(controls or {}),
        ).payload
        return dict(payload or {})

    def results(
        self,
        point_names: Iterable[str],
        *,
        start_time: float,
        final_time: float,
    ) -> Dict[str, Any]:
        names = [str(name) for name in point_names if str(name).strip()]
        if not names:
            raise ValueError("point_names must be non-empty")
        if float(final_time) < float(start_time):
            raise ValueError("final_time must be >= start_time")
        payload = self._request(
            "PUT",
            self._instance_path("results"),
            {
                "point_names": names,
                "start_time": float(start_time),
                "final_time": float(final_time),
            },
        ).payload
        return dict(payload or {})

    def kpi(self) -> Dict[str, float]:
        payload = self._request("GET", self._instance_path("kpi")).payload
        if not isinstance(payload, Mapping):
            raise BOPTESTProtocolError("BOPTEST KPI response must be an object")
        result: Dict[str, float] = {}
        for key, value in payload.items():
            if value is None:
                continue
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise BOPTESTProtocolError(
                    f"Invalid KPI value for {key}: {value!r}"
                ) from exc
        return result

    def __enter__(self) -> "BOPTESTClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.testid:
            try:
                self.stop()
            except BOPTESTError:
                if exc is None:
                    raise
