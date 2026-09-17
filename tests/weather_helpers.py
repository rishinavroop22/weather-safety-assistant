"""Builders for synthetic Open-Meteo payloads and mocked HTTP clients (no network)."""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

import httpx

from app.weather import Location, OpenMeteoClient

# Calm hourly values used when a test doesn't care about a variable (test data only).
CALM_HOURLY: dict[str, float | int] = {
    "temperature_2m": 27.0,
    "apparent_temperature": 28.0,
    "relative_humidity_2m": 60.0,
    "precipitation": 0.0,
    "precipitation_probability": 10.0,
    "weather_code": 1,
    "wind_speed_10m": 8.0,
    "wind_gusts_10m": 15.0,
    "uv_index": 5.0,
    "visibility": 20000.0,
}

SeriesSpec = Callable[[datetime], Any] | Any


def make_location(**overrides: Any) -> Location:
    fields = {
        "name": "Testville",
        "admin1": "Test State",
        "country": "Testland",
        "country_code": "TL",
        "latitude": 23.25,
        "longitude": 77.41,
        "timezone": "Asia/Kolkata",
    }
    fields.update(overrides)
    return Location(**fields)


def make_geocoding_payload(*results: dict[str, Any]) -> dict[str, Any]:
    if not results:
        results = (
            {
                "id": 1,
                "name": "Testville",
                "latitude": 23.25,
                "longitude": 77.41,
                "timezone": "Asia/Kolkata",
                "admin1": "Test State",
                "country": "Testland",
                "country_code": "TL",
            },
        )
    return {"results": list(results), "generationtime_ms": 0.5}


def make_forecast_payload(
    *,
    current_time: str = "2026-09-17T14:30",
    first_day: str = "2026-09-17",
    days: int = 2,
    timezone: str = "Asia/Kolkata",
    variables: Sequence[str] = tuple(CALM_HOURLY),
    series: dict[str, SeriesSpec] | None = None,
) -> dict[str, Any]:
    """Hourly payload shaped like Open-Meteo's. ``series[var]`` is a constant or f(hour) -> value."""
    start = datetime.fromisoformat(f"{first_day}T00:00")
    times = [start + timedelta(hours=i) for i in range(24 * days)]
    specs = {**CALM_HOURLY, **(series or {})}

    hourly: dict[str, Any] = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for variable in variables:
        spec = specs[variable]
        hourly[variable] = [spec(t) if callable(spec) else spec for t in times]

    return {
        "latitude": 23.25,
        "longitude": 77.375,
        "generationtime_ms": 0.1,
        "utc_offset_seconds": 19800,
        "timezone": timezone,
        "timezone_abbreviation": "GMT+5:30",
        "elevation": 500.0,
        "current_units": {"time": "iso8601", "interval": "seconds"},
        "current": {"time": current_time, "interval": 900},
        "hourly_units": {"time": "iso8601"},
        "hourly": hourly,
    }


def mock_client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> OpenMeteoClient:
    """An OpenMeteoClient whose HTTP calls go to ``handler`` instead of the network."""
    return OpenMeteoClient(httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def json_handler(payload: Any, status: int = 200, seen: list[httpx.Request] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=payload)

    return handler


def raising_handler(exc_type: type[httpx.TransportError], seen: list[httpx.Request] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        raise exc_type("simulated failure", request=request)

    return handler
