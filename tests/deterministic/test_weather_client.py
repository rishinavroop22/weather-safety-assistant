"""Open-Meteo client: geocoding and forecast over mocked HTTP (no network)."""

import json
from typing import Any

import httpx
import pytest

from app.policy import required_weather_variables
from app.weather import FixtureWeatherProvider, LocationError, WeatherError
from tests.weather_helpers import (
    CALM_HOURLY,
    json_handler,
    make_forecast_payload,
    make_geocoding_payload,
    make_location,
    mock_client,
    raising_handler,
)

VARIABLES = sorted(CALM_HOURLY)

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------


def test_geocode_success_returns_structured_location():
    seen: list[httpx.Request] = []
    client = mock_client(json_handler(make_geocoding_payload(), seen=seen))

    location = client.geocode("  Testville ")

    assert location.name == "Testville"
    assert location.admin1 == "Test State"
    assert location.country == "Testland"
    assert (location.latitude, location.longitude) == (23.25, 77.41)
    assert location.timezone == "Asia/Kolkata"
    assert location.display_name == "Testville, Test State, Testland"
    assert seen[0].url.host == "geocoding-api.open-meteo.com"
    assert seen[0].url.params["name"] == "Testville"


def test_geocode_uses_first_valid_result():
    missing_timezone = {"name": "Broken", "latitude": 1.0, "longitude": 2.0}
    valid = {"name": "Second", "latitude": 10.0, "longitude": 20.0, "timezone": "Europe/Berlin"}
    other = {"name": "Third", "latitude": 30.0, "longitude": 40.0, "timezone": "Asia/Tokyo"}
    client = mock_client(json_handler(make_geocoding_payload(missing_timezone, valid, other)))

    assert client.geocode("x").name == "Second"


@pytest.mark.parametrize("payload", [{"generationtime_ms": 0.2}, {"results": []}], ids=["no-results-key", "empty-results"])
def test_geocode_empty_result_is_not_found(payload):
    with pytest.raises(LocationError) as excinfo:
        mock_client(json_handler(payload)).geocode("Qwertyvillezz")
    assert excinfo.value.kind == "not_found"
    assert excinfo.value.query == "Qwertyvillezz"


def test_geocode_blank_query_fails_without_http_call():
    seen: list[httpx.Request] = []
    with pytest.raises(LocationError) as excinfo:
        mock_client(json_handler(make_geocoding_payload(), seen=seen)).geocode("   ")
    assert excinfo.value.kind == "not_found"
    assert seen == []


def test_geocode_http_failure():
    with pytest.raises(LocationError) as excinfo:
        mock_client(json_handler({"error": True, "reason": "Internal error"}, status=503)).geocode("Testville")
    assert excinfo.value.kind == "http_error"
    assert excinfo.value.status_code == 503
    assert "Internal error" in excinfo.value.detail


def test_geocode_timeout():
    with pytest.raises(LocationError) as excinfo:
        mock_client(raising_handler(httpx.ReadTimeout)).geocode("Testville")
    assert excinfo.value.kind == "timeout"


def test_geocode_invalid_json():
    client = mock_client(lambda request: httpx.Response(200, content=b"<html>not json"))
    with pytest.raises(LocationError) as excinfo:
        client.geocode("Testville")
    assert excinfo.value.kind == "invalid_response"


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"name": "NoCoords", "timezone": "Asia/Kolkata"}]},
        {"results": [{"name": "BadLat", "latitude": 123.0, "longitude": 1.0, "timezone": "UTC"}]},
        {"results": "Testville"},
        ["not", "an", "object"],
    ],
    ids=["missing-fields", "latitude-out-of-range", "results-not-list", "payload-not-object"],
)
def test_geocode_malformed_results(payload):
    with pytest.raises(LocationError) as excinfo:
        mock_client(json_handler(payload)).geocode("Testville")
    assert excinfo.value.kind == "invalid_response"


def test_connection_error_is_retried_once_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=make_geocoding_payload())

    assert mock_client(handler).geocode("Testville").name == "Testville"
    assert calls["n"] == 2


def test_persistent_connection_error_is_network_error():
    seen: list[httpx.Request] = []
    with pytest.raises(LocationError) as excinfo:
        mock_client(raising_handler(httpx.ConnectError, seen=seen)).geocode("Testville")
    assert excinfo.value.kind == "network_error"
    assert len(seen) == 2


def test_timeouts_are_not_retried():
    seen: list[httpx.Request] = []
    with pytest.raises(LocationError):
        mock_client(raising_handler(httpx.ConnectTimeout, seen=seen)).geocode("Testville")
    assert len(seen) == 1


def test_error_messages_are_user_safe():
    with pytest.raises(LocationError) as excinfo:
        mock_client(raising_handler(httpx.ReadTimeout)).geocode("Testville")
    error = excinfo.value
    assert str(error) == "Could not look up the location 'Testville'."
    assert "Traceback" not in json.dumps(error.to_dict())
    assert error.to_dict()["type"] == "LocationError"


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def test_forecast_success_preserves_raw_payload():
    payload = make_forecast_payload()
    seen: list[httpx.Request] = []
    location = make_location(latitude=12.3, longitude=76.6)

    forecast = mock_client(json_handler(payload, seen=seen)).forecast(location, VARIABLES)

    assert forecast.payload == payload
    assert forecast.location == location
    assert forecast.timezone == "Asia/Kolkata"
    assert forecast.local_now.isoformat() == "2026-09-17T14:30:00"
    assert len(forecast.hourly_times) == 48
    assert forecast.requested_variables == VARIABLES
    assert forecast.source_url is not None and forecast.fetched_at_utc is not None

    params = seen[0].url.params
    assert seen[0].url.host == "api.open-meteo.com"
    assert (params["latitude"], params["longitude"]) == ("12.3", "76.6")
    assert params["timezone"] == "auto"
    assert params["forecast_days"] == "2"
    assert params["hourly"] == ",".join(VARIABLES)
    assert params["current"] == ",".join(VARIABLES)


def test_forecast_requests_exactly_the_policy_variables(policy):
    required = required_weather_variables(policy)
    seen: list[httpx.Request] = []
    payload = make_forecast_payload(variables=required)

    forecast = mock_client(json_handler(payload, seen=seen)).forecast(make_location(), required)

    assert seen[0].url.params["hourly"].split(",") == required
    assert forecast.requested_variables == required


def test_forecast_requests_a_new_variable_when_policy_needs_it(policy_dir):
    from app.policy import load_policy
    from tests.conftest import minimal_sop

    humid = minimal_sop("SOP-TEST-01", when={"all": [{"fact": "wx.min.relative_humidity_2m", "op": "lte", "value": 20}]}, cite=[])
    required = required_weather_variables(load_policy(policy_dir(humid)))
    seen: list[httpx.Request] = []

    mock_client(json_handler(make_forecast_payload(variables=required), seen=seen)).forecast(make_location(), required)

    assert required == ["relative_humidity_2m"]
    assert seen[0].url.params["hourly"] == "relative_humidity_2m"


def test_forecast_empty_variables_is_a_programming_error():
    with pytest.raises(ValueError):
        mock_client(json_handler({})).forecast(make_location(), [])


def test_forecast_http_failure_includes_api_reason():
    body = {"error": True, "reason": "Cannot initialize WeatherVariable from invalid String value snow_dpth"}
    with pytest.raises(WeatherError) as excinfo:
        mock_client(json_handler(body, status=400)).forecast(make_location(), VARIABLES)
    assert excinfo.value.kind == "http_error"
    assert excinfo.value.status_code == 400
    assert "snow_dpth" in excinfo.value.detail
    assert str(excinfo.value) == "Could not get the weather forecast for Testville, Test State, Testland."


def test_forecast_timeout():
    with pytest.raises(WeatherError) as excinfo:
        mock_client(raising_handler(httpx.ReadTimeout)).forecast(make_location(), VARIABLES)
    assert excinfo.value.kind == "timeout"


def test_forecast_network_error():
    with pytest.raises(WeatherError) as excinfo:
        mock_client(raising_handler(httpx.ConnectError)).forecast(make_location(), VARIABLES)
    assert excinfo.value.kind == "network_error"


def test_forecast_invalid_json():
    client = mock_client(lambda request: httpx.Response(200, content=b"{truncated"))
    with pytest.raises(WeatherError) as excinfo:
        client.forecast(make_location(), VARIABLES)
    assert excinfo.value.kind == "invalid_response"


def _broken(mutate) -> dict[str, Any]:
    payload = make_forecast_payload()
    mutate(payload)
    return payload


MALFORMED_FORECASTS = [
    pytest.param(lambda p: p["hourly"].pop("time"), "missing or empty 'hourly.time'", id="missing-hourly-time"),
    pytest.param(lambda p: p["hourly"].update(time=[]), "missing or empty 'hourly.time'", id="empty-hourly-time"),
    pytest.param(lambda p: p.pop("hourly"), "missing 'hourly'", id="missing-hourly"),
    pytest.param(lambda p: p["hourly"].pop("uv_index"), "missing hourly variable 'uv_index'", id="missing-variable"),
    pytest.param(lambda p: p["hourly"]["uv_index"].pop(), "has 47 values for 48 times", id="short-array"),
    pytest.param(lambda p: p["hourly"].update(uv_index=5.0), "is not a list", id="not-a-list"),
    pytest.param(lambda p: p["hourly"]["uv_index"].__setitem__(3, "high"), "non-numeric value 'high'", id="string-value"),
    pytest.param(lambda p: p["hourly"]["uv_index"].__setitem__(3, True), "non-numeric value True", id="bool-value"),
    pytest.param(lambda p: p["hourly"]["time"].__setitem__(5, "yesterday"), "not an ISO timestamp", id="bad-time"),
    pytest.param(lambda p: p["hourly"]["time"].__setitem__(5, p["hourly"]["time"][4]), "not strictly increasing", id="duplicate-time"),
    pytest.param(lambda p: p["hourly"]["time"].__setitem__(0, "2026-09-17T00:00+05:30"), "expected local time", id="offset-time"),
    pytest.param(lambda p: p.pop("current"), "missing 'current.time'", id="missing-current"),
    pytest.param(lambda p: p.pop("timezone"), "missing 'timezone'", id="missing-timezone"),
]


@pytest.mark.parametrize(("mutate", "detail"), MALFORMED_FORECASTS)
def test_forecast_malformed_response_raises_weather_error(mutate, detail):
    with pytest.raises(WeatherError) as excinfo:
        mock_client(json_handler(_broken(mutate))).forecast(make_location(), VARIABLES)
    assert excinfo.value.kind == "invalid_response"
    assert detail in excinfo.value.detail


def test_forecast_null_values_are_kept_as_null():
    payload = make_forecast_payload(series={"visibility": None})
    forecast = mock_client(json_handler(payload)).forecast(make_location(), VARIABLES)
    assert set(forecast.hourly_series("visibility")) == {None}


# ---------------------------------------------------------------------------
# Fixture provider (dependency injection without HTTP)
# ---------------------------------------------------------------------------


def test_fixture_provider_validates_like_the_live_client(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps({"meta": {}, "geocoding": make_geocoding_payload(), "forecast": make_forecast_payload(variables=["uv_index"])}),
        encoding="utf-8",
    )
    provider = FixtureWeatherProvider.from_file(path)

    location = provider.geocode("anything")
    assert location.name == "Testville"
    assert provider.forecast(location, ["uv_index"]).hourly_series("uv_index")[0] == 5.0
    with pytest.raises(WeatherError, match="incomplete"):
        provider.forecast(location, ["uv_index", "visibility"])
