"""Weather-layer data models, errors, and validation of Open-Meteo JSON.

No HTTP happens here. The same parsers validate live responses (``client.py``)
and saved fixtures, so both paths enforce identical rules.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.policy.models import is_number

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WeatherLayerError(Exception):
    """Base error for the weather layer.

    Attributes:
        message: short, user-safe sentence (no stack traces or raw payloads).
        kind: machine-readable cause, one of
            ``not_found | timeout | http_error | network_error | invalid_response``.
        detail: technical detail for logs and evals; never shown to end users.
        status_code: HTTP status when ``kind == "http_error"``.
    """

    def __init__(self, message: str, *, kind: str, detail: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.detail = detail
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary for graph state and logs."""
        return {
            "type": type(self).__name__,
            "kind": self.kind,
            "message": self.message,
            "detail": self.detail,
            "status_code": self.status_code,
        }


class LocationError(WeatherLayerError):
    """The location could not be resolved (not found, API failure, or bad response)."""

    def __init__(self, message: str, *, query: str, kind: str, detail: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message, kind=kind, detail=detail, status_code=status_code)
        self.query = query


class WeatherError(WeatherLayerError):
    """Weather data could not be fetched, or is incomplete/malformed."""


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


class Location(BaseModel):
    """A resolved place. Coordinates and timezone come from Open-Meteo geocoding."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    admin1: str | None = None
    country: str | None = None
    country_code: str | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = Field(min_length=1)

    @property
    def display_name(self) -> str:
        """E.g. ``Bhopal, Madhya Pradesh, India`` - shown to users so a wrong match is visible."""
        return ", ".join(part for part in (self.name, self.admin1, self.country) if part)


def parse_geocoding_payload(payload: Any, query: str) -> Location:
    """Return the first valid result of an Open-Meteo geocoding response.

    Raises:
        LocationError: no results (``not_found``) or a malformed response
            / no result with the required fields (``invalid_response``).
    """
    if not isinstance(payload, dict):
        raise LocationError(
            f"Could not look up the location '{query}'.",
            query=query, kind="invalid_response", detail="geocoding response is not a JSON object",
        )
    results = payload.get("results")
    if results is None or results == []:
        raise LocationError(f"No place called '{query}' was found.", query=query, kind="not_found")
    if not isinstance(results, list):
        raise LocationError(
            f"Could not look up the location '{query}'.",
            query=query, kind="invalid_response", detail="geocoding 'results' is not a list",
        )

    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            return Location.model_validate(result)
        except ValidationError:
            continue
    raise LocationError(
        f"Could not look up the location '{query}'.",
        query=query, kind="invalid_response",
        detail="no geocoding result had name, latitude, longitude and timezone",
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


class RawForecast(BaseModel):
    """A validated Open-Meteo forecast response.

    ``payload`` is the untouched JSON (kept for auditing and for checking that
    numbers shown to users came from the API). ``local_now`` and
    ``hourly_times`` are parsed from it; they are naive datetimes in the
    location's local time, because the request uses ``timezone=auto``.
    """

    model_config = ConfigDict(frozen=True)

    location: Location
    requested_variables: list[str]
    payload: dict[str, Any]
    timezone: str
    local_now: datetime
    hourly_times: list[datetime]
    source_url: str | None = None
    fetched_at_utc: datetime | None = None

    def hourly_series(self, variable: str) -> list[float | int | None]:
        """The hourly values for one requested variable, aligned with ``hourly_times``."""
        return self.payload["hourly"][variable]


def parse_forecast_payload(
    payload: Any,
    *,
    location: Location,
    variables: Sequence[str],
    source_url: str | None = None,
    fetched_at_utc: datetime | None = None,
) -> RawForecast:
    """Validate an Open-Meteo forecast response and wrap it in ``RawForecast``.

    Checks: JSON object, ``timezone``, ``current.time``, ``hourly.time``
    (non-empty, parseable, strictly increasing), and for every requested
    variable an hourly list of the same length containing only numbers or null.
    Values are never filled in or converted.

    Raises:
        WeatherError: with ``kind="invalid_response"`` on any violation.
    """

    def fail(detail: str) -> WeatherError:
        return WeatherError(
            f"The weather data for {location.display_name} was incomplete, so it can't be used.",
            kind="invalid_response", detail=detail,
        )

    if not isinstance(payload, dict):
        raise fail("forecast response is not a JSON object")

    timezone = payload.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        raise fail("missing 'timezone'")

    current = payload.get("current")
    if not isinstance(current, dict) or "time" not in current:
        raise fail("missing 'current.time'")
    local_now = _parse_local_time(current["time"], "current.time", fail)

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise fail("missing 'hourly'")
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list) or not raw_times:
        raise fail("missing or empty 'hourly.time'")
    times = [_parse_local_time(t, "hourly.time", fail) for t in raw_times]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise fail("'hourly.time' is not strictly increasing")

    for variable in variables:
        series = hourly.get(variable)
        if series is None:
            raise fail(f"missing hourly variable '{variable}'")
        if not isinstance(series, list):
            raise fail(f"hourly '{variable}' is not a list")
        if len(series) != len(times):
            raise fail(f"hourly '{variable}' has {len(series)} values for {len(times)} times")
        bad = next((v for v in series if v is not None and not is_number(v)), None)
        if bad is not None:
            raise fail(f"hourly '{variable}' contains a non-numeric value {bad!r}")

    return RawForecast(
        location=location,
        requested_variables=list(variables),
        payload=payload,
        timezone=timezone,
        local_now=local_now,
        hourly_times=times,
        source_url=source_url,
        fetched_at_utc=fetched_at_utc,
    )


def _parse_local_time(value: Any, field: str, fail: Any) -> datetime:
    if not isinstance(value, str):
        raise fail(f"'{field}' value {value!r} is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise fail(f"'{field}' value {value!r} is not an ISO timestamp") from None
    if parsed.tzinfo is not None:
        raise fail(f"'{field}' value {value!r} has a UTC offset; expected local time (timezone=auto)")
    return parsed
