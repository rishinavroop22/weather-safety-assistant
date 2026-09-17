"""Weather providers: the live Open-Meteo HTTP client and a fixture-backed provider.

Anything with ``geocode()`` and ``forecast()`` satisfies ``WeatherProvider``,
so the graph can be built with a real, fake, failing, or fixture provider
without monkeypatching.
"""

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.weather.models import (
    Location,
    LocationError,
    RawForecast,
    WeatherError,
    parse_forecast_payload,
    parse_geocoding_payload,
)

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SECONDS = 8.0
GEOCODING_RESULT_COUNT = 5   # a few candidates, so an invalid first result can be skipped
FORECAST_DAYS = 2            # today + tomorrow covers every supported time window


class WeatherProvider(Protocol):
    """What the rest of the app needs from a weather source."""

    def geocode(self, query: str) -> Location:
        """Resolve a place name. Raises ``LocationError``."""
        ...

    def forecast(self, location: Location, variables: Sequence[str]) -> RawForecast:
        """Fetch hourly values for ``variables`` at ``location``. Raises ``WeatherError``."""
        ...


class _FetchError(Exception):
    """Internal: a failed HTTP call, converted to LocationError/WeatherError by the caller."""

    def __init__(self, kind: str, detail: str, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.kind, self.detail, self.status_code = kind, detail, status_code


class OpenMeteoClient:
    """Live Open-Meteo client.

    Args:
        http_client: injected ``httpx.Client`` (tests pass one with ``httpx.MockTransport``).
            If omitted, the client creates and owns one.
        timeout: seconds, used only when creating the ``httpx.Client``.
        connect_retries: extra attempts after a connection error (not after timeouts
            or HTTP errors).
    """

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connect_retries: int = 1,
        geocoding_url: str = GEOCODING_URL,
        forecast_url: str = FORECAST_URL,
    ) -> None:
        self._http = http_client or httpx.Client(timeout=timeout)
        self._owns_http = http_client is None
        self.connect_retries = connect_retries
        self.geocoding_url = geocoding_url
        self.forecast_url = forecast_url

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def geocode(self, query: str) -> Location:
        """Resolve a place name to the first valid Open-Meteo geocoding result.

        Raises:
            LocationError: empty query or no results (``not_found``), timeout,
                HTTP/network failure, or malformed response.
        """
        cleaned = query.strip() if isinstance(query, str) else ""
        if not cleaned:
            raise LocationError("No location was given.", query=str(query), kind="not_found")

        params = {"name": cleaned, "count": GEOCODING_RESULT_COUNT, "language": "en", "format": "json"}
        try:
            payload, _ = self._get_json(self.geocoding_url, params)
        except _FetchError as exc:
            logger.warning("Geocoding failed for %r: %s (%s)", cleaned, exc.kind, exc.detail)
            raise LocationError(
                f"Could not look up the location '{cleaned}'.",
                query=cleaned, kind=exc.kind, detail=exc.detail, status_code=exc.status_code,
            ) from exc

        location = parse_geocoding_payload(payload, cleaned)
        logger.info(
            "Geocoded %r -> %s (%.4f, %.4f, %s)",
            cleaned, location.display_name, location.latitude, location.longitude, location.timezone,
        )
        return location

    def forecast(self, location: Location, variables: Sequence[str]) -> RawForecast:
        """Fetch today's and tomorrow's hourly forecast for exactly ``variables``.

        ``variables`` should come from ``required_weather_variables(policy)``.
        The request uses ``timezone=auto``, so all returned times are local to
        the location's coordinates.

        Raises:
            ValueError: if ``variables`` is empty.
            WeatherError: timeout, HTTP/network failure, or incomplete/malformed data.
        """
        requested = _normalise_variables(variables)
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": ",".join(requested),
            "current": ",".join(requested),
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
        }
        fetched_at = datetime.now(timezone.utc)
        try:
            payload, url = self._get_json(self.forecast_url, params)
        except _FetchError as exc:
            logger.warning("Forecast failed for %s: %s (%s)", location.display_name, exc.kind, exc.detail)
            raise WeatherError(
                f"Could not get the weather forecast for {location.display_name}.",
                kind=exc.kind, detail=exc.detail, status_code=exc.status_code,
            ) from exc

        forecast = parse_forecast_payload(
            payload, location=location, variables=requested, source_url=url, fetched_at_utc=fetched_at
        )
        if forecast.timezone != location.timezone:
            logger.warning(
                "Timezone mismatch for %s: geocoding=%s forecast=%s (using forecast)",
                location.display_name, location.timezone, forecast.timezone,
            )
        return forecast

    def _get_json(self, url: str, params: dict[str, Any]) -> tuple[Any, str]:
        """GET and decode JSON, mapping every failure to a ``_FetchError``."""
        for attempt in range(self.connect_retries + 1):
            try:
                response = self._http.get(url, params=params)
                break
            except httpx.TimeoutException as exc:
                raise _FetchError("timeout", f"{type(exc).__name__}: request timed out") from exc
            except httpx.ConnectError as exc:
                if attempt < self.connect_retries:
                    logger.info("Connection error to %s, retrying: %s", url, exc)
                    continue
                raise _FetchError("network_error", f"ConnectError: {exc}") from exc
            except httpx.TransportError as exc:
                raise _FetchError("network_error", f"{type(exc).__name__}: {exc}") from exc

        if not response.is_success:
            raise _FetchError("http_error", f"HTTP {response.status_code}: {_error_reason(response)}", response.status_code)
        try:
            return response.json(), str(response.url)
        except ValueError as exc:
            raise _FetchError("invalid_response", "response body is not valid JSON") from exc


class FixtureWeatherProvider:
    """Serves saved Open-Meteo responses, validated exactly like live ones.

    Fixture file format (see ``evals/fixtures/``)::

        {"meta": {...}, "geocoding": <raw geocoding JSON>, "forecast": <raw forecast JSON>}

    ``geocode`` ignores the query text and returns the fixture's location.
    """

    def __init__(self, geocoding_payload: Any, forecast_payload: Any) -> None:
        self._geocoding = geocoding_payload
        self._forecast = forecast_payload

    @classmethod
    def from_file(cls, path: Path | str) -> "FixtureWeatherProvider":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["geocoding"], data["forecast"])

    def geocode(self, query: str) -> Location:
        return parse_geocoding_payload(self._geocoding, query)

    def forecast(self, location: Location, variables: Sequence[str]) -> RawForecast:
        return parse_forecast_payload(self._forecast, location=location, variables=_normalise_variables(variables))


def _normalise_variables(variables: Sequence[str]) -> list[str]:
    requested = sorted(set(variables))
    if not requested:
        raise ValueError("at least one weather variable must be requested")
    return requested


def _error_reason(response: httpx.Response) -> str:
    """Open-Meteo errors look like {"error": true, "reason": "..."}; fall back to a short body excerpt."""
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("reason"):
            return str(body["reason"])
    except ValueError:
        pass
    return response.text[:200] or response.reason_phrase
