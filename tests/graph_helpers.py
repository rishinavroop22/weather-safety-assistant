"""Test doubles for graph tests: fake weather provider, scripted composers, graph factory."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.graph import build_graph
from app.llm import ComposedAnswer, CompositionRequest, ScriptedIntentParser, TemplateAnswerComposer
from app.policy import PolicyStore
from app.weather import Location, LocationError, RawForecast, WeatherError, parse_forecast_payload
from tests.conftest import POLICY_DIR
from tests.weather_helpers import make_forecast_payload, make_location

PLACES = {
    "mysuru": make_location(name="Mysuru", admin1="Karnataka", country="India", latitude=12.30, longitude=76.64),
    "bengaluru": make_location(name="Bengaluru", admin1="Karnataka", country="India", latitude=12.97, longitude=77.59),
    "kochi": make_location(name="Kochi", admin1="Kerala", country="India", latitude=9.93, longitude=76.26),
}


class FakeWeather:
    """In-memory WeatherProvider. Records calls; weather can differ per place."""

    def __init__(
        self,
        *,
        series: Mapping[str, Any] | None = None,
        series_by_place: Mapping[str, Mapping[str, Any]] | None = None,
        current_time: str = "2026-09-17T14:30",
        geocode_error: LocationError | None = None,
        forecast_error: Exception | None = None,
    ) -> None:
        self.series = dict(series or {})
        self.series_by_place = {k: dict(v) for k, v in (series_by_place or {}).items()}
        self.current_time = current_time
        self.geocode_error = geocode_error
        self.forecast_error = forecast_error
        self.geocode_calls: list[str] = []
        self.forecast_calls: list[tuple[str, list[str]]] = []
        self.last_forecast: RawForecast | None = None

    def geocode(self, query: str) -> Location:
        self.geocode_calls.append(query)
        if self.geocode_error is not None:
            raise self.geocode_error
        location = PLACES.get(query.strip().casefold())
        if location is None:
            raise LocationError(f"No place called '{query}' was found.", query=query, kind="not_found")
        return location

    def forecast(self, location: Location, variables: Sequence[str]) -> RawForecast:
        self.forecast_calls.append((location.name, list(variables)))
        if self.forecast_error is not None:
            raise self.forecast_error
        series = {**self.series, **self.series_by_place.get(location.name, {})}
        payload = make_forecast_payload(current_time=self.current_time, series=series)
        self.last_forecast = parse_forecast_payload(payload, location=location, variables=sorted(set(variables)))
        return self.last_forecast


class FixedComposer:
    """Returns the same draft (or raises) regardless of input; records what it was shown."""

    def __init__(self, result: ComposedAnswer | dict[str, Any] | Callable[[CompositionRequest], Any] | Exception) -> None:
        self.result = result
        self.requests: list[CompositionRequest] = []

    def compose(self, request: CompositionRequest) -> Any:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        if callable(self.result):
            return self.result(request)
        return self.result


def intent(kind: str = "activity_safety", **fields: Any) -> dict[str, Any]:
    return {"intent": kind, **fields}


def make_graph(
    script: Mapping[str, Any] | None = None,
    *,
    weather: FakeWeather | None = None,
    composer: Any = None,
    parser: Any = None,
    policy_dir: Path = POLICY_DIR,
):
    """A compiled graph with the real policy engine and injected fakes (or real LLM roles)."""
    parser = parser or ScriptedIntentParser(script or {})
    weather = weather or FakeWeather()
    composer = composer or TemplateAnswerComposer()
    graph = build_graph(
        policy_store=PolicyStore(policy_dir),
        weather_provider=weather,
        intent_parser=parser,
        answer_composer=composer,
    )
    return graph, parser, weather, composer
