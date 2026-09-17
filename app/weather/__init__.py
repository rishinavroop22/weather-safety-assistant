"""Weather layer: resolve a location, fetch Open-Meteo data, build policy facts.

It never selects SOPs, judges safety, or calls the LLM.
"""

from app.weather.client import FixtureWeatherProvider, OpenMeteoClient, WeatherProvider
from app.weather.facts import (
    Day,
    PartOfDay,
    TimeWindow,
    WeatherFacts,
    WindowPassedError,
    build_facts,
    resolve_window,
)
from app.weather.models import (
    Location,
    LocationError,
    RawForecast,
    WeatherError,
    WeatherLayerError,
    parse_forecast_payload,
    parse_geocoding_payload,
)

__all__ = [
    "Day",
    "FixtureWeatherProvider",
    "Location",
    "LocationError",
    "OpenMeteoClient",
    "PartOfDay",
    "RawForecast",
    "TimeWindow",
    "WeatherError",
    "WeatherFacts",
    "WeatherLayerError",
    "WeatherProvider",
    "WindowPassedError",
    "build_facts",
    "parse_forecast_payload",
    "parse_geocoding_payload",
    "resolve_window",
]
