"""One-off live integration check against the real Open-Meteo APIs.

    python -m evals.smoke_open_meteo            # default city: Bhopal
    python -m evals.smoke_open_meteo "Mysuru"

Resolves the city, fetches the forecast for the variables the current SOPs
need, builds facts for a few windows, and saves the raw API responses to
evals/fixtures/open_meteo_<city>_<date>.json for later offline evals.

Not part of the unit-test suite (it needs the internet).
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.policy import load_policy, required_weather_variables
from app.weather import OpenMeteoClient, WindowPassedError, build_facts

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "evals" / "fixtures"


def main(city: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    policy = load_policy(ROOT / "policy")
    variables = required_weather_variables(policy)

    recorded: dict[str, Any] = {}

    def record(response: httpx.Response) -> None:
        response.read()
        key = "geocoding" if "geocoding" in response.request.url.host else "forecast"
        recorded[key] = {"url": str(response.request.url), "status": response.status_code, "body": response.json()}

    with httpx.Client(timeout=10.0, event_hooks={"response": [record]}) as http:
        client = OpenMeteoClient(http)
        location = client.geocode(city)
        forecast = client.forecast(location, variables)

    print(f"\nLocation: {location.display_name}  lat={location.latitude} lon={location.longitude} tz={location.timezone}")
    print(f"Forecast timezone: {forecast.timezone}  local now: {forecast.local_now:%Y-%m-%d %H:%M}  hours: {len(forecast.hourly_times)}")
    missing = [v for v in variables if v not in forecast.payload["hourly"]]
    print(f"Requested variables ({len(variables)}): {', '.join(variables)}")
    print(f"Missing variables: {missing or 'none'}")

    for day, part in [("today", "now"), ("today", "whole_day"), ("today", "evening"), ("tomorrow", "morning")]:
        try:
            facts = build_facts(forecast, day=day, part_of_day=part, vocabulary=policy.vocabulary)
        except WindowPassedError as exc:
            print(f"\n[{day} {part}] {exc}")
            continue
        print(f"\n[{day} {part}] {facts.window.label} ({facts.timezone}), null hours: {facts.null_hours or 'none'}")
        for key, value in sorted(facts.values.items()):
            print(f"  {key:<36} {value}")

    slug = re.sub(r"[^a-z0-9]+", "_", location.name.lower()).strip("_")
    path = FIXTURES_DIR / f"open_meteo_{slug}_{forecast.local_now:%Y%m%d}.json"
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {
        "meta": {
            "description": "Real Open-Meteo responses recorded by evals/smoke_open_meteo.py",
            "query": city,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requested_variables": variables,
            "geocoding_url": recorded["geocoding"]["url"],
            "forecast_url": recorded["forecast"]["url"],
        },
        "geocoding": recorded["geocoding"]["body"],
        "forecast": recorded["forecast"]["body"],
    }
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved fixture: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Bhopal"))
