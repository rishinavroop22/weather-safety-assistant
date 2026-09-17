"""Time windows and build_facts (no network)."""

from datetime import datetime
from pathlib import Path

import pytest

from app.policy import QueryProfile, match_sops, required_weather_variables
from app.policy.models import iter_leaves
from app.weather import (
    FixtureWeatherProvider,
    WeatherError,
    WindowPassedError,
    build_facts,
    parse_forecast_payload,
    resolve_window,
)
from tests.weather_helpers import CALM_HOURLY, make_forecast_payload, make_location

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "evals" / "fixtures"


def forecast_from(payload, variables=None):
    return parse_forecast_payload(payload, location=make_location(), variables=variables or sorted(CALM_HOURLY))


def hours(window) -> list[str]:
    return [h.strftime("%d %H") for h in window.hours]


def at(local: str) -> datetime:
    return datetime.fromisoformat(local)


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------


def test_now_is_current_hour_plus_two():
    window = resolve_window("today", "now", at("2026-09-17T14:30"))
    assert hours(window) == ["17 14", "17 15", "17 16"]


def test_now_crosses_midnight():
    window = resolve_window("today", "now", at("2026-09-17T23:15"))
    assert hours(window) == ["17 23", "18 00", "18 01"]


def test_now_with_tomorrow_is_rejected():
    with pytest.raises(ValueError):
        resolve_window("tomorrow", "now", at("2026-09-17T10:00"))


@pytest.mark.parametrize(
    ("part", "first", "last"),
    [("morning", "17 06", "17 11"), ("afternoon", "17 12", "17 16"), ("evening", "17 17", "17 20"), ("night", "17 21", "17 23")],
)
def test_part_of_day_definitions_today(part, first, last):
    window = resolve_window("today", part, at("2026-09-17T00:30"))
    assert (hours(window)[0], hours(window)[-1]) == (first, last)
    assert not window.partially_elapsed


def test_evening_before_it_starts_is_the_full_window():
    window = resolve_window("today", "evening", at("2026-09-17T14:30"))
    assert hours(window) == ["17 17", "17 18", "17 19", "17 20"]
    assert window.label == "2026-09-17 17:00-21:00"


def test_evening_already_started_drops_past_hours():
    window = resolve_window("today", "evening", at("2026-09-17T18:40"))
    assert hours(window) == ["17 18", "17 19", "17 20"]
    assert window.partially_elapsed


def test_evening_after_it_ended_is_explicitly_passed():
    with pytest.raises(WindowPassedError) as excinfo:
        resolve_window("today", "evening", at("2026-09-17T21:05"))
    assert "already passed" in str(excinfo.value)
    assert excinfo.value.local_now == at("2026-09-17T21:05")


def test_tomorrow_morning():
    window = resolve_window("tomorrow", "morning", at("2026-09-17T22:00"))
    assert hours(window) == ["18 06", "18 07", "18 08", "18 09", "18 10", "18 11"]


def test_whole_day_today_runs_from_now_to_23():
    window = resolve_window("today", "whole_day", at("2026-09-17T14:30"))
    assert hours(window)[0] == "17 14" and hours(window)[-1] == "17 23"
    assert len(window.hours) == 10


def test_whole_day_tomorrow_is_06_to_21():
    window = resolve_window("tomorrow", "whole_day", at("2026-09-17T14:30"))
    assert hours(window)[0] == "18 06" and hours(window)[-1] == "18 21"
    assert len(window.hours) == 16


# ---------------------------------------------------------------------------
# Timezone: "now" comes from the API's local current.time, not the server clock
# ---------------------------------------------------------------------------


def test_same_instant_in_two_timezones_gives_different_windows(vocabulary):
    # 13:15 UTC is 18:45 in Kolkata and 09:15 in New York.
    kolkata = forecast_from(make_forecast_payload(current_time="2026-09-17T18:45", timezone="Asia/Kolkata"))
    new_york = forecast_from(make_forecast_payload(current_time="2026-09-17T09:15", timezone="America/New_York"))

    kolkata_facts = build_facts(kolkata, day="today", part_of_day="evening", vocabulary=vocabulary)
    new_york_facts = build_facts(new_york, day="today", part_of_day="evening", vocabulary=vocabulary)

    assert hours(kolkata_facts.window) == ["17 18", "17 19", "17 20"]
    assert hours(new_york_facts.window) == ["17 17", "17 18", "17 19", "17 20"]
    assert (kolkata_facts.timezone, new_york_facts.timezone) == ("Asia/Kolkata", "America/New_York")


def test_server_clock_is_not_used(vocabulary):
    # A payload whose local "now" is years in the past still resolves windows relative to it.
    payload = make_forecast_payload(current_time="2020-01-01T08:00", first_day="2020-01-01")
    facts = build_facts(forecast_from(payload), day="today", part_of_day="now", vocabulary=vocabulary)
    assert facts.window.start == at("2020-01-01T08:00")
    assert facts.local_now == at("2020-01-01T08:00")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregates_follow_the_vocabulary(vocabulary):
    payload = make_forecast_payload(
        current_time="2026-09-17T14:10",
        series={
            "precipitation": lambda t: {17: 1.2, 18: 4.5, 19: 0.1, 20: 0.0}.get(t.hour, 0.0),
            "temperature_2m": lambda t: 20.0 + t.hour,
            "uv_index": lambda t: {17: 3.0, 18: 1.0}.get(t.hour, 0.0),
            "visibility": lambda t: 24000.0 - t.hour * 100,
            "weather_code": lambda t: {18: 95, 19: 61}.get(t.hour, 3),
        },
    )
    facts = build_facts(forecast_from(payload), day="today", part_of_day="evening", vocabulary=vocabulary)
    values = facts.values

    assert values["wx.sum.precipitation"] == 5.8
    assert values["wx.max.precipitation"] == 4.5
    assert (values["wx.max.temperature_2m"], values["wx.min.temperature_2m"]) == (40.0, 37.0)
    assert values["wx.max.uv_index"] == 3.0
    assert values["wx.min.visibility"] == 22000.0
    assert values["wx.codes"] == [3, 61, 95]


def test_only_meaningful_aggregates_are_produced(vocabulary):
    facts = build_facts(forecast_from(make_forecast_payload()), day="today", part_of_day="now", vocabulary=vocabulary)
    expected = {"wx.codes"} | {
        f"wx.{aggregate}.{variable}"
        for variable, info in vocabulary.weather_variables.items()
        for aggregate in info.aggregates
    }
    assert set(facts.values) == expected
    assert "wx.sum.uv_index" not in facts.values
    assert "wx.max.weather_code" not in facts.values


def test_hours_outside_the_window_are_ignored(vocabulary):
    payload = make_forecast_payload(series={"wind_gusts_10m": lambda t: 90.0 if t.hour == 12 else 15.0})
    facts = build_facts(forecast_from(payload), day="today", part_of_day="evening", vocabulary=vocabulary)
    assert facts.values["wx.max.wind_gusts_10m"] == 15.0


# ---------------------------------------------------------------------------
# Missing values stay missing; nothing is invented
# ---------------------------------------------------------------------------


def test_one_null_hour_makes_the_variable_unknown(vocabulary):
    payload = make_forecast_payload(series={"visibility": lambda t: None if t.hour == 18 else 15000.0})
    facts = build_facts(forecast_from(payload), day="today", part_of_day="evening", vocabulary=vocabulary)

    assert "wx.min.visibility" in facts.values
    assert facts.values["wx.min.visibility"] is None
    assert facts.null_hours == {"visibility": 1}
    assert facts.values["wx.max.uv_index"] == 5.0  # other variables unaffected


def test_null_weather_codes_make_codes_unknown(vocabulary):
    payload = make_forecast_payload(series={"weather_code": None})
    facts = build_facts(forecast_from(payload), day="today", part_of_day="now", vocabulary=vocabulary)
    assert facts.values["wx.codes"] is None


def test_all_null_precipitation_is_none_not_zero(vocabulary):
    payload = make_forecast_payload(series={"precipitation": None})
    facts = build_facts(forecast_from(payload), day="today", part_of_day="now", vocabulary=vocabulary)
    assert facts.values["wx.sum.precipitation"] is None
    assert facts.values["wx.max.precipitation"] is None


def test_null_outside_the_window_does_not_matter(vocabulary):
    payload = make_forecast_payload(series={"visibility": lambda t: None if t.hour == 3 else 15000.0})
    facts = build_facts(forecast_from(payload), day="today", part_of_day="evening", vocabulary=vocabulary)
    assert facts.values["wx.min.visibility"] == 15000.0
    assert facts.null_hours == {}


def test_unrequested_variables_produce_no_facts(vocabulary):
    payload = make_forecast_payload(variables=["uv_index"])
    facts = build_facts(forecast_from(payload, ["uv_index"]), day="today", part_of_day="now", vocabulary=vocabulary)
    assert facts.values == {"wx.max.uv_index": 5.0}


def test_forecast_not_covering_the_window_is_an_error(vocabulary):
    one_day = make_forecast_payload(days=1)
    with pytest.raises(WeatherError) as excinfo:
        build_facts(forecast_from(one_day), day="tomorrow", part_of_day="morning", vocabulary=vocabulary)
    assert excinfo.value.kind == "invalid_response"
    assert "2026-09-18T06:00" in excinfo.value.detail


def test_window_passed_is_raised_from_build_facts(vocabulary):
    payload = make_forecast_payload(current_time="2026-09-17T22:10")
    with pytest.raises(WindowPassedError):
        build_facts(forecast_from(payload), day="today", part_of_day="afternoon", vocabulary=vocabulary)


def test_facts_keep_provenance(vocabulary):
    forecast = forecast_from(make_forecast_payload())
    facts = build_facts(forecast, day="today", part_of_day="now", vocabulary=vocabulary)
    assert facts.location == forecast.location
    assert facts.local_now == forecast.local_now
    assert facts.policy_facts() == facts.values


# ---------------------------------------------------------------------------
# Compatibility with the Phase 1 policy engine
# ---------------------------------------------------------------------------


def _policy_facts(policy, vocabulary, series=None, part="now"):
    required = required_weather_variables(policy)
    payload = make_forecast_payload(variables=required, series=series)
    forecast = forecast_from(payload, required)
    return build_facts(forecast, day="today", part_of_day=part, vocabulary=vocabulary).policy_facts()


def test_every_weather_fact_used_by_sops_is_produced(policy, vocabulary):
    facts = _policy_facts(policy, vocabulary)
    referenced = {leaf.fact for sop in policy.sops for leaf in iter_leaves(sop.when)} | {f for sop in policy.sops for f in sop.cite}
    assert {f for f in referenced if f.startswith("wx.")} <= set(facts)


def test_calm_forecast_leads_to_baseline_sop(policy, vocabulary):
    facts = _policy_facts(policy, vocabulary)
    assert match_sops(policy, QueryProfile(activities=["cycling"]), facts).primary.id == "SOP-BASE-01"


def test_thunderstorm_inside_window_is_matched(policy, vocabulary):
    facts = _policy_facts(policy, vocabulary, series={"weather_code": lambda t: 95 if t.hour == 18 else 2}, part="evening")
    assert match_sops(policy, QueryProfile(activities=["walking"]), facts).primary.id == "SOP-GEN-01"


def test_null_visibility_becomes_unknown_in_the_engine(policy, vocabulary):
    facts = _policy_facts(policy, vocabulary, series={"visibility": None})
    result = match_sops(policy, QueryProfile(activities=["cycling"]), facts)
    assert result.primary is None
    assert [u.id for u in result.unevaluable] == ["SOP-TRV-02"]


@pytest.mark.skipif(not list(FIXTURES_DIR.glob("open_meteo_*.json")), reason="no recorded Open-Meteo fixture yet")
def test_recorded_real_fixture_builds_policy_facts(policy, vocabulary):
    required = required_weather_variables(policy)
    for path in sorted(FIXTURES_DIR.glob("open_meteo_*.json")):
        provider = FixtureWeatherProvider.from_file(path)
        forecast = provider.forecast(provider.geocode("fixture"), required)
        for day, part in [("today", "now"), ("today", "whole_day"), ("tomorrow", "morning")]:
            facts = build_facts(forecast, day=day, part_of_day=part, vocabulary=vocabulary)
            assert facts.window.hours
            match_sops(policy, QueryProfile(activities=["walking"]), facts.policy_facts())
