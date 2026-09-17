"""Turn a raw forecast into policy-engine facts for one local time window.

``build_facts`` produces the flat namespace the policy engine reads:
``wx.<aggregate>.<variable>`` for each aggregate enabled in ``vocabulary.yaml``,
plus ``wx.codes``. It never fills gaps: if any hour in the window is null for a
variable, that variable's aggregates are ``None`` (UNKNOWN to the engine).

"Now" is always the API's ``current.time`` in the location's local time,
never the server clock.
"""

from datetime import datetime, time, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.policy.models import WEATHER_CODE_VARIABLE, WEATHER_CODES_FACT, Vocabulary
from app.weather.models import Location, RawForecast, WeatherError

Day = Literal["today", "tomorrow"]
PartOfDay = Literal["now", "morning", "afternoon", "evening", "night", "whole_day"]

NOW_EXTRA_HOURS = 2
PART_OF_DAY_HOURS: dict[str, tuple[int, int]] = {   # inclusive hour-starts, local time
    "morning": (6, 11),
    "afternoon": (12, 16),
    "evening": (17, 20),
    "night": (21, 23),
}
WHOLE_DAY_TODAY_LAST_HOUR = 23
WHOLE_DAY_TOMORROW_HOURS = (6, 21)
SUM_DECIMALS = 2   # sums are computed, not API values; rounding avoids float noise like 0.30000000000000004


class WindowPassedError(Exception):
    """The requested window is entirely in the past for the location's local time."""

    def __init__(self, day: Day, part_of_day: PartOfDay, local_now: datetime) -> None:
        self.day, self.part_of_day, self.local_now = day, part_of_day, local_now
        super().__init__(
            f"The {part_of_day.replace('_', ' ')} window for {day} has already passed "
            f"(local time is {local_now:%H:%M})."
        )


class TimeWindow(BaseModel):
    """A resolved window of whole local hours, ``start``..``end`` inclusive."""

    model_config = ConfigDict(frozen=True)

    day: Day
    part_of_day: PartOfDay
    start: datetime
    end: datetime
    partially_elapsed: bool = False   # today's window had already started; past hours dropped

    @property
    def hours(self) -> list[datetime]:
        count = int((self.end - self.start) / timedelta(hours=1)) + 1
        return [self.start + timedelta(hours=i) for i in range(count)]

    @property
    def label(self) -> str:
        """E.g. ``2026-09-17 17:00-21:00``: the end is exclusive, i.e. up to 20:59."""
        return f"{self.start:%Y-%m-%d %H:%M}-{self.end + timedelta(hours=1):%H:%M}"


class WeatherFacts(BaseModel):
    """Facts for one location and window, plus the provenance needed for auditing."""

    model_config = ConfigDict(frozen=True)

    location: Location
    timezone: str
    local_now: datetime
    window: TimeWindow
    values: dict[str, float | int | list[int] | None]
    null_hours: dict[str, int]      # variable -> number of null hourly values inside the window
    source_url: str | None = None
    fetched_at_utc: datetime | None = None

    def policy_facts(self) -> dict[str, Any]:
        """The ``wx.*`` facts, ready for ``match_sops``."""
        return dict(self.values)


def resolve_window(day: Day, part_of_day: PartOfDay, local_now: datetime) -> TimeWindow:
    """Map (day, part_of_day) to concrete local hours relative to ``local_now``.

    * now       - current hour through the next 2 hours (today only)
    * morning 06-11, afternoon 12-16, evening 17-20, night 21-23
    * whole_day - today: current hour-23; tomorrow: 06-21
    For today, hours already past are dropped (``partially_elapsed``).

    Raises:
        WindowPassedError: every hour of today's window is already past.
        ValueError: ``now`` combined with ``tomorrow``.
    """
    current_hour = local_now.replace(minute=0, second=0, microsecond=0)

    if part_of_day == "now":
        if day != "today":
            raise ValueError("'now' can only be used with 'today'")
        return TimeWindow(day=day, part_of_day=part_of_day, start=current_hour, end=current_hour + timedelta(hours=NOW_EXTRA_HOURS))

    date = current_hour.date() + timedelta(days=1 if day == "tomorrow" else 0)
    if part_of_day == "whole_day":
        first, last = (current_hour.hour, WHOLE_DAY_TODAY_LAST_HOUR) if day == "today" else WHOLE_DAY_TOMORROW_HOURS
    else:
        first, last = PART_OF_DAY_HOURS[part_of_day]
    start = datetime.combine(date, time(first))
    end = datetime.combine(date, time(last))

    partially_elapsed = False
    if day == "today":
        if end < current_hour:
            raise WindowPassedError(day, part_of_day, local_now)
        if start < current_hour:
            start, partially_elapsed = current_hour, True
    return TimeWindow(day=day, part_of_day=part_of_day, start=start, end=end, partially_elapsed=partially_elapsed)


def build_facts(
    forecast: RawForecast,
    *,
    day: Day,
    part_of_day: PartOfDay,
    vocabulary: Vocabulary,
) -> WeatherFacts:
    """Aggregate the forecast's hourly data over the requested local window.

    For each requested variable, computes the aggregates enabled for it in the
    vocabulary (``weather_code`` becomes ``wx.codes``, the sorted distinct codes).
    Any null hour in the window makes that variable's facts ``None``.

    Raises:
        WindowPassedError: the window is already over (see ``resolve_window``).
        WeatherError: the forecast does not contain every hour of the window.
        ValueError: a requested variable is not in the vocabulary (a programming error).
    """
    window = resolve_window(day, part_of_day, forecast.local_now)
    index = {t: i for i, t in enumerate(forecast.hourly_times)}
    absent = [h for h in window.hours if h not in index]
    if absent:
        raise WeatherError(
            f"The forecast for {forecast.location.display_name} does not cover the requested time.",
            kind="invalid_response",
            detail=f"window {window.label} missing hours: {', '.join(f'{h:%Y-%m-%dT%H:%M}' for h in absent)}",
        )
    positions = [index[h] for h in window.hours]

    values: dict[str, float | int | list[int] | None] = {}
    null_hours: dict[str, int] = {}
    for variable in forecast.requested_variables:
        info = vocabulary.weather_variables.get(variable)
        if info is None:
            raise ValueError(f"weather variable '{variable}' is not in the vocabulary")

        series = forecast.hourly_series(variable)
        window_values = [series[i] for i in positions]
        nulls = sum(v is None for v in window_values)
        if nulls:
            null_hours[variable] = nulls
        complete = None if nulls else window_values

        if variable == WEATHER_CODE_VARIABLE:
            values[WEATHER_CODES_FACT] = None if complete is None else sorted({int(v) for v in complete})
        for aggregate in info.aggregates:
            values[f"wx.{aggregate}.{variable}"] = _aggregate(aggregate, complete)

    return WeatherFacts(
        location=forecast.location,
        timezone=forecast.timezone,
        local_now=forecast.local_now,
        window=window,
        values=values,
        null_hours=null_hours,
        source_url=forecast.source_url,
        fetched_at_utc=forecast.fetched_at_utc,
    )


def _aggregate(aggregate: str, values: list[Any] | None) -> float | int | None:
    if values is None:
        return None
    if aggregate == "max":
        return max(values)
    if aggregate == "min":
        return min(values)
    if aggregate == "sum":
        return round(sum(values), SUM_DECIMALS)
    raise ValueError(f"unsupported aggregate '{aggregate}'")
