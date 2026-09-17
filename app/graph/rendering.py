"""Deterministic user-facing text.

Every number a user sees is formatted here from structured state (weather
facts, SOP metadata), never copied from model output. Failure, clarification
and no-guidance messages are fixed templates, so those paths involve no LLM.
"""

import re
from collections.abc import Iterable, Sequence
from datetime import datetime

from app.llm import ComposedAnswer, CompositionRequest
from app.policy import SOPMatch, Vocabulary
from app.weather import WeatherFacts

AGGREGATE_WORDS = {"max": "highest", "min": "lowest", "sum": "total"}
UNITLESS = {"index", "WMO code", ""}
TIGHT_UNITS = {"°C", "%"}
PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
CONTEXT_PLACEHOLDERS = ("location", "window")

FAILURE_MESSAGES = {
    "invalid_intent": "Sorry, I couldn't understand that request. Could you rephrase it with the activity, place and time you have in mind?",
    "unsupported_request": "I can only help with weather-related safety questions about outdoor activities, so I can't help with that request.",
    "activity_missing": "Which outdoor activity are you planning, and where?",
    "location_missing": "Which city or area should I check the weather for?",
    "location_not_found": "I couldn't determine that location. Please provide a city or nearby town name.",
    "location_error": "I couldn't look up that location right now, so I can't provide weather-based guidance. Please try again shortly.",
    "weather_error": "I couldn't retrieve the current weather data, so I can't provide weather-based guidance right now.",
    "window_passed": "That time window has already passed. Please specify another time.",
    "no_sop": "I don't currently have guidance covering this situation.",
    "verification_failed": "I couldn't reliably generate a policy-grounded response for that request.",
}


# ---------------------------------------------------------------------------
# Facts and windows
# ---------------------------------------------------------------------------


def fact_label(fact: str, vocabulary: Vocabulary) -> str:
    """``wx.max.wind_gusts_10m`` -> ``wind gusts (highest)``."""
    _, aggregate, variable = fact.split(".")
    info = vocabulary.weather_variables.get(variable)
    name = info.label if info and info.label else variable.replace("_", " ")
    return f"{name} ({AGGREGATE_WORDS.get(aggregate, aggregate)})"


def format_fact_value(fact: str, value: float | int, vocabulary: Vocabulary) -> str:
    """``41.3`` for ``wx.max.apparent_temperature`` -> ``41.3°C``."""
    info = vocabulary.weather_variables.get(fact.split(".")[2])
    unit = info.unit if info else ""
    number = f"{value:g}"
    if unit in UNITLESS:
        return number
    if unit in TIGHT_UNITS:
        return f"{number}{unit}"
    return f"{number} {unit}"


def window_description(facts: WeatherFacts) -> str:
    """E.g. ``this evening (2026-09-17 17:00-21:00, Asia/Kolkata time)``."""
    window = facts.window
    if window.part_of_day == "now":
        when = "right now"
    elif window.part_of_day == "whole_day":
        when = "the rest of today" if window.day == "today" else "tomorrow"
    else:
        when = f"this {window.part_of_day}" if window.day == "today" else f"tomorrow {window.part_of_day}"
    return f"{when} ({window.label}, {facts.timezone} time)"


# ---------------------------------------------------------------------------
# Failure / clarification / no-guidance
# ---------------------------------------------------------------------------


def failure_message(
    kind: str,
    *,
    location_name: str | None = None,
    local_time: datetime | None = None,
    unsupported_concerns: Sequence[str] = (),
    unavailable: Sequence[str] = (),
) -> str:
    """Fixed, user-safe text for every non-answer outcome."""
    message = FAILURE_MESSAGES[kind]
    if kind == "window_passed" and location_name and local_time:
        message += f" It is already {local_time:%H:%M} in {location_name}."
    if kind == "no_sop":
        if unsupported_concerns:
            message += f" Our policies don't cover {_join(unsupported_concerns)}."
        if unavailable:
            message += f" Some weather data I needed was unavailable ({_join(unavailable)}), so I couldn't check every policy."
    return message


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def fill_placeholders(text: str, composition: CompositionRequest, vocabulary: Vocabulary) -> str:
    """Replace ``{wx...}``, ``{location}`` and ``{window}`` with values from state."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "location":
            return composition.location_name
        if name == "window":
            return composition.window_description
        return format_fact_value(name, composition.facts[name], vocabulary)

    return PLACEHOLDER_PATTERN.sub(replace, text)


def weather_section(composition: CompositionRequest, facts: WeatherFacts, vocabulary: Vocabulary) -> str:
    lines = [f"Weather data used (Open-Meteo forecast for {composition.location_name}, {composition.window_description}):"]
    lines += [f"- {fact_label(f, vocabulary)}: {format_fact_value(f, v, vocabulary)}" for f, v in composition.facts.items()]
    if composition.unavailable_facts:
        lines.append(f"- not available: {_join(fact_label(f, vocabulary) for f in composition.unavailable_facts)}")
    if facts.fetched_at_utc:
        lines.append(f"Retrieved {facts.fetched_at_utc:%Y-%m-%d %H:%M} UTC.")
    return "\n".join(lines)


def policy_section(composition: CompositionRequest) -> str:
    lines = [f"Policy applied: {_sop_line(composition.primary)}"]
    lines += [f"Also applies: {_sop_line(sop)}" for sop in composition.secondary]
    return "\n".join(lines)


def render_answer_text(draft: ComposedAnswer, composition: CompositionRequest, facts: WeatherFacts, vocabulary: Vocabulary) -> str:
    """Verified composer prose + weather values and policy citation from state."""
    body = fill_placeholders(draft.text, composition, vocabulary).strip()
    return "\n\n".join([body, weather_section(composition, facts, vocabulary), policy_section(composition)])


def render_fallback_text(composition: CompositionRequest, facts: WeatherFacts, vocabulary: Vocabulary) -> str:
    """Used when verification fails: the authored guidance verbatim, no model prose."""
    lines = [FAILURE_MESSAGES["verification_failed"] + " Here is the guidance from our policy, as written:"]
    for sop in [composition.primary, *composition.secondary]:
        lines += ["", f"{sop.title}:"] + [f"- {point}" for point in sop.guidance]
    return "\n\n".join(["\n".join(lines), weather_section(composition, facts, vocabulary), policy_section(composition)])


def _sop_line(sop: SOPMatch) -> str:
    return f'{sop.id} v{sop.version} "{sop.title}" (severity: {sop.severity})'


def _join(items: Iterable[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
