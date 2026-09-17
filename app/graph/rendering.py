"""Deterministic user-facing text.

Every number a user sees is formatted here from structured state (weather
facts, SOP metadata), never copied from model output. Failure, clarification
and no-guidance messages are fixed templates, so those paths involve no LLM.
"""

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from app.llm import ComposedAnswer, CompositionRequest
from app.policy import SOPMatch, Vocabulary
from app.weather import WeatherFacts

AGGREGATE_WORDS = {"max": "highest", "min": "lowest", "sum": "total"}
UNITLESS = {"index", "WMO code", ""}
TIGHT_UNITS = {"°C", "%"}
PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
CONTEXT_PLACEHOLDERS = ("location", "window")
WEATHER_SECTION_HEADER = "Weather data used ("
"""Start of the weather block appended to every SOP answer; lets the API separate prose from the data block."""
# {window} renders as a complete time phrase ("this evening (...)", "from now until 17:00 (...)"),
# so a preposition written directly before it ("during {window}") is dropped when filling it in.
PREPOSITION_BEFORE_WINDOW = re.compile(r"\b(during|for|in|at|on|over|from|until|by)\s+\{window\}", re.IGNORECASE)
# Evidence lines from the policy engine, e.g. "wx.max.apparent_temperature = 41.3, meets >= 40".
EVIDENCE_PATTERN = re.compile(r"^(wx\.[a-z]+\.[a-z0-9_]+) = (-?[\d.]+), meets (>=|>|<=|<|==) (-?[\d.]+)$")
COMPARISON_WORDS = {">=": "at or above", ">": "above", "<=": "at or below", "<": "below", "==": "equal to"}

FAILURE_MESSAGES = {
    "invalid_intent": "Sorry, I couldn't understand that request. Could you rephrase it with the activity, place and time you have in mind?",
    "llm_unavailable": "I'm unable to process requests right now. Please try again in a moment.",
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
    """A complete, natural time phrase for the window that needs no preposition.

    ``from now until 17:00 (Asia/Kolkata time)``, ``for the rest of today (14:00-midnight, ...)``,
    ``this evening (17:00-21:00, ...)``, ``tonight (21:00-midnight, ...)``,
    ``tomorrow morning (06:00-12:00, ...)``, ``tomorrow (06:00-22:00, ...)``.
    """
    window = facts.window
    start = _clock(window.start)
    end = _clock(window.end + timedelta(hours=1))
    zone = f"{facts.timezone} time"
    if window.part_of_day == "now":
        return f"from now until {end} ({zone})"
    if window.part_of_day == "whole_day":
        when = "for the rest of today" if window.day == "today" else "tomorrow"
    elif window.part_of_day == "night":
        when = "tonight" if window.day == "today" else "tomorrow night"
    else:
        when = f"this {window.part_of_day}" if window.day == "today" else f"tomorrow {window.part_of_day}"
    return f"{when} ({start}-{end}, {zone})"


def _clock(moment: datetime) -> str:
    return "midnight" if (moment.hour, moment.minute) == (0, 0) else f"{moment:%H:%M}"


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

    def window_without_preposition(match: re.Match[str]) -> str:
        phrase = composition.window_description
        return phrase[0].upper() + phrase[1:] if match.group(1)[0].isupper() else phrase

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "location":
            return composition.location_name
        if name == "window":
            return composition.window_description
        return format_fact_value(name, composition.facts[name], vocabulary)

    text = PREPOSITION_BEFORE_WINDOW.sub(window_without_preposition, text)
    return PLACEHOLDER_PATTERN.sub(replace, text)


def weather_section(composition: CompositionRequest, facts: WeatherFacts, vocabulary: Vocabulary) -> str:
    window = facts.window
    period = f"{window.start:%Y-%m-%d} {_clock(window.start)}-{_clock(window.end + timedelta(hours=1))}, {facts.timezone} time"
    lines = [f"{WEATHER_SECTION_HEADER}Open-Meteo forecast for {composition.location_name}, {period}):"]
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
    """Verified composer prose + weather values and policy citation from state.

    The answer contract requires the prose to name the SOPs it follows. Verification
    already guarantees the selected SOP is in ``cited_sop_ids``; any cited id the model
    left out of the text is appended here from state (never removed or replaced).
    """
    body = fill_placeholders(draft.text, composition, vocabulary).strip()
    missing = [sop_id for sop_id in composition.sop_ids if sop_id in draft.cited_sop_ids and sop_id not in body]
    if missing:
        body = f"{body} ({', '.join(missing)})"
    return "\n\n".join([body, weather_section(composition, facts, vocabulary), policy_section(composition)])


def render_fallback_text(composition: CompositionRequest, facts: WeatherFacts, vocabulary: Vocabulary) -> str:
    """Deterministic answer used when the composed draft is rejected or unavailable.

    Built only from state: the selected SOPs' authored (user-facing) guidance, the weather
    conditions that made them apply (from the engine's evidence), the weather values and
    the policy citation. No model prose and no advice beyond the SOP guidance.
    """
    blocks = []
    for index, sop in enumerate([composition.primary, *composition.secondary]):
        heading = f"{composition.location_name}, {composition.window_description}" if index == 0 else "Also"
        lines = [f"{heading}: {sop.title} ({sop.id}, {sop.severity} severity)."]
        lines += [f"- {point}" for point in sop.guidance]
        reasons = evidence_summary(sop, vocabulary)
        if reasons:
            lines.append(f"Why this applies: {'; '.join(reasons)}.")
        blocks.append("\n".join(lines))
    return "\n\n".join([*blocks, weather_section(composition, facts, vocabulary), policy_section(composition)])


def evidence_summary(sop: SOPMatch, vocabulary: Vocabulary) -> list[str]:
    """Weather conditions that triggered an SOP, e.g. ``wind gusts (highest) 55 km/h, at or above 45 km/h``."""
    reasons = []
    for line in sop.evidence:
        match = EVIDENCE_PATTERN.match(line)
        if not match:
            continue   # non-weather evidence (e.g. intent tags) is not shown to users
        fact, value, operator, threshold = match.groups()
        reasons.append(
            f"{fact_label(fact, vocabulary)} {format_fact_value(fact, float(value), vocabulary)}, "
            f"{COMPARISON_WORDS[operator]} {format_fact_value(fact, float(threshold), vocabulary)}"
        )
    return reasons


def _sop_line(sop: SOPMatch) -> str:
    return f'{sop.id} v{sop.version} "{sop.title}" (severity: {sop.severity})'


def _join(items: Iterable[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
