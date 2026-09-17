"""Context engineering: what each LLM role receives, and how turns are merged.

* ``validate_intent``   - reject malformed parser output before it enters state.
* ``build_request``     - deterministic follow-up merge (message overrides memory).
* ``conversation_context`` - the small summary the intent parser may see.
* ``build_composition`` - the only information the answer composer may see.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

from app.graph.rendering import fact_label, window_description
from app.graph.state import SessionMemory, TurnRequest
from app.llm import CompositionRequest, ConversationContext, ParsedIntent
from app.policy import MatchResult, PolicySet, Vocabulary
from app.policy.models import WEATHER_CODES_FACT, iter_leaves
from app.weather import Location, WeatherFacts


class IntentError(ValueError):
    """Parser output is malformed or uses tags outside the vocabulary."""


def validate_intent(raw: Any, vocabulary: Vocabulary) -> ParsedIntent:
    """Validate parser output against the ``ParsedIntent`` schema and the vocabulary.

    Raises:
        IntentError: with a short, loggable reason.
    """
    data = raw.model_dump() if isinstance(raw, BaseModel) else raw
    try:
        intent = ParsedIntent.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'intent'}: {e['msg']}" for e in exc.errors())
        raise IntentError(f"malformed intent: {problems}") from None

    for field, allowed in (
        ("activities", vocabulary.activities),
        ("groups", vocabulary.groups),
        ("concerns", vocabulary.concerns),
    ):
        unknown = [tag for tag in getattr(intent, field) if tag not in allowed]
        if unknown:
            raise IntentError(f"unknown {field} tag(s): {unknown}")
    return intent


def build_request(intent: ParsedIntent, memory: SessionMemory | None) -> TurnRequest:
    """Merge this message with session memory.

    Rules:
      * Anything stated in this message wins.
      * For a ``follow_up`` (with memory), each missing field is inherited:
        activities, groups, concerns, day, part of day.
      * Location: an explicit location is used; otherwise the session's last
        resolved location is reused by ``resolve_location`` (any intent).
      * Defaults when still missing: day = today, part of day = whole_day.
    """
    inherited: list[str] = []
    values: dict[str, Any] = {
        "activities": list(intent.activities),
        "groups": list(intent.groups),
        "concerns": list(intent.concerns),
        "day": intent.day,
        "part_of_day": intent.part_of_day,
    }

    if intent.intent == "follow_up" and memory is not None:
        for field in values:
            previous = getattr(memory, field)
            if not values[field] and previous:
                values[field] = list(previous) if isinstance(previous, list) else previous
                inherited.append(field)

    if intent.location is None and memory is not None and memory.location is not None:
        inherited.append("location")

    day = values["day"] or "today"
    part_of_day = values["part_of_day"] or "whole_day"
    if part_of_day == "now" and day == "tomorrow":   # e.g. inherited "now" + "what about tomorrow?"
        part_of_day = "whole_day"

    return TurnRequest(
        intent=intent.intent,
        activities=values["activities"],
        groups=values["groups"],
        concerns=values["concerns"],
        location_query=intent.location,
        day=day,
        part_of_day=part_of_day,
        inherited=inherited,
    )


def conversation_context(memory: SessionMemory | None) -> ConversationContext | None:
    if memory is None or not (memory.activities or memory.location):
        return None
    return ConversationContext(
        activities=memory.activities,
        groups=memory.groups,
        location_name=memory.location.display_name if memory.location else None,
        day=memory.day,
        part_of_day=memory.part_of_day,
    )


def build_composition(
    *,
    request: TurnRequest,
    location: Location,
    facts: WeatherFacts,
    match: MatchResult,
    memory: SessionMemory | None,
    policy: PolicySet,
) -> CompositionRequest:
    """Assemble the composer's input from authoritative state.

    Reportable facts are exactly those cited by the selected SOPs; a cited fact
    with no value is listed as unavailable rather than invented. Policy thresholds
    are read from the selected SOPs' conditions, for the verifier's number checks.
    """
    vocabulary = policy.vocabulary
    if match.primary is None:
        raise ValueError("build_composition requires a selected SOP")
    selected = [match.primary, *match.secondary]
    cited = list(dict.fromkeys(fact for sop in selected for fact in sop.cite))
    values = facts.policy_facts()
    reportable = {fact: values[fact] for fact in cited if values.get(fact) is not None}

    return CompositionRequest(
        activities=request.activities,
        groups=request.groups,
        location_name=location.display_name,
        window_description=window_description(facts),
        primary=match.primary,
        secondary=match.secondary,
        facts=reportable,
        fact_labels={fact: fact_label(fact, vocabulary) for fact in reportable},
        unavailable_facts=[fact for fact in cited if fact not in reportable],
        policy_thresholds=policy_thresholds(policy, [sop.id for sop in selected]),
        previous_primary_sop_id=memory.last_primary_sop_id if memory else None,
        previous_window_description=memory.last_window_description if memory else None,
    )


def policy_thresholds(policy: PolicySet, sop_ids: list[str]) -> dict[str, list[float]]:
    """Numeric weather-condition values of the given SOPs, e.g. {"wx.max.wind_gusts_10m": [45.0]}."""
    thresholds: dict[str, set[float]] = {}
    for sop_id in sop_ids:
        sop = policy.get(sop_id)
        if sop is None:
            continue
        for leaf in iter_leaves(sop.when):
            if not leaf.fact.startswith("wx.") or leaf.fact == WEATHER_CODES_FACT:
                continue
            values = leaf.value if isinstance(leaf.value, list) else [leaf.value]
            thresholds.setdefault(leaf.fact, set()).update(float(v) for v in values)
    return {fact: sorted(values) for fact, values in thresholds.items()}
