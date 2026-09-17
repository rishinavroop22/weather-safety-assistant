"""Map final graph state to the public ``ChatResponse``.

Pure formatting: every value comes from state the graph already produced
(location, window, the composer's reportable facts, the ranked SOPs). No weather
logic, SOP matching or safety judgement happens here.
"""

from datetime import timedelta
from typing import Any

from app.api.schemas import ChatResponse, PolicyOut, PolicyRefOut, TimeWindowOut, WeatherSummaryOut, WeatherValueOut
from app.graph.rendering import WEATHER_SECTION_HEADER, fact_label, format_fact_value, window_description
from app.policy import SOPMatch, Vocabulary

# Service problems the user can't fix by rephrasing: returned with HTTP 503.
SERVICE_FAILURES = frozenset({"llm_unavailable", "weather_error", "location_error"})


def to_chat_response(session_id: str, state: dict[str, Any], vocabulary: Vocabulary) -> ChatResponse:
    answer = state.get("final_answer") or ""
    failure = state.get("failure")
    location = state.get("location")
    facts = state.get("weather_facts")
    composition = state.get("composition")
    match = state.get("match_result")
    outcome = state.get("outcome") or "failure"

    has_policy_answer = outcome in ("answered", "answered_fallback") and composition is not None
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        answer_text=answer.split(f"\n\n{WEATHER_SECTION_HEADER}", 1)[0] if has_policy_answer else answer,
        status=outcome,
        reason=failure.kind if failure is not None else None,
        location=location.display_name if location is not None and has_policy_answer else None,
        time_window=_time_window(facts) if facts is not None and has_policy_answer else None,
        weather_summary=_weather_summary(composition, facts, vocabulary) if has_policy_answer else None,
        policy=_policy(match.primary, match.secondary) if has_policy_answer and match and match.primary else None,
    )


def http_status_for(response: ChatResponse) -> int:
    return 503 if response.reason in SERVICE_FAILURES else 200


def _time_window(facts: Any) -> TimeWindowOut:
    window = facts.window
    return TimeWindowOut(
        description=window_description(facts),
        start=window.start.isoformat(timespec="minutes"),
        end=(window.end + timedelta(hours=1)).isoformat(timespec="minutes"),
        timezone=facts.timezone,
    )


def _weather_summary(composition: Any, facts: Any, vocabulary: Vocabulary) -> WeatherSummaryOut:
    return WeatherSummaryOut(
        retrieved_at_utc=facts.fetched_at_utc.isoformat(timespec="seconds") if facts and facts.fetched_at_utc else None,
        values=[
            WeatherValueOut(label=fact_label(fact, vocabulary), value=format_fact_value(fact, value, vocabulary))
            for fact, value in composition.facts.items()
        ],
        unavailable=[fact_label(fact, vocabulary) for fact in composition.unavailable_facts],
    )


def _policy(primary: SOPMatch, secondary: list[SOPMatch]) -> PolicyOut:
    def ref(sop: SOPMatch) -> PolicyRefOut:
        return PolicyRefOut(id=sop.id, title=sop.title, severity=sop.severity)

    return PolicyOut(
        sop_ids=[primary.id, *(sop.id for sop in secondary)],
        severity=primary.severity,
        primary=ref(primary),
        also_applies=[ref(sop) for sop in secondary],
    )
