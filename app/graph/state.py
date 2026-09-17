"""Graph state: what flows between nodes, and what survives between turns.

Two kinds of fields:

* Session fields (``memory``, ``messages``) persist across turns via the checkpointer.
* Per-turn fields are reset by ``understand_query`` at the start of every turn,
  so nothing from a previous turn can leak into this turn's answer.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app.llm import ComposedAnswer, CompositionRequest, IntentType, ParsedIntent
from app.policy import MatchResult
from app.weather import Day, Location, PartOfDay, RawForecast, WeatherFacts

FailureKind = Literal[
    "invalid_intent",       # parser output malformed / unknown tags / not valid JSON / parser crashed
    "llm_unavailable",      # intent LLM call failed (timeout, rate limit, HTTP error, model unavailable)
    "unsupported_request",  # off topic: not a weather/outdoor-activity question (rejected early)
    "activity_missing",     # a follow-up with no earlier activity to build on
    "location_missing",     # no location in the message or the session
    "location_not_found",   # geocoding found nothing
    "location_error",       # geocoding failed (timeout, HTTP, bad response)
    "weather_error",        # forecast failed, or data does not cover the window
    "window_passed",        # the requested window is already over
    "no_sop",               # valid request, but match_sops found no applicable SOP
    "verification_failed",  # composed answer rejected; deterministic fallback used
]

Outcome = Literal["answered", "answered_fallback", "no_guidance", "unsupported", "clarification", "failure"]

OUTCOME_BY_FAILURE: dict[str, Outcome] = {
    "invalid_intent": "failure",
    "llm_unavailable": "failure",
    "unsupported_request": "unsupported",
    "activity_missing": "clarification",
    "location_missing": "clarification",
    "location_not_found": "failure",
    "location_error": "failure",
    "weather_error": "failure",
    "window_passed": "clarification",
    "no_sop": "no_guidance",
    "verification_failed": "answered_fallback",
}


class Failure(BaseModel):
    """Why a turn left the happy path. ``detail`` is for logs/evals, never shown to users."""

    model_config = ConfigDict(frozen=True)

    kind: FailureKind
    detail: str | None = None


class TurnRequest(BaseModel):
    """The request the graph acts on: this message merged with session memory."""

    model_config = ConfigDict(frozen=True)

    intent: IntentType
    activities: list[str]
    groups: list[str]
    concerns: list[str]
    location_query: str | None          # explicit location in this message, if any
    day: Day
    part_of_day: PartOfDay
    inherited: list[str] = Field(default_factory=list)   # fields taken from memory, for traceability


class SessionMemory(BaseModel):
    """Structured context carried between turns of one session (in memory only)."""

    model_config = ConfigDict(frozen=True)

    activities: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    day: Day | None = None
    part_of_day: PartOfDay | None = None
    location: Location | None = None
    location_query: str | None = None
    location_cache: dict[str, Location] = Field(default_factory=dict)   # casefolded query -> Location
    last_outcome: Outcome | None = None
    last_primary_sop_id: str | None = None
    last_window_description: str | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    violations: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    # --- input ---
    session_id: str
    user_message: str

    # --- session (persisted by the checkpointer) ---
    messages: Annotated[list[AnyMessage], add_messages]   # transcript, for Phase 4 context
    memory: SessionMemory | None

    # --- per turn (reset by understand_query) ---
    parsed_intent: ParsedIntent | None          # raw parser output, validated
    request: TurnRequest | None                 # after merging with memory
    location: Location | None                   # resolved place (lat/lon/timezone)
    raw_weather: RawForecast | None             # untouched Open-Meteo JSON + source URL
    weather_facts: WeatherFacts | None          # window + wx.* facts (authoritative numbers)
    match_result: MatchResult | None            # ranked SOPs, primary = selected SOP, evidence
    composition: CompositionRequest | None      # exactly what the composer was shown
    draft: ComposedAnswer | None                # composer output (None if invalid)
    composer_error: str | None
    verification: VerificationResult | None
    failure: Failure | None
    outcome: Outcome | None
    final_answer: str
    trace: list[str]                            # nodes executed this turn, in order


PER_TURN_RESET: dict[str, Any] = {
    "parsed_intent": None,
    "request": None,
    "location": None,
    "raw_weather": None,
    "weather_facts": None,
    "match_result": None,
    "composition": None,
    "draft": None,
    "composer_error": None,
    "verification": None,
    "failure": None,
    "outcome": None,
    "final_answer": "",
}
