"""Contracts between the graph and language models.

The graph talks to two narrow roles, each with a typed input and output:

* ``IntentParser``   - free text -> ``ParsedIntent`` (vocabulary tags only).
* ``AnswerComposer`` - ``CompositionRequest`` -> ``ComposedAnswer`` (wording only).

Neither role can select an SOP, set a severity, or supply weather values:
those fields do not exist in their outputs. Phase 4 plugs a real LLM in
behind these protocols; the graph does not change.
"""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.policy import SOPMatch
from app.weather import Day, PartOfDay

IntentType = Literal["activity_safety", "follow_up", "off_topic"]
"""``off_topic`` = not a weather/outdoor-activity question at all. The parser never
decides whether an SOP applies; that is ``match_sops``'s job after weather is known."""


class ParsedIntent(BaseModel):
    """What the latest user message says, and nothing more.

    Nothing is inherited here; merging with earlier turns is deterministic
    code in the graph. Tags must come from ``policy/vocabulary.yaml``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: IntentType
    activities: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    location: str | None = Field(default=None, max_length=100)
    day: Day | None = None
    part_of_day: PartOfDay | None = None

    @field_validator("location")
    @classmethod
    def _blank_location_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def _now_is_today_only(self) -> "ParsedIntent":
        if self.part_of_day == "now" and self.day == "tomorrow":
            raise ValueError("part_of_day 'now' cannot be combined with day 'tomorrow'")
        return self


class ConversationContext(BaseModel):
    """What the intent parser may know about earlier turns (for spotting follow-ups)."""

    model_config = ConfigDict(frozen=True)

    activities: list[str]
    groups: list[str]
    location_name: str | None
    day: Day | None
    part_of_day: PartOfDay | None


class IntentParser(Protocol):
    def parse(self, message: str, context: ConversationContext | None) -> ParsedIntent | dict[str, Any]:
        """Extract structured intent. May return a dict; the graph validates it."""
        ...


class CompositionRequest(BaseModel):
    """Everything the composer is allowed to see.

    It deliberately excludes the raw user message, the raw weather payload,
    and SOP conditions. Weather numbers should be written as ``{placeholders}``:
    the keys of ``facts`` plus ``{location}`` and ``{window}``. The graph fills
    them in from its own state after verification. ``policy_thresholds`` holds the
    numeric condition values of the selected SOPs (what the evidence already shows).
    """

    model_config = ConfigDict(frozen=True)

    activities: list[str]
    groups: list[str]
    location_name: str
    window_description: str
    primary: SOPMatch
    secondary: list[SOPMatch]
    facts: dict[str, float | int]
    fact_labels: dict[str, str]
    policy_thresholds: dict[str, list[float]] = Field(default_factory=dict)
    unavailable_facts: list[str]
    previous_primary_sop_id: str | None = None
    previous_window_description: str | None = None

    @property
    def sop_ids(self) -> list[str]:
        return [self.primary.id, *(sop.id for sop in self.secondary)]


class ComposedAnswer(BaseModel):
    """The composer's output: prose plus the SOP ids it claims to follow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    cited_sop_ids: list[str]


class AnswerComposer(Protocol):
    def compose(self, request: CompositionRequest) -> ComposedAnswer | dict[str, Any]:
        """Write the answer. May return a dict; the graph validates and verifies it."""
        ...
