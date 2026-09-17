"""Public request/response models for the HTTP API.

The response carries the user-facing answer plus a small, stable subset of the
graph state (location, window, weather values shown, SOP citation). Internal
state - prompts, drafts, raw Open-Meteo JSON, conversation history, verification
details, traces - is never exposed.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_CHARS = 1000
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"

ChatStatus = Literal["answered", "answered_fallback", "no_guidance", "unsupported", "clarification", "failure"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, description="The user's question.")
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Conversation id from a previous response; omit to start a new conversation.",
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class TimeWindowOut(BaseModel):
    description: str = Field(description='e.g. "this evening (17:00-21:00, Asia/Kolkata time)"')
    start: str = Field(description="first hour, local ISO time")
    end: str = Field(description="end of the window (exclusive), local ISO time")
    timezone: str


class WeatherValueOut(BaseModel):
    label: str
    value: str


class WeatherSummaryOut(BaseModel):
    source: Literal["Open-Meteo"] = "Open-Meteo"
    retrieved_at_utc: str | None
    values: list[WeatherValueOut]
    unavailable: list[str]


class PolicyRefOut(BaseModel):
    id: str
    title: str
    severity: str


class PolicyOut(BaseModel):
    sop_ids: list[str]
    severity: str = Field(description="severity of the selected (primary) SOP")
    primary: PolicyRefOut
    also_applies: list[PolicyRefOut]


class ChatResponse(BaseModel):
    session_id: str
    answer: str = Field(description="the complete user-facing answer, including the weather block and policy citation")
    answer_text: str = Field(description="the answer without the weather block and policy citation (shown separately as metadata)")
    status: ChatStatus
    reason: str | None = Field(default=None, description="why the turn left the normal path, e.g. location_missing, llm_unavailable")
    location: str | None = None
    time_window: TimeWindowOut | None = None
    weather_summary: WeatherSummaryOut | None = None
    policy: PolicyOut | None = None


class FieldProblem(BaseModel):
    field: str
    problem: str


class ErrorBody(BaseModel):
    code: Literal["invalid_request", "service_unavailable", "internal_error"]
    message: str
    fields: list[FieldProblem] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
