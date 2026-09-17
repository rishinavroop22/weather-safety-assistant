"""The LangGraph orchestration: 9 nodes, failure branches, session checkpointing.

    START -> understand_query -> resolve_location -> fetch_weather -> build_facts
          -> match_sops -> compose_answer -> verify_answer -> render_answer -> update_memory -> END

Any node that records a ``failure`` routes straight to ``update_memory`` with a
deterministic message already in ``final_answer``; ``verify_answer`` routes there
with the fallback answer when the draft is rejected.

Nodes only orchestrate: geocoding/forecasting live in ``app.weather``, SOP
selection in ``app.policy``, wording in the injected LLM roles, checks in
``verification.py`` and text in ``rendering.py``.
"""

import logging
from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.graph.context import IntentError, build_composition, build_request, conversation_context, validate_intent
from app.graph.rendering import failure_message, fact_label, render_answer_text, render_fallback_text, window_description
from app.graph.state import (
    OUTCOME_BY_FAILURE,
    PER_TURN_RESET,
    Failure,
    GraphState,
    SessionMemory,
    TurnRequest,
    VerificationResult,
)
from app.graph.verification import verify_draft
from app.llm import AnswerComposer, ComposedAnswer, CompositionRequest, IntentParser, ParsedIntent
from app.policy import MatchResult, PolicySet, QueryProfile, SOPMatch, UnevaluableSOP, match_sops, required_weather_variables
from app.weather import (
    Location,
    LocationError,
    RawForecast,
    TimeWindow,
    WeatherError,
    WeatherFacts,
    WeatherLayerError,
    WeatherProvider,
    WindowPassedError,
    build_facts,
)

logger = logging.getLogger(__name__)

NODE_ORDER = (
    "understand_query",
    "resolve_location",
    "fetch_weather",
    "build_facts",
    "match_sops",
    "compose_answer",
    "verify_answer",
    "render_answer",
    "update_memory",
)

# Pydantic models stored in state; registered so the checkpointer may deserialize them.
STATE_MODELS: tuple[type[BaseModel], ...] = (
    ParsedIntent, TurnRequest, SessionMemory, Failure, VerificationResult,
    Location, RawForecast, TimeWindow, WeatherFacts,
    MatchResult, SOPMatch, UnevaluableSOP,
    CompositionRequest, ComposedAnswer,
)


class PolicySource(Protocol):
    """Anything that returns the current policy (e.g. ``PolicyStore``)."""

    def get(self) -> PolicySet: ...


class AdvisorNodes:
    """The node functions, holding the injected dependencies."""

    def __init__(
        self,
        policy_store: PolicySource,
        weather_provider: WeatherProvider,
        intent_parser: IntentParser,
        answer_composer: AnswerComposer,
    ) -> None:
        self.policy_store = policy_store
        self.weather_provider = weather_provider
        self.intent_parser = intent_parser
        self.answer_composer = answer_composer

    # 1 ---------------------------------------------------------------------
    def understand_query(self, state: GraphState) -> dict[str, Any]:
        """LLM role #1 (structured intent), then a deterministic merge with memory."""
        vocabulary = self.policy_store.get().vocabulary
        memory = state.get("memory")
        update: dict[str, Any] = dict(PER_TURN_RESET)
        try:
            raw = self.intent_parser.parse(state["user_message"], conversation_context(memory))
            parsed = validate_intent(raw, vocabulary)
        except IntentError as exc:
            return {**update, **_fail("invalid_intent", detail=str(exc))}
        except Exception as exc:  # the parser is an external model: never let it crash the turn
            logger.exception("Intent parser raised")
            return {**update, **_fail("invalid_intent", detail=f"parser raised {type(exc).__name__}")}

        request = build_request(parsed, memory)
        update |= {"parsed_intent": parsed, "request": request}
        if parsed.intent == "off_topic":   # topic check only; SOP applicability is decided in match_sops
            return {**update, **_fail("unsupported_request", detail="off-topic request")}
        if not request.activities:
            return {**update, **_fail("activity_missing", detail="no activity in message or session")}
        return update

    # 2 ---------------------------------------------------------------------
    def resolve_location(self, state: GraphState) -> dict[str, Any]:
        """Explicit location -> session cache or geocoding; otherwise reuse the session's location."""
        request, memory = state["request"], state.get("memory")
        query = request.location_query
        if query is None:
            if memory is not None and memory.location is not None:
                return {"location": memory.location}
            return _fail("location_missing", detail="no location in message or session")

        cached = memory.location_cache.get(_cache_key(query)) if memory else None
        if cached is not None:
            return {"location": cached}
        try:
            return {"location": self.weather_provider.geocode(query)}
        except LocationError as exc:
            kind = "location_not_found" if exc.kind == "not_found" else "location_error"
            return _fail(kind, detail=f"{exc.kind}: {exc.detail or exc.message}")
        except Exception as exc:
            logger.exception("Geocoding raised")
            return _fail("location_error", detail=f"geocoding raised {type(exc).__name__}")

    # 3 ---------------------------------------------------------------------
    def fetch_weather(self, state: GraphState) -> dict[str, Any]:
        """Fetch exactly the variables the current SOPs need."""
        variables = required_weather_variables(self.policy_store.get())
        try:
            return {"raw_weather": self.weather_provider.forecast(state["location"], variables)}
        except WeatherLayerError as exc:
            return _fail("weather_error", detail=f"{exc.kind}: {exc.detail or exc.message}")
        except Exception as exc:
            logger.exception("Forecast raised")
            return _fail("weather_error", detail=f"forecast raised {type(exc).__name__}")

    # 4 ---------------------------------------------------------------------
    def build_facts(self, state: GraphState) -> dict[str, Any]:
        """Window + aggregates, via the weather layer. A passed window is never silently moved."""
        request, location = state["request"], state["location"]
        try:
            facts = build_facts(
                state["raw_weather"],
                day=request.day,
                part_of_day=request.part_of_day,
                vocabulary=self.policy_store.get().vocabulary,
            )
        except WindowPassedError as exc:
            return _fail("window_passed", detail=str(exc), location_name=location.name, local_time=exc.local_now)
        except (WeatherError, ValueError) as exc:
            return _fail("weather_error", detail=f"facts: {getattr(exc, 'detail', None) or exc}")
        return {"weather_facts": facts}

    # 5 ---------------------------------------------------------------------
    def match_sops(self, state: GraphState) -> dict[str, Any]:
        """The deterministic policy engine selects and ranks SOPs. No LLM involved."""
        policy = self.policy_store.get()
        request, facts = state["request"], state["weather_facts"]
        profile = QueryProfile(activities=request.activities, groups=request.groups, concerns=request.concerns)
        result = match_sops(policy, profile, facts.policy_facts())
        if result.primary is not None:
            return {"match_result": result}

        vocabulary = policy.vocabulary
        unsupported = [c.replace("_", " ") for c in request.concerns if c in vocabulary.concerns and not vocabulary.concerns[c].supported]
        unknown_facts = list(dict.fromkeys(f for u in result.unevaluable for f in u.unknown_facts if f.startswith("wx.") and f.count(".") == 2))
        return {
            "match_result": result,
            **_fail(
                "no_sop",
                detail=f"no SOP matched; unevaluable={[u.id for u in result.unevaluable]}",
                unsupported_concerns=unsupported,
                unavailable=[fact_label(f, vocabulary) for f in unknown_facts],
            ),
        }

    # 6 ---------------------------------------------------------------------
    def compose_answer(self, state: GraphState) -> dict[str, Any]:
        """LLM role #2 writes prose from a restricted request. Its output is only a draft."""
        composition = build_composition(
            request=state["request"],
            location=state["location"],
            facts=state["weather_facts"],
            match=state["match_result"],
            memory=state.get("memory"),
            policy=self.policy_store.get(),
        )
        try:
            raw = self.answer_composer.compose(composition)
            data = raw.model_dump() if isinstance(raw, BaseModel) else raw
            draft = ComposedAnswer.model_validate(data)
        except Exception as exc:
            logger.warning("Composer failed: %s", type(exc).__name__)
            return {"composition": composition, "draft": None, "composer_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        return {"composition": composition, "draft": draft}

    # 7 ---------------------------------------------------------------------
    def verify_answer(self, state: GraphState) -> dict[str, Any]:
        """Deterministic guard. On failure, the answer becomes the authored guidance verbatim."""
        vocabulary = self.policy_store.get().vocabulary
        composition = state.get("composition")
        result = verify_draft(
            state.get("draft"),
            composition,
            vocabulary=vocabulary,
            composer_error=state.get("composer_error"),
        )
        if result.passed:
            return {"verification": result}
        logger.warning("Verification failed: %s", result.violations)
        update = _fail("verification_failed", detail="; ".join(result.violations))
        if composition is not None:
            update["final_answer"] = render_fallback_text(composition, state["weather_facts"], vocabulary)
        return {"verification": result, **update}

    # 8 ---------------------------------------------------------------------
    def render_answer(self, state: GraphState) -> dict[str, Any]:
        """Fill placeholders and append weather values and the SOP citation from state."""
        text = render_answer_text(state["draft"], state["composition"], state["weather_facts"], self.policy_store.get().vocabulary)
        return {"final_answer": text, "outcome": "answered"}

    # 9 ---------------------------------------------------------------------
    def update_memory(self, state: GraphState) -> dict[str, Any]:
        """Record this turn in session memory. Every path ends here."""
        memory = state.get("memory") or SessionMemory()
        request, location = state.get("request"), state.get("location")
        match, facts = state.get("match_result"), state.get("weather_facts")
        changes: dict[str, Any] = {"last_outcome": state.get("outcome")}

        if request is not None and request.activities:   # don't wipe context on out-of-scope turns
            changes |= {
                "activities": request.activities,
                "groups": request.groups,
                "concerns": request.concerns,
                "day": request.day,
                "part_of_day": request.part_of_day,
            }
        if location is not None:
            changes |= {"location": location, "location_query": request.location_query or memory.location_query}
            if request.location_query:
                changes["location_cache"] = {**memory.location_cache, _cache_key(request.location_query): location}
        changes["last_primary_sop_id"] = match.primary.id if match and match.primary else None
        changes["last_window_description"] = window_description(facts) if facts else None

        return {
            "memory": memory.model_copy(update=changes),
            "messages": [HumanMessage(content=state["user_message"]), AIMessage(content=state["final_answer"])],
        }


def build_graph(
    *,
    policy_store: PolicySource,
    weather_provider: WeatherProvider,
    intent_parser: IntentParser,
    answer_composer: AnswerComposer,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Wire the 9 nodes with their dependencies.

    Args:
        policy_store: e.g. ``PolicyStore("policy")``; read on every turn, so SOP edits apply live.
        weather_provider: ``OpenMeteoClient``, ``FixtureWeatherProvider`` or any fake.
        intent_parser / answer_composer: LLM roles (stubs until Phase 4).
        checkpointer: defaults to an in-memory saver; the session id is the thread id.
    """
    nodes = AdvisorNodes(policy_store, weather_provider, intent_parser, answer_composer)
    graph = StateGraph(GraphState)
    for name in NODE_ORDER:
        graph.add_node(name, _traced(name, getattr(nodes, name)))

    graph.add_edge(START, "understand_query")
    for node, next_node in [
        ("understand_query", "resolve_location"),
        ("resolve_location", "fetch_weather"),
        ("fetch_weather", "build_facts"),
        ("build_facts", "match_sops"),
        ("match_sops", "compose_answer"),
        ("verify_answer", "render_answer"),
    ]:
        graph.add_conditional_edges(node, _continue_unless_failed(next_node), [next_node, "update_memory"])
    graph.add_edge("compose_answer", "verify_answer")
    graph.add_edge("render_answer", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(checkpointer=checkpointer or default_checkpointer())


def default_checkpointer() -> InMemorySaver:
    """Session memory that lives as long as the process (resets on restart, by design)."""
    allowed = [(model.__module__, model.__name__) for model in STATE_MODELS]
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def run_turn(graph: CompiledStateGraph, session_id: str, message: str) -> GraphState:
    """Run one chat turn for a session and return the final state."""
    return graph.invoke(
        {"session_id": session_id, "user_message": message},
        config={"configurable": {"thread_id": session_id}},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail(kind: str, *, detail: str | None = None, **message_context: Any) -> dict[str, Any]:
    return {
        "failure": Failure(kind=kind, detail=detail),
        "outcome": OUTCOME_BY_FAILURE[kind],
        "final_answer": failure_message(kind, **message_context),
    }


def _continue_unless_failed(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "update_memory" if state.get("failure") is not None else next_node

    route.__name__ = f"route_after_{next_node}"
    return route


def _traced(name: str, node: Callable[[GraphState], dict[str, Any]]) -> Callable[[GraphState], dict[str, Any]]:
    """Append the node name to ``trace``; the first node starts a fresh trace each turn."""

    def run(state: GraphState) -> dict[str, Any]:
        update = node(state)
        previous = [] if name == NODE_ORDER[0] else state.get("trace", [])
        return {**update, "trace": [*previous, name]}

    run.__name__ = name
    return run


def _cache_key(query: str) -> str:
    return " ".join(query.split()).casefold()
