"""HTTP API tests with injected graphs (scripted intents, fake weather, stub/fixed composers). No network, no LLM key."""

import re

import pytest
from fastapi.testclient import TestClient

from app.api import ApiSettings, ApiSettingsError, create_app
from app.llm import LLMConfigError, LLMError
from app.policy import PolicyStore
from app.weather import LocationError, WeatherError
from tests.conftest import POLICY_DIR
from tests.graph_helpers import FakeWeather, FixedComposer, intent, make_graph

ORIGIN = "http://localhost:5173"
HOT = {"apparent_temperature": lambda t: 41.3 if t.hour == 18 else 30.0}
CYCLE = "Is it safe to cycle in Mysuru this evening?"
SCRIPT = {
    CYCLE: intent(activities=["cycling"], location="Mysuru", day="today", part_of_day="evening"),
    "Is it safe to cycle in Mysuru today?": intent(activities=["cycling"], location="Mysuru", day="today"),
    "What about this evening?": intent("follow_up", part_of_day="evening"),
    "scuba diving in Kochi?": intent(activities=["other_outdoor"], location="Kochi"),
    "cycle in Qwertyville": intent(activities=["cycling"], location="Qwertyville"),
}
INTERNAL_KEYS = {"raw_weather", "composition", "draft", "verification", "trace", "messages", "memory", "parsed_intent", "request", "composer_error", "failure"}


def client_for(weather=None, composer=None, parser=None, settings=None, script=SCRIPT):
    graph, parser, weather, composer = make_graph(script, weather=weather, composer=composer, parser=parser)
    app = create_app(graph=graph, policy_store=PolicyStore(POLICY_DIR), settings=settings or ApiSettings(frontend_origins=(ORIGIN,)))
    return TestClient(app, raise_server_exceptions=False), weather, composer


def chat(client, message, session_id=None):
    body = {"message": message} if session_id is None else {"message": message, "session_id": session_id}
    return client.post("/api/chat", json=body)


# 1 --------------------------------------------------------------------------


def test_health():
    client, *_ = client_for()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# 2 --------------------------------------------------------------------------


def test_chat_returns_answer_and_public_metadata_only():
    client, *_ = client_for(weather=FakeWeather(series=HOT))
    response = chat(client, CYCLE)
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "answered" and data["reason"] is None
    assert "SOP-ACT-01" in data["answer"] and "Weather data used (" in data["answer"]
    assert "Weather data used" not in data["answer_text"] and "Policy applied" not in data["answer_text"]
    assert data["answer"].startswith(data["answer_text"])
    assert data["location"] == "Mysuru, Karnataka, India"
    assert data["time_window"] == {
        "description": "this evening (17:00-21:00, Asia/Kolkata time)",
        "start": "2026-09-17T17:00",
        "end": "2026-09-17T21:00",
        "timezone": "Asia/Kolkata",
    }
    assert data["policy"] == {
        "sop_ids": ["SOP-ACT-01"],
        "severity": "high",
        "primary": {"id": "SOP-ACT-01", "title": "Dangerous heat for strenuous outdoor exercise", "severity": "high"},
        "also_applies": [],
    }
    assert data["weather_summary"]["source"] == "Open-Meteo"
    assert {"label": "feels-like temperature (highest)", "value": "41.3°C"} in data["weather_summary"]["values"]
    assert set(data) == {"session_id", "answer", "answer_text", "status", "reason", "location", "time_window", "weather_summary", "policy"}
    assert not INTERNAL_KEYS & set(data)


def test_message_whitespace_is_trimmed_not_truncated():
    client, *_ = client_for(weather=FakeWeather(series=HOT))
    assert chat(client, f"   {CYCLE}\n").json()["status"] == "answered"


# 3 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "message"),
        ({"message": ""}, "message"),
        ({"message": "   "}, "message"),
        ({"message": "x" * 1001}, "message"),
        ({"message": 42}, "message"),
        ({"message": "hi", "session_id": "short"}, "session_id"),
        ({"message": "hi", "session_id": "bad id with spaces!"}, "session_id"),
        ({"message": "hi", "session_id": "a" * 65}, "session_id"),
        ({"message": "hi", "extra": True}, "extra"),
    ],
    ids=["missing", "empty", "blank", "too-long", "not-string", "short-session", "bad-session-chars", "long-session", "extra-field"],
)
def test_invalid_requests_get_a_clean_422(body, field):
    client, weather, _ = client_for()
    response = client.post("/api/chat", json=body)
    data = response.json()

    assert response.status_code == 422
    assert data["error"]["code"] == "invalid_request"
    assert any(problem["field"] == field for problem in data["error"]["fields"]), data
    assert "Traceback" not in response.text
    assert weather.geocode_calls == [] and weather.forecast_calls == []


def test_validation_problem_text_is_clean():
    client, *_ = client_for()
    problems = client.post("/api/chat", json={"message": "   "}).json()["error"]["fields"]
    assert problems == [{"field": "message", "problem": "message must not be blank"}]


def test_non_json_body_is_a_clean_422():
    client, *_ = client_for()
    response = client.post("/api/chat", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_maximum_length_message_is_accepted():
    client, *_ = client_for()
    assert chat(client, "x" * 1000).status_code == 200


# 4-5 ------------------------------------------------------------------------


def test_session_id_is_created_when_missing():
    client, *_ = client_for()
    first = chat(client, CYCLE).json()["session_id"]
    second = chat(client, CYCLE).json()["session_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert first != second


def test_session_id_reuse_keeps_graph_memory():
    client, weather, _ = client_for(weather=FakeWeather(series=HOT))
    first = chat(client, "Is it safe to cycle in Mysuru today?").json()
    second = chat(client, "What about this evening?", first["session_id"]).json()

    assert second["session_id"] == first["session_id"]
    assert second["status"] == "answered"
    assert second["location"] == "Mysuru, Karnataka, India"
    assert second["time_window"]["description"] == "this evening (17:00-21:00, Asia/Kolkata time)"
    assert weather.geocode_calls == ["Mysuru"] and len(weather.forecast_calls) == 2


def test_a_new_session_does_not_see_another_sessions_memory():
    client, *_ = client_for()
    chat(client, "Is it safe to cycle in Mysuru today?", "session-aaaaaaaa")
    data = chat(client, "What about this evening?", "session-bbbbbbbb").json()
    assert data["status"] == "clarification" and data["reason"] == "activity_missing"


# 6-9 ------------------------------------------------------------------------


class RaisingParser:
    def __init__(self, error):
        self.error = error

    def parse(self, message, context):
        raise self.error


def test_llm_unavailable_is_503_with_an_honest_answer():
    client, weather, _ = client_for(parser=RaisingParser(LLMError("rate_limited", "HTTP 429: quota", 429)))
    response = chat(client, CYCLE)
    data = response.json()

    assert response.status_code == 503
    assert (data["status"], data["reason"]) == ("failure", "llm_unavailable")
    assert data["answer"] == "I'm unable to process requests right now. Please try again in a moment."
    assert data["policy"] is None and data["weather_summary"] is None
    assert "429" not in response.text and "quota" not in response.text
    assert weather.forecast_calls == []


def test_weather_failure_is_503_without_weather_values():
    client, *_ = client_for(weather=FakeWeather(forecast_error=WeatherError("x", kind="timeout", detail="ReadTimeout after 8s")))
    response = chat(client, CYCLE)
    data = response.json()

    assert response.status_code == 503
    assert (data["status"], data["reason"]) == ("failure", "weather_error")
    assert not re.search(r"\d", data["answer"])
    assert data["weather_summary"] is None and data["time_window"] is None
    assert "ReadTimeout" not in response.text


def test_location_problems():
    client, *_ = client_for()
    data = chat(client, "cycle in Qwertyville").json()
    assert (data["status"], data["reason"]) == ("failure", "location_not_found")

    error = LocationError("Could not look up the location 'Mysuru'.", query="Mysuru", kind="http_error", status_code=500)
    client, *_ = client_for(weather=FakeWeather(geocode_error=error))
    response = chat(client, CYCLE)
    assert response.status_code == 503 and response.json()["reason"] == "location_error"


def test_no_sop_response():
    client, *_ = client_for()
    response = chat(client, "scuba diving in Kochi?")
    data = response.json()

    assert response.status_code == 200
    assert (data["status"], data["reason"]) == ("no_guidance", "no_sop")
    assert data["answer"] == "I don't currently have guidance covering this situation."
    assert data["policy"] is None and data["weather_summary"] is None and data["location"] is None


def test_off_topic_response():
    client, *_ = client_for()
    data = chat(client, "Which laptop should I buy?").json()
    assert (data["status"], data["reason"]) == ("unsupported", "unsupported_request")


def test_verification_fallback_response():
    fabricated = FixedComposer({"text": "It is only 25°C, fine to ride (SOP-ACT-01).", "cited_sop_ids": ["SOP-ACT-01"]})
    client, *_ = client_for(weather=FakeWeather(series=HOT), composer=fabricated)
    response = chat(client, CYCLE)
    data = response.json()

    assert response.status_code == 200
    assert (data["status"], data["reason"]) == ("answered_fallback", "verification_failed")
    assert "25°C" not in data["answer"]
    assert "Avoid strenuous exercise during this time." in data["answer_text"]
    assert data["policy"]["sop_ids"] == ["SOP-ACT-01"]
    assert "unsupported number" not in response.text          # verifier details stay internal


# 10 -------------------------------------------------------------------------


def test_cors_allows_the_configured_origin_only():
    client, *_ = client_for()
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}

    allowed = client.options("/api/chat", headers={"Origin": ORIGIN, **preflight})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN

    blocked = client.options("/api/chat", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in blocked.headers

    simple = client.get("/health", headers={"Origin": ORIGIN})
    assert simple.headers["access-control-allow-origin"] == ORIGIN


def test_cors_settings_from_environment(tmp_path):
    settings = ApiSettings.from_env({"FRONTEND_ORIGIN": "http://localhost:3000, https://app.example.com/"}, load_dotenv_file=False)
    assert settings.frontend_origins == ("http://localhost:3000", "https://app.example.com")
    assert ApiSettings.from_env({}, load_dotenv_file=False).frontend_origins == ("http://localhost:5173",)
    with pytest.raises(ApiSettingsError, match="explicit origins"):
        ApiSettings.from_env({"FRONTEND_ORIGIN": "*"}, load_dotenv_file=False)
    with pytest.raises(ApiSettingsError, match="http"):
        ApiSettings.from_env({"FRONTEND_ORIGIN": "localhost:3000"}, load_dotenv_file=False)


# Startup, unexpected errors and static frontend ------------------------------


def test_missing_llm_configuration_keeps_health_up_and_chat_503():
    def factory():
        raise LLMConfigError("Missing LLM configuration: LLM_API_KEY")

    app = create_app(settings=ApiSettings(frontend_origins=(ORIGIN,)), graph_factory=factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = chat(client, CYCLE)
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "service_unavailable", "message": "The assistant is not available right now.", "fields": None}}
    assert "LLM_API_KEY" not in response.text


def test_unexpected_error_is_a_clean_500():
    class BrokenGraph:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    app = create_app(graph=BrokenGraph(), policy_store=PolicyStore(POLICY_DIR), settings=ApiSettings(frontend_origins=(ORIGIN,)))
    response = TestClient(app, raise_server_exceptions=False).post("/api/chat", json={"message": "hi"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in response.text and "Traceback" not in response.text


def test_built_frontend_is_served_when_present(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Weather Safety Assistant</title>", encoding="utf-8")
    graph, *_ = make_graph(SCRIPT)
    app = create_app(graph=graph, policy_store=PolicyStore(POLICY_DIR), settings=ApiSettings(frontend_origins=(ORIGIN,), frontend_dist=tmp_path))
    client = TestClient(app)
    assert "Weather Safety Assistant" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}
