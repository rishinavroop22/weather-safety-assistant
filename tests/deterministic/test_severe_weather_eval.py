"""Offline tests for evals/severe_live_weather.py: selection, SKIPPED handling, assertions, fixture replay."""

import pytest

from evals.severe_live_weather import grounding_assertions, run_fixture_replay, run_live, scan_for_severe_weather
from app.graph import run_turn
from app.policy import PolicyStore
from tests.conftest import POLICY_DIR
from tests.graph_helpers import FakeWeather, FixedComposer, intent, make_graph

STORM = {"weather_code": 95}


@pytest.fixture
def store():
    return PolicyStore(POLICY_DIR)


def test_fixture_replay_passes_all_grounding_assertions(store):
    result = run_fixture_replay(store)
    case = result["case"]

    assert result["status"] == "PASS", [a for a in case["assertions"] if not a["passed"]]
    assert "NOT a live-weather result" in result["note"]
    assert case["matched_sops"][0]["id"] == "SOP-GEN-01"            # the recorded response contains thunderstorm codes
    assert {a["name"] for a in case["assertions"]} == {
        "policy_answer_produced", "severity_high_or_critical", "sop_reproducible_from_request_facts",
        "sop_id_in_answer_text", "sop_cited_in_policy_line", "weather_values_equal_api_payload",
        "weather_values_shown_in_answer", "no_unsupported_numbers_or_thresholds",
    }


def test_scan_selects_the_most_severe_city(store):
    weather = FakeWeather(series_by_place={"Kochi": STORM, "Bengaluru": {"precipitation": 6.0}})
    rows, best = scan_for_severe_weather(weather, store.get(), ["Mysuru", "Bengaluru", "Kochi"])
    assert (best.city, best.sop_id, best.severity) == ("Kochi", "SOP-GEN-01", "critical")
    assert any(r.get("city") == "Bengaluru" and r.get("severity") == "high" for r in rows)   # high, but critical wins


def test_no_severe_weather_is_skipped_not_passed(store):
    result = run_live(store, FakeWeather(), ["Mysuru", "Bengaluru"], llm_mode="stub")
    assert result["status"] == "SKIPPED"
    assert "no scanned city" in result["reason"]
    assert result["attempts"] == []


def test_unknown_city_is_recorded_and_scan_continues(store):
    rows, best = scan_for_severe_weather(FakeWeather(series_by_place={"Kochi": STORM}), store.get(), ["Atlantis", "Kochi"])
    assert rows[0] == {"city": "Atlantis", "error": "LocationError: not_found"}
    assert best.city == "Kochi"


def test_live_run_with_stub_roles_passes_on_severe_fake_weather(store):
    result = run_live(store, FakeWeather(series_by_place={"Kochi": STORM}), ["Kochi"], llm_mode="stub")
    assert result["status"] == "PASS"
    assert result["attempts"][-1]["llm_roles"].startswith("stub")


def test_assertions_fail_on_a_non_severe_answer(store):
    graph, *_ = make_graph({"cycle": intent(activities=["cycling"], location="Kochi")})
    checks = {c["name"]: c["passed"] for c in grounding_assertions(run_turn(graph, "s-baseline", "cycle"), store.get())}
    assert checks["policy_answer_produced"] and not checks["severity_high_or_critical"]


def test_assertions_fail_when_the_answer_invents_a_threshold(store):
    # A draft the verifier rejects never reaches the user; here we check the eval's own number check directly
    # by feeding it a verified answer and then tampering with the final text.
    graph, *_ = make_graph({"cycle": intent(activities=["cycling"], location="Kochi")}, weather=FakeWeather(series=STORM))
    state = dict(run_turn(graph, "s-tamper", "cycle"))
    state["final_answer"] = state["final_answer"].replace("(SOP-GEN-01).", "(SOP-GEN-01). Gusts only matter above 90 km/h.", 1)
    checks = {c["name"]: c for c in grounding_assertions(state, store.get())}
    assert not checks["no_unsupported_numbers_or_thresholds"]["passed"]
    assert "unsupported number 90" in checks["no_unsupported_numbers_or_thresholds"]["detail"]


def test_llm_unavailable_real_attempt_is_not_counted_as_pass(store, monkeypatch):
    import evals.severe_live_weather as severe

    def fake_roles(mode, message, intent_, store_):
        if mode == "real":
            class Down:
                def parse(self, *_):
                    from app.llm import LLMError
                    raise LLMError("rate_limited", "HTTP 429", 429)
            return Down(), FixedComposer({"text": "", "cited_sop_ids": []}), "real LLM (test)"
        return severe.ScriptedIntentParser({message: intent_}), severe.TemplateAnswerComposer(), "stub LLM roles"

    monkeypatch.setattr(severe, "_roles", fake_roles)
    result = run_live(store, FakeWeather(series_by_place={"Kochi": STORM}), ["Kochi"], llm_mode="auto")
    assert [a["llm_roles"] for a in result["attempts"]] == ["real LLM (test)", "stub LLM roles"]
    assert result["attempts"][0]["failure"] == "llm_unavailable"
    assert result["status"] == "PASS" and result["attempts"][-1]["llm_roles"] == "stub LLM roles"

    only_real = run_live(store, FakeWeather(series_by_place={"Kochi": STORM}), ["Kochi"], llm_mode="real")
    assert only_real["status"] == "NOT RUN"
