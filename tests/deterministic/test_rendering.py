"""Regression tests for user-facing rendering: fallback answers, time phrases, SOP ids.

Rendering only formats state; these tests pin the wording users see.
"""

import re
import uuid
from datetime import datetime

import pytest

from app.graph import run_turn
from app.graph.rendering import fill_placeholders, render_answer_text, render_fallback_text, window_description
from app.llm import ComposedAnswer
from app.weather import WeatherFacts, resolve_window
from tests.graph_helpers import FakeWeather, FixedComposer, intent, make_graph
from tests.weather_helpers import make_location

HOT = {"apparent_temperature": lambda t: 41.3 if t.hour == 18 else 30.0}
CYCLE = "Is it safe to cycle in Mysuru this evening?"
CYCLE_SCRIPT = {CYCLE: intent(activities=["cycling"], location="Mysuru", part_of_day="evening")}

# Phrases that address the answer writer rather than the user.
WRITER_INSTRUCTIONS = re.compile(
    r"\b(the user|user's|state the|say that|tell the user|lead with|recommend|advise|suggest|explain that|name the|do not describe)\b",
    re.IGNORECASE,
)


def facts_for(day: str, part: str, now: str) -> WeatherFacts:
    local_now = datetime.fromisoformat(now)
    return WeatherFacts(
        location=make_location(),
        timezone="Asia/Kolkata",
        local_now=local_now,
        window=resolve_window(day, part, local_now),
        values={},
        null_hours={},
    )


def ask(graph, message):
    return run_turn(graph, str(uuid.uuid4()), message)


# ---------------------------------------------------------------------------
# 1. Fallback answers are user-facing and policy-grounded
# ---------------------------------------------------------------------------


def test_shipped_sop_guidance_is_written_for_users(policy):
    for sop in policy.sops:
        for point in sop.guidance:
            assert not WRITER_INSTRUCTIONS.search(point), f"{sop.id}: {point!r}"


def test_fallback_is_a_user_facing_answer_built_from_state():
    graph, *_ = make_graph(CYCLE_SCRIPT, weather=FakeWeather(series=HOT), composer=FixedComposer(RuntimeError("model down")))
    state = ask(graph, CYCLE)
    answer = state["final_answer"]
    primary = state["match_result"].primary

    assert state["outcome"] == "answered_fallback"
    assert not WRITER_INSTRUCTIONS.search(answer)
    assert "couldn't reliably" not in answer and "as written" not in answer
    assert answer.startswith("Mysuru, Karnataka, India, this evening (17:00-21:00, Asia/Kolkata time): "
                             "Dangerous heat for strenuous outdoor exercise (SOP-ACT-01, high severity).")
    for point in primary.guidance:                      # the SOP's actual guidance, unchanged
        assert f"- {point}" in answer
    assert "Why this applies: feels-like temperature (highest) 41.3°C, at or above 40°C." in answer
    assert "feels-like temperature (highest): 41.3°C" in answer
    assert 'Policy applied: SOP-ACT-01 v1 "Dangerous heat for strenuous outdoor exercise" (severity: high)' in answer


def test_fallback_lists_every_selected_sop_and_all_triggering_factors():
    script = {"picnic": intent(activities=["picnic_outing"], location="Kochi", part_of_day="afternoon")}
    weather = FakeWeather(series={"precipitation_probability": 55.0, "wind_speed_10m": 28.0})
    graph, *_ = make_graph(script, weather=weather, composer=FixedComposer({"text": "", "cited_sop_ids": []}))
    answer = ask(graph, "picnic")["final_answer"]
    assert "Why this applies: chance of rain (highest) 55%, at or above 40%; wind speed (highest) 28 km/h, at or above 25 km/h." in answer

    script = {"beach": intent(activities=["beach_outing"], groups=["children", "elderly"], location="Kochi", part_of_day="afternoon")}
    weather = FakeWeather(series={"apparent_temperature": 38.0, "uv_index": 10.0})
    graph, *_ = make_graph(script, weather=weather, composer=FixedComposer({"text": "", "cited_sop_ids": []}))
    answer = ask(graph, "beach")["final_answer"]
    assert "Older adults outdoors in heat (SOP-VUL-02, high severity)." in answer
    assert "Also: Children outdoors with high UV (SOP-VUL-01, moderate severity)." in answer
    assert "Why this applies: UV index (highest) 10, at or above 7." in answer


def test_fallback_hides_non_weather_evidence(policy):
    script = {"dog": intent(activities=["dog_walk"], location="Kochi", part_of_day="afternoon")}
    weather = FakeWeather(series={"temperature_2m": 34.0, "uv_index": 7.0})
    graph, *_ = make_graph(script, weather=weather, composer=FixedComposer(RuntimeError("down")))
    answer = ask(graph, "dog")["final_answer"]
    assert "SOP-VUL-03" in answer
    assert "intent." not in answer and "dog_walk" not in answer
    assert "Why this applies: temperature (highest) 34°C, at or above 32°C; UV index (highest) 7, at or above 6." in answer


# ---------------------------------------------------------------------------
# 2. Natural time phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "part", "now", "expected"),
    [
        ("today", "now", "2026-09-17T14:30", "from now until 17:00 (Asia/Kolkata time)"),
        ("today", "now", "2026-09-17T21:10", "from now until midnight (Asia/Kolkata time)"),
        ("today", "now", "2026-09-17T23:15", "from now until 02:00 (Asia/Kolkata time)"),
        ("today", "whole_day", "2026-09-17T14:30", "for the rest of today (14:00-midnight, Asia/Kolkata time)"),
        ("tomorrow", "whole_day", "2026-09-17T14:30", "tomorrow (06:00-22:00, Asia/Kolkata time)"),
        ("today", "morning", "2026-09-17T00:30", "this morning (06:00-12:00, Asia/Kolkata time)"),
        ("today", "afternoon", "2026-09-17T00:30", "this afternoon (12:00-17:00, Asia/Kolkata time)"),
        ("today", "evening", "2026-09-17T14:30", "this evening (17:00-21:00, Asia/Kolkata time)"),
        ("today", "evening", "2026-09-17T18:40", "this evening (18:00-21:00, Asia/Kolkata time)"),
        ("today", "night", "2026-09-17T14:30", "tonight (21:00-midnight, Asia/Kolkata time)"),
        ("tomorrow", "morning", "2026-09-17T22:00", "tomorrow morning (06:00-12:00, Asia/Kolkata time)"),
        ("tomorrow", "evening", "2026-09-17T14:30", "tomorrow evening (17:00-21:00, Asia/Kolkata time)"),
        ("tomorrow", "night", "2026-09-17T14:30", "tomorrow night (21:00-midnight, Asia/Kolkata time)"),
    ],
)
def test_window_description_for_every_window(day, part, now, expected):
    assert window_description(facts_for(day, part, now)) == expected


ALL_WINDOWS = [
    ("today", "now", "2026-09-17T14:30"),
    ("today", "whole_day", "2026-09-17T14:30"),
    ("tomorrow", "whole_day", "2026-09-17T14:30"),
    ("today", "morning", "2026-09-17T00:30"),
    ("today", "afternoon", "2026-09-17T00:30"),
    ("today", "evening", "2026-09-17T14:30"),
    ("today", "night", "2026-09-17T14:30"),
    ("tomorrow", "evening", "2026-09-17T14:30"),
    ("tomorrow", "night", "2026-09-17T14:30"),
]
AWKWARD = re.compile(
    r"\b(during|for|in|at|on|over|from|until|by)\s+(from now|for the rest|this (morning|afternoon|evening)|tonight|tomorrow|right now)\b",
    re.IGNORECASE,
)
TEMPLATES = [
    "Storms are forecast in {location} during {window}.",
    "Avoid cycling for {window}.",
    "During {window}, stay indoors.",
    "Conditions in {location} {window} are hot.",
    "Plan ahead over {window}.",
    "For {location}, {window}: take care.",
]


@pytest.mark.parametrize(("day", "part", "now"), ALL_WINDOWS)
def test_window_placeholder_never_reads_awkwardly(day, part, now, vocabulary):
    facts = facts_for(day, part, now)
    composition = _composition(window_description(facts))
    for template in TEMPLATES:
        rendered = fill_placeholders(template, composition, vocabulary)
        assert not AWKWARD.search(rendered), rendered
        assert "{" not in rendered
        assert rendered[0].isupper(), rendered


def test_capitalised_preposition_capitalises_the_phrase(vocabulary):
    composition = _composition("from now until 17:00 (Asia/Kolkata time)")
    assert fill_placeholders("During {window}, stay indoors.", composition, vocabulary) == "From now until 17:00 (Asia/Kolkata time), stay indoors."
    assert fill_placeholders("Stay in {location} during {window}.", composition, vocabulary) == \
        "Stay in Mysuru, Karnataka, India from now until 17:00 (Asia/Kolkata time)."


def test_graph_answer_has_no_during_right_now():
    script = {"hike now": intent(activities=["hiking"], location="Kochi", day="today", part_of_day="now")}
    composer = FixedComposer({"text": "Strong gusts up to {wx.max.wind_gusts_10m} are forecast in {location} during {window} (SOP-ACT-03).",
                              "cited_sop_ids": ["SOP-ACT-03"]})
    graph, *_ = make_graph(script, weather=FakeWeather(series={"wind_gusts_10m": 55.0}), composer=composer)
    state = ask(graph, "hike now")
    assert state["outcome"] == "answered"
    assert state["final_answer"].startswith(
        "Strong gusts up to 55 km/h are forecast in Kochi, Kerala, India from now until 17:00 (Asia/Kolkata time) (SOP-ACT-03)."
    )
    assert "during right now" not in state["final_answer"] and "during from" not in state["final_answer"]


# ---------------------------------------------------------------------------
# 3. The applicable SOP id always appears in the answer text
# ---------------------------------------------------------------------------


def test_sop_id_is_added_when_the_verified_draft_omits_it():
    composer = FixedComposer({"text": "It will feel like {wx.max.apparent_temperature}, so skip the hard ride.", "cited_sop_ids": ["SOP-ACT-01"]})
    graph, *_ = make_graph(CYCLE_SCRIPT, weather=FakeWeather(series=HOT), composer=composer)
    state = ask(graph, CYCLE)

    assert state["verification"].passed                       # verifier rules unchanged
    body = state["final_answer"].split("\n\n")[0]
    assert body == "It will feel like 41.3°C, so skip the hard ride. (SOP-ACT-01)"


def test_sop_id_is_not_duplicated_when_present(vocabulary):
    composition = _composition("this evening (17:00-21:00, Asia/Kolkata time)")
    facts = facts_for("today", "evening", "2026-09-17T14:30")
    draft = ComposedAnswer(text="Take care (SOP-ACT-01).", cited_sop_ids=["SOP-ACT-01"])
    body = render_answer_text(draft, composition, facts, vocabulary).split("\n\n")[0]
    assert body == "Take care (SOP-ACT-01)."
    assert body.count("SOP-ACT-01") == 1


def test_every_cited_sop_id_appears_in_multi_sop_answers():
    script = {"beach": intent(activities=["beach_outing"], groups=["children", "elderly"], location="Kochi", part_of_day="afternoon")}
    composer = FixedComposer({"text": "Keep it short and shaded, and keep children in the shade.", "cited_sop_ids": ["SOP-VUL-02", "SOP-VUL-01"]})
    graph, *_ = make_graph(script, weather=FakeWeather(series={"apparent_temperature": 38.0, "uv_index": 10.0}), composer=composer)
    state = ask(graph, "beach")
    body = state["final_answer"].split("\n\n")[0]
    assert state["outcome"] == "answered"
    assert body.endswith("(SOP-VUL-02, SOP-VUL-01)")


def test_fallback_text_always_contains_the_sop_id(vocabulary):
    composition = _composition("this evening (17:00-21:00, Asia/Kolkata time)")
    facts = facts_for("today", "evening", "2026-09-17T14:30")
    assert "(SOP-ACT-01, high severity)" in render_fallback_text(composition, facts, vocabulary)


def _composition(window: str):
    from app.llm import CompositionRequest
    from app.policy import SOPMatch

    primary = SOPMatch(
        id="SOP-ACT-01", version=1, title="Dangerous heat", category="active_outdoor", severity="high",
        severity_rank=3, scope="activity", specificity=2, fallback=False,
        guidance=["Avoid strenuous exercise during this time."], cite=["wx.max.apparent_temperature"],
        evidence=["wx.max.apparent_temperature = 41.3, meets >= 40"],
    )
    return CompositionRequest(
        activities=["cycling"], groups=[], location_name="Mysuru, Karnataka, India", window_description=window,
        primary=primary, secondary=[], facts={"wx.max.apparent_temperature": 41.3},
        fact_labels={}, unavailable_facts=[], policy_thresholds={"wx.max.apparent_temperature": [40.0]},
    )
