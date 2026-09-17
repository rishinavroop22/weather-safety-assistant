"""Unit tests for the deterministic verifier: citations and field-aware number rules."""

import pytest

from app.graph import verify_draft
from app.llm import ComposedAnswer, CompositionRequest
from app.policy import SOPMatch

PRIMARY = SOPMatch(
    id="SOP-ACT-01", version=1, title="Dangerous heat", category="active_outdoor", severity="high",
    severity_rank=3, scope="activity", specificity=2, fallback=False,
    guidance=[
        "Use SPF 30 or higher.",
        "Wait at least 30 minutes after the last thunder.",
        "Keep babies under 6 months out of direct sun.",
        "Drink water every 15 to 20 minutes.",
    ],
    cite=["wx.max.apparent_temperature", "wx.max.wind_speed_10m"],
    evidence=["wx.max.apparent_temperature = 41.3, meets >= 40"],
)
COMPOSITION = CompositionRequest(
    activities=["running"], groups=[], location_name="Mysuru, Karnataka, India",
    window_description="this evening (2026-09-17 17:00-21:00, Asia/Kolkata time)",
    primary=PRIMARY, secondary=[],
    facts={
        "wx.max.apparent_temperature": 41.3,
        "wx.max.wind_speed_10m": 42.0,
        "wx.max.precipitation_probability": 78,
        "wx.min.visibility": 2140.0,
        "wx.max.uv_index": 7.3,
    },
    fact_labels={},
    policy_thresholds={"wx.max.apparent_temperature": [40.0], "wx.max.wind_gusts_10m": [45.0]},
    unavailable_facts=[],
)


def check(text: str, vocabulary, cited: list[str] | None = None):
    draft = ComposedAnswer(text=text, cited_sop_ids=["SOP-ACT-01"] if cited is None else cited)
    return verify_draft(draft, COMPOSITION, vocabulary=vocabulary)


# ---------------------------------------------------------------------------
# Accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "It feels like {wx.max.apparent_temperature} in {location} {window}.",
        "It feels like 41.3°C.",                      # exact weather value, right unit
        "It feels like about 41 degrees.",            # rounded value, spelled-out unit
        "Our limit is 40°C.",                         # policy threshold for a °C variable
        "Winds of 42 km/h are expected.",             # km/h value
        "The wind speed will reach 42.",              # no unit, but wind label nearby
        "Expect 42 kmph at most.",                    # unit alias
        "There is a 78% chance of rain.",
        "Rain is 78 percent likely.",
        "Visibility drops to 2140 m.",
        "Gusts above 45 km/h are dangerous.",         # threshold, narrowed by label
        "The UV index is 7.3.",                       # unitless variable via its label
        "Use SPF 30 sunscreen.",                      # authored number, same preceding word
        "Wait 30 minutes after thunder.",             # authored number, same following word
        "Keep babies under 6 months indoors.",
        "Drink water every 15 to 20 minutes.",
        "With gusts, it feels like 41.3°C.",          # label for another unit does not block a valid value
        "It is 41.3°C and you should use SPF 30.",
        "Between 17:00 and 21:00 on 2026-09-17.",     # times/dates from the window
        "Per SOP-ACT-01, take care.",                 # SOP id digits are not numbers
    ],
)
def test_supported_numbers_pass(text, vocabulary):
    result = check(text, vocabulary)
    assert result.passed, result.violations


# ---------------------------------------------------------------------------
# Rejected - including SPF-30-style collisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "number"),
    [
        # "30" appears in the guidance (SPF 30, 30 minutes) but not as a weather value.
        ("The wind speed is 30 km/h.", "30"),
        ("Wind speed is 30.", "30"),
        ("Expect 30 kmph winds.", "30"),
        ("There is a 30% chance of rain.", "30"),
        ("Visibility is only 30 metres.", "30"),
        ("It feels like 30 degrees.", "30"),
        ("Gusts could reach at least 30.", "30"),     # "least 30" is authored, but a weather label is nearby
        ("Rain of 30 mm is possible.", "30"),
        # Other authored numbers used as weather.
        ("Winds up to 6 km/h.", "6"),
        ("Humidity around 20 percent.", "20"),
        # Real values in the wrong field.
        ("It feels like 42°C.", "42"),                # 42 is the wind speed, not a temperature
        ("The wind speed is 41.3.", "41.3"),          # 41.3 is the temperature
        ("UV is 7.3 mm.", "7.3"),
        # Wrong values, thresholds and unrelated numbers.
        ("It feels like 38°C.", "38"),
        ("It feels like 41.4°C.", "41.4"),
        ("Risky above 50 km/h gusts.", "50"),
        ("Wait 45 minutes before leaving.", "45"),     # a threshold, but not in a weather context
        ("Children aged 5 should stay inside.", "5"),
        ("Leave before 18:30.", "18:30"),
    ],
)
def test_unsupported_numbers_fail(text, number, vocabulary):
    result = check(text, vocabulary)
    assert not result.passed
    assert any(v.startswith(f"unsupported number {number} ") for v in result.violations), result.violations


def test_placeholder_and_phrase_rules(vocabulary):
    result = check("Humidity is {wx.max.relative_humidity_2m}, guaranteed.", vocabulary)
    assert "unknown placeholder {wx.max.relative_humidity_2m}" in result.violations
    assert "banned phrase 'guaranteed'" in result.violations


def test_all_violations_are_reported(vocabulary):
    result = check("SOP-ZZZ-01 says 99 km/h is fine, guaranteed.", vocabulary, cited=["SOP-ZZZ-01"])
    assert result.violations[:3] == [
        "selected SOP SOP-ACT-01 is not cited",
        "cites SOP SOP-ZZZ-01 that was not selected",
        "mentions SOP SOP-ZZZ-01 that was not selected",
    ]
    assert any(v.startswith("unsupported number 99 ") for v in result.violations)
    assert "banned phrase 'guaranteed'" in result.violations


def test_missing_inputs_fail_closed(vocabulary):
    assert not verify_draft(None, COMPOSITION, vocabulary=vocabulary).passed
    assert not verify_draft(ComposedAnswer(text="x", cited_sop_ids=["SOP-ACT-01"]), None, vocabulary=vocabulary).passed
    result = verify_draft(None, COMPOSITION, vocabulary=vocabulary, composer_error="boom")
    assert "composer produced no valid answer (boom)" in result.violations
