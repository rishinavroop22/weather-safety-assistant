"""The shipped SOPs behave as designed on realistic weather facts.

These tests name SOP ids on purpose: they check the policy content, while the
engine code itself never refers to any SOP.
"""

from app.policy import QueryProfile, match_sops


def ids(result):
    return [m.id for m in result.matched]


def test_dangerous_heat_for_a_run(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["running"]), weather({"wx.max.apparent_temperature": 41.3}))
    assert ids(result) == ["SOP-ACT-01"]  # the moderate heat SOP stops at < 40
    assert result.primary.evidence == ["wx.max.apparent_temperature = 41.3, meets >= 40"]


def test_elevated_heat_for_a_run(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["running"]), weather({"wx.max.apparent_temperature": 35.0}))
    assert ids(result) == ["SOP-ACT-02"]
    assert result.primary.severity == "moderate"


def test_scooter_in_heavy_rain(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["two_wheeler_ride"]), weather({"wx.max.precipitation": 6.2}))
    assert result.primary.id == "SOP-TRV-01"


def test_dense_fog_for_a_drive(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["driving"]), weather({"wx.min.visibility": 150.0}))
    assert result.primary.id == "SOP-TRV-02"


def test_gusts_on_a_hike(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["hiking"]), weather({"wx.max.wind_gusts_10m": 52.0}))
    assert result.primary.id == "SOP-ACT-03"


def test_kids_and_grandmother_at_the_beach_multiple_match(policy, weather):
    profile = QueryProfile(activities=["beach_outing"], groups=["children", "elderly"])
    facts = weather({"wx.max.apparent_temperature": 38.0, "wx.max.temperature_2m": 33.0, "wx.max.uv_index": 10.0})

    result = match_sops(policy, profile, facts)

    assert ids(result) == ["SOP-VUL-02", "SOP-VUL-01"]  # high beats moderate
    assert result.primary.id == "SOP-VUL-02"
    assert [m.id for m in result.secondary] == ["SOP-VUL-01"]


def test_thunderstorm_outranks_activity_heat(policy, weather):
    facts = weather({"wx.codes": [3, 95], "wx.max.apparent_temperature": 41.0})
    result = match_sops(policy, QueryProfile(activities=["running"]), facts)
    assert ids(result) == ["SOP-GEN-01", "SOP-ACT-01"]


def test_same_severity_tie_uses_scope_then_specificity_then_id(policy, weather):
    # Cycling in heavy rain AND fog AND a rain system: three high-severity SOPs.
    facts = weather({"wx.max.precipitation": 6.0, "wx.min.visibility": 150.0, "wx.sum.precipitation": 45.0})
    result = match_sops(policy, QueryProfile(activities=["cycling"]), facts)
    # GEN-02 is global -> first. TRV-01 and TRV-02 tie on specificity (2) -> id order.
    assert ids(result) == ["SOP-GEN-02", "SOP-TRV-01", "SOP-TRV-02"]


def test_rain_system_compound_branch(policy, weather):
    walker = QueryProfile(activities=["walking"])
    assert match_sops(policy, walker, weather({"wx.sum.precipitation": 18.0, "wx.max.wind_gusts_10m": 42.0})).primary.id == "SOP-GEN-02"
    assert match_sops(policy, walker, weather({"wx.sum.precipitation": 45.0})).primary.id == "SOP-GEN-02"
    # Moderate rain alone is not enough.
    assert match_sops(policy, walker, weather({"wx.sum.precipitation": 18.0})).primary.id == "SOP-BASE-01"


def test_pets_via_group_or_dog_walk_activity(policy, weather):
    facts = weather({"wx.max.temperature_2m": 34.0, "wx.max.uv_index": 7.0})
    assert match_sops(policy, QueryProfile(activities=["dog_walk"]), facts).primary.id == "SOP-VUL-03"
    assert match_sops(policy, QueryProfile(activities=["walking"], groups=["pets"]), facts).primary.id == "SOP-VUL-03"
    assert match_sops(policy, QueryProfile(activities=["walking"]), facts).primary.id == "SOP-BASE-01"


def test_fuzzy_picnic_with_two_mild_factors(policy, weather):
    facts = weather({"wx.max.precipitation_probability": 55.0, "wx.max.wind_speed_10m": 28.0})
    result = match_sops(policy, QueryProfile(activities=["picnic_outing"]), facts)
    assert result.primary.id == "SOP-LEI-01"
    assert result.primary.severity == "low"
    assert result.primary.evidence == [
        "wx.max.precipitation_probability = 55, meets >= 40",
        "wx.max.wind_speed_10m = 28, meets >= 25",
    ]


def test_fuzzy_picnic_with_one_mild_factor_falls_back(policy, weather):
    facts = weather({"wx.max.wind_speed_10m": 28.0})
    result = match_sops(policy, QueryProfile(activities=["picnic_outing"]), facts)
    assert result.primary.id == "SOP-BASE-01"
    assert result.fallback_used


def test_calm_day_for_a_known_activity_uses_baseline(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["cycling"]), weather())
    assert ids(result) == ["SOP-BASE-01"]
    assert result.fallback_used


def test_unrecognised_activity_on_a_calm_day_has_no_guidance(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["other_outdoor"]), weather())
    assert result.primary is None
    assert not result.fallback_used


def test_unrecognised_activity_still_gets_global_hazards(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["other_outdoor"]), weather({"wx.codes": [96]}))
    assert result.primary.id == "SOP-GEN-01"


def test_unsupported_concern_has_no_guidance(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["running"], concerns=["air_quality"]), weather())
    assert result.primary is None


def test_not_an_outdoor_question_matches_nothing(policy, weather):
    result = match_sops(policy, QueryProfile(activities=[]), weather({"wx.codes": [95]}))
    assert result.matched == [] and result.unevaluable == []


def test_missing_visibility_blocks_the_baseline_answer(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["cycling"]), weather({"wx.min.visibility": None}))
    assert result.primary is None
    assert not result.fallback_used
    assert [u.id for u in result.unevaluable] == ["SOP-TRV-02"]


def test_missing_visibility_does_not_affect_unrelated_activity(policy, weather):
    result = match_sops(policy, QueryProfile(activities=["running"]), weather({"wx.min.visibility": None}))
    assert result.primary.id == "SOP-BASE-01"
