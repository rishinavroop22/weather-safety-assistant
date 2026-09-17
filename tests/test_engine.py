"""Generic engine behaviour, tested with synthetic conditions and SOPs (no shipped SOP ids)."""

import random
from typing import Any

import pytest

from app.policy import (
    PolicySet,
    QueryProfile,
    SOPMatch,
    Truth,
    check_applicability,
    evaluate_condition,
    match_sops,
    parse_condition,
    rank_matches,
    specificity,
)
from app.policy.models import SOP
from tests.conftest import minimal_sop

# Leaves with a known verdict against FACTS.
FACTS = {"wx.max.uv_index": 9.0, "wx.max.wind_gusts_10m": 20.0, "wx.codes": [3, 95]}
T = {"fact": "wx.max.uv_index", "op": "gte", "value": 7}          # TRUE
T2 = {"fact": "wx.codes", "op": "intersects", "value": [95, 96]}   # TRUE
F = {"fact": "wx.max.wind_gusts_10m", "op": "gte", "value": 45}    # FALSE
U = {"fact": "wx.min.visibility", "op": "lt", "value": 200}        # UNKNOWN (fact missing)


@pytest.fixture
def evaluate(vocabulary):
    def run(raw: dict[str, Any], facts: dict[str, Any] | None = None):
        return evaluate_condition(parse_condition(raw, vocabulary), FACTS if facts is None else facts)

    return run


# ---------------------------------------------------------------------------
# Leaf operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "actual", "value", "expected"),
    [
        ("gt", 10.0, 9, Truth.TRUE),
        ("gt", 9.0, 9, Truth.FALSE),
        ("gte", 9.0, 9, Truth.TRUE),
        ("gte", 8.9, 9, Truth.FALSE),
        ("lt", 199.0, 200, Truth.TRUE),
        ("lt", 200.0, 200, Truth.FALSE),
        ("lte", 200.0, 200, Truth.TRUE),
        ("lte", 200.1, 200, Truth.FALSE),
        ("eq", 5, 5, Truth.TRUE),
        ("eq", 5.5, 5, Truth.FALSE),
        ("in", 3, [1, 2, 3], Truth.TRUE),
        ("in", 4, [1, 2, 3], Truth.FALSE),
    ],
)
def test_number_operators(evaluate, op, actual, value, expected):
    result = evaluate({"fact": "wx.max.uv_index", "op": op, "value": value}, {"wx.max.uv_index": actual})
    assert result.truth is expected


@pytest.mark.parametrize(
    ("op", "actual", "expected"),
    [
        ("intersects", [3, 95], Truth.TRUE),
        ("intersects", [1, 2], Truth.FALSE),
        ("intersects", [], Truth.FALSE),
        ("not_intersects", [1, 2], Truth.TRUE),
        ("not_intersects", [3, 95], Truth.FALSE),
        ("not_intersects", [], Truth.TRUE),
    ],
)
def test_list_operators(evaluate, op, actual, expected):
    result = evaluate({"fact": "wx.codes", "op": op, "value": [95, 96, 99]}, {"wx.codes": actual})
    assert result.truth is expected


@pytest.mark.parametrize("facts", [{}, {"wx.max.uv_index": None}, {"wx.max.uv_index": float("nan")}], ids=["missing", "none", "nan"])
def test_missing_null_or_nan_fact_is_unknown_not_false(evaluate, facts):
    result = evaluate(T, facts)
    assert result.truth is Truth.UNKNOWN
    assert result.unknown_facts == ("wx.max.uv_index",)
    assert result.evidence == ()


def test_wrong_fact_type_is_an_error_not_a_verdict(evaluate):
    with pytest.raises(TypeError, match="must be a number"):
        evaluate(T, {"wx.max.uv_index": "9"})
    with pytest.raises(TypeError, match="must be a list"):
        evaluate(T2, {"wx.codes": 95})


# ---------------------------------------------------------------------------
# Three-valued combinators
# ---------------------------------------------------------------------------

LEAVES = {"T": T, "F": F, "U": U}


@pytest.mark.parametrize(
    ("children", "expected"),
    [("TT", "T"), ("TF", "F"), ("TU", "U"), ("FU", "F"), ("UU", "U"), ("FF", "F")],
)
def test_all_truth_table(evaluate, children, expected):
    result = evaluate({"all": [LEAVES[c] for c in children]})
    assert result.truth is {"T": Truth.TRUE, "F": Truth.FALSE, "U": Truth.UNKNOWN}[expected]


@pytest.mark.parametrize(
    ("children", "expected"),
    [("TT", "T"), ("TF", "T"), ("TU", "T"), ("FU", "U"), ("UU", "U"), ("FF", "F")],
)
def test_any_truth_table(evaluate, children, expected):
    result = evaluate({"any": [LEAVES[c] for c in children]})
    assert result.truth is {"T": Truth.TRUE, "F": Truth.FALSE, "U": Truth.UNKNOWN}[expected]


@pytest.mark.parametrize(("child", "expected"), [("T", Truth.FALSE), ("F", Truth.TRUE), ("U", Truth.UNKNOWN)])
def test_not_truth_table(evaluate, child, expected):
    assert evaluate({"not": LEAVES[child]}).truth is expected


@pytest.mark.parametrize(
    ("n", "children", "expected"),
    [
        (2, "TTF", Truth.TRUE),
        (2, "TFF", Truth.FALSE),
        (2, "TUF", Truth.UNKNOWN),   # the unknown could tip it either way
        (2, "TUU", Truth.UNKNOWN),
        (3, "TTU", Truth.UNKNOWN),
        (2, "FFU", Truth.FALSE),     # even if the unknown were true, only 1 of 2
        (1, "FFT", Truth.TRUE),
    ],
)
def test_at_least_truth_table(evaluate, n, children, expected):
    result = evaluate({"at_least": {"n": n, "of": [LEAVES[c] for c in children]}})
    assert result.truth is expected


def test_nested_conditions(evaluate):
    raw = {"all": [T, {"any": [F, {"not": F}]}, {"at_least": {"n": 1, "of": [U, T2]}}]}
    assert evaluate(raw).truth is Truth.TRUE


def test_unknown_facts_are_collected_even_when_verdict_is_decided(evaluate):
    result = evaluate({"any": [T, U]})
    assert result.truth is Truth.TRUE
    assert result.unknown_facts == ("wx.min.visibility",)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_for_true_all_lists_every_condition(evaluate):
    result = evaluate({"all": [T, T2]})
    assert result.evidence == (
        "wx.max.uv_index = 9, meets >= 7",
        "wx.codes = [3, 95], meets includes any of [95, 96]",
    )


def test_evidence_for_true_any_lists_only_true_children(evaluate):
    result = evaluate({"any": [F, T]})
    assert result.evidence == ("wx.max.uv_index = 9, meets >= 7",)


def test_evidence_for_at_least_lists_the_factors_that_fired(evaluate):
    result = evaluate({"at_least": {"n": 2, "of": [T, F, T2]}})
    assert result.truth is Truth.TRUE
    assert len(result.evidence) == 2
    assert all("meets" in line and "does not" not in line for line in result.evidence)


def test_evidence_for_not_explains_the_negated_fact(evaluate):
    result = evaluate({"not": F})
    assert result.truth is Truth.TRUE
    assert result.evidence == ("wx.max.wind_gusts_10m = 20, does not meet >= 45",)


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def _sop(vocabulary, **fields: Any) -> SOP:
    return SOP.model_validate(minimal_sop(**fields), context={"vocabulary": vocabulary})


def test_no_activity_means_nothing_applies(vocabulary):
    sop = _sop(vocabulary, scope="global", applies_to={"activities": ["any_outdoor"]})
    assert not check_applicability(sop, QueryProfile(activities=[])).applies


def test_activity_scope_requires_overlap(vocabulary):
    sop = _sop(vocabulary, applies_to={"activities": ["running", "cycling"]})
    assert check_applicability(sop, QueryProfile(activities=["cycling"])).applies
    assert not check_applicability(sop, QueryProfile(activities=["driving"])).applies


def test_any_outdoor_includes_unrecognised_activities(vocabulary):
    sop = _sop(vocabulary, scope="global", applies_to={"activities": ["any_outdoor"]})
    assert check_applicability(sop, QueryProfile(activities=["other_outdoor"])).applies


def test_any_known_activity_excludes_only_unrecognised(vocabulary):
    sop = _sop(vocabulary, scope="global", applies_to={"activities": ["any_known_activity"]})
    assert not check_applicability(sop, QueryProfile(activities=["other_outdoor"])).applies
    assert check_applicability(sop, QueryProfile(activities=["other_outdoor", "walking"])).applies


def test_groups_are_an_additional_requirement(vocabulary):
    sop = _sop(vocabulary, scope="global", applies_to={"activities": ["any_outdoor"], "groups": ["children"]})
    assert check_applicability(sop, QueryProfile(activities=["walking"], groups=["children", "pets"])).applies
    assert not check_applicability(sop, QueryProfile(activities=["walking"], groups=["elderly"])).applies


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _match(sop_id: str, rank: int, scope: str = "activity", spec: int = 1) -> SOPMatch:
    return SOPMatch(
        id=sop_id, version=1, title=sop_id, category="test", severity=str(rank), severity_rank=rank,
        scope=scope, specificity=spec, fallback=False, guidance=["g"], cite=[], evidence=[],
    )


def test_ranking_severity_first():
    ranked = rank_matches([_match("A", 2, "global", 9), _match("B", 3, "activity", 1)])
    assert [m.id for m in ranked] == ["B", "A"]


def test_ranking_global_scope_breaks_severity_ties():
    ranked = rank_matches([_match("A", 3, "activity", 9), _match("B", 3, "global", 1)])
    assert [m.id for m in ranked] == ["B", "A"]


def test_ranking_specificity_breaks_scope_ties():
    ranked = rank_matches([_match("A", 3, "activity", 1), _match("B", 3, "activity", 4)])
    assert [m.id for m in ranked] == ["B", "A"]


def test_ranking_id_is_the_final_tie_breaker():
    ranked = rank_matches([_match("SOP-B", 3), _match("SOP-A", 3)])
    assert [m.id for m in ranked] == ["SOP-A", "SOP-B"]


def test_ranking_does_not_depend_on_input_order():
    matches = [_match(f"SOP-{i}", i % 3, "global" if i % 2 else "activity", i % 4) for i in range(12)]
    expected = [m.id for m in rank_matches(matches)]
    rng = random.Random(7)
    for _ in range(20):
        rng.shuffle(matches)
        assert [m.id for m in rank_matches(matches)] == expected


def test_specificity_counts_constraints_and_leaves(vocabulary):
    sop = _sop(
        vocabulary,
        applies_to={"activities": ["running"], "groups": ["children"]},
        when={"all": [T, {"any": [F, U]}]},
    )
    assert specificity(sop) == 1 + 1 + 3


# ---------------------------------------------------------------------------
# Matching and fallback
# ---------------------------------------------------------------------------


def _policy(vocabulary, *sops: dict[str, Any]) -> PolicySet:
    return PolicySet(vocabulary, tuple(SOP.model_validate(s, context={"vocabulary": vocabulary}) for s in sops))


UV_SOP = minimal_sop("SOP-UV", when={"all": [T]})
FOG_SOP = minimal_sop("SOP-FOG", severity="high", when={"all": [U]})
BASELINE = minimal_sop(
    "SOP-BASE", severity="info", scope="global", fallback=True,
    applies_to={"activities": ["any_known_activity"]},
    when={"all": [{"fact": "intent.concerns", "op": "not_intersects", "value": ["pollen"]}]},
)
RUNNER = QueryProfile(activities=["running"])


def test_fallback_is_not_used_when_a_regular_sop_matches(vocabulary):
    result = match_sops(_policy(vocabulary, UV_SOP, BASELINE), RUNNER, {"wx.max.uv_index": 9.0})
    assert [m.id for m in result.matched] == ["SOP-UV"]
    assert not result.fallback_used


def test_fallback_is_used_when_nothing_matches_and_nothing_is_unknown(vocabulary):
    result = match_sops(_policy(vocabulary, UV_SOP, BASELINE), RUNNER, {"wx.max.uv_index": 3.0})
    assert result.primary is not None and result.primary.id == "SOP-BASE"
    assert result.fallback_used


def test_fallback_is_suppressed_when_a_regular_sop_is_unknown(vocabulary):
    result = match_sops(_policy(vocabulary, UV_SOP, FOG_SOP, BASELINE), RUNNER, {"wx.max.uv_index": 3.0})
    assert result.matched == []
    assert result.primary is None
    assert not result.fallback_used
    assert [(u.id, u.unknown_facts) for u in result.unevaluable] == [("SOP-FOG", ["wx.min.visibility"])]


def test_unknown_sop_is_reported_alongside_a_match(vocabulary):
    result = match_sops(_policy(vocabulary, UV_SOP, FOG_SOP), RUNNER, {"wx.max.uv_index": 9.0})
    assert result.primary.id == "SOP-UV"
    assert [u.id for u in result.unevaluable] == ["SOP-FOG"]


def test_inapplicable_sop_is_never_unknown(vocabulary):
    result = match_sops(_policy(vocabulary, FOG_SOP), QueryProfile(activities=["driving"]), {})
    assert result.unevaluable == []


def test_fallback_condition_can_still_reject(vocabulary):
    profile = QueryProfile(activities=["running"], concerns=["pollen"])
    result = match_sops(_policy(vocabulary, UV_SOP, BASELINE), profile, {"wx.max.uv_index": 3.0})
    assert result.primary is None and not result.fallback_used


def test_multiple_fallbacks_are_ranked_deterministically(vocabulary):
    second = {**BASELINE, "id": "SOP-BASE-2", "severity": "low"}
    result = match_sops(_policy(vocabulary, BASELINE, second), RUNNER, {})
    assert [m.id for m in result.matched] == ["SOP-BASE-2", "SOP-BASE"]


def test_secondary_matches_are_capped(vocabulary):
    sops = [minimal_sop(f"SOP-UV-{i}", when={"all": [T]}) for i in range(5)]
    result = match_sops(_policy(vocabulary, *sops), RUNNER, {"wx.max.uv_index": 9.0}, max_secondary=2)
    assert result.primary.id == "SOP-UV-0"
    assert [m.id for m in result.secondary] == ["SOP-UV-1", "SOP-UV-2"]
    assert len(result.matched) == 5


def test_match_records_evidence(vocabulary):
    result = match_sops(_policy(vocabulary, UV_SOP), RUNNER, {"wx.max.uv_index": 9.5})
    assert result.primary.evidence == ["wx.max.uv_index = 9.5, meets >= 7"]


def test_weather_facts_cannot_override_intent(vocabulary):
    with pytest.raises(ValueError, match="must not contain intent facts"):
        match_sops(_policy(vocabulary, UV_SOP), RUNNER, {"intent.groups": ["children"]})
