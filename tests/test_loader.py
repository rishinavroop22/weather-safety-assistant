"""SOP validation and loading: the shipped policy is valid, bad policy fails clearly."""

import os
from pathlib import Path
from typing import Any

import pytest

from app.policy import PolicyError, PolicyStore, QueryProfile, load_policy, match_sops, required_weather_variables
from app.policy.models import AtLeast, iter_leaves
from tests.conftest import POLICY_DIR, minimal_sop, write_sop

# ---------------------------------------------------------------------------
# The shipped policy
# ---------------------------------------------------------------------------


def test_shipped_policy_is_valid(policy):
    assert len(policy.sops) == 12
    assert len(set(policy.ids)) == 12


def test_shipped_policy_meets_assignment_requirements(policy):
    categories = {sop.category for sop in policy.sops if not sop.fallback}
    severities = {sop.severity for sop in policy.sops}
    assert len(categories) >= 3
    assert len(severities) >= 3
    # At least one fuzzy, multi-factor SOP (not a single threshold).
    assert any(isinstance(sop.when, AtLeast) for sop in policy.sops)


def test_exactly_one_fallback_sop_is_shipped(policy):
    assert sum(sop.fallback for sop in policy.sops) == 1


def test_required_weather_variables_come_from_sops(policy):
    assert required_weather_variables(policy) == [
        "apparent_temperature",
        "precipitation",
        "precipitation_probability",
        "relative_humidity_2m",
        "temperature_2m",
        "uv_index",
        "visibility",
        "weather_code",
        "wind_gusts_10m",
        "wind_speed_10m",
    ]


def test_every_fact_used_by_shipped_sops_is_valid(policy):
    # Validation already guarantees this; the test documents the fact namespace.
    for sop in policy.sops:
        for leaf in iter_leaves(sop.when):
            assert leaf.fact.startswith(("wx.", "intent."))


# ---------------------------------------------------------------------------
# Invalid SOPs fail with clear, located messages
# ---------------------------------------------------------------------------

LEAF = {"fact": "wx.max.uv_index", "op": "gte", "value": 7}


def _when(leaf: dict[str, Any]) -> dict[str, Any]:
    return {"when": {"all": [leaf]}}


INVALID_SOPS = [
    pytest.param(_when({**LEAF, "op": "greater_than"}), "when.all.0.op: Input should be", id="unknown-operator"),
    pytest.param(_when({**LEAF, "fact": "wx.max.snow_depth"}), "unknown weather variable 'snow_depth'", id="unknown-weather-variable"),
    pytest.param(_when({**LEAF, "fact": "wx.avg.uv_index"}), "unknown aggregate 'avg'", id="unknown-aggregate"),
    pytest.param(_when({**LEAF, "fact": "weather.uv"}), "unknown fact 'weather.uv'", id="unknown-fact"),
    pytest.param(_when({**LEAF, "value": [7, 8]}), "operator 'gte' requires a number", id="numeric-op-with-list"),
    pytest.param(_when({**LEAF, "value": True}), "operator 'gte' requires a number", id="numeric-op-with-bool"),
    pytest.param(_when({**LEAF, "op": "intersects", "value": [7]}), "cannot be used with numeric fact", id="list-op-on-number"),
    pytest.param(_when({"fact": "intent.groups", "op": "gte", "value": 1}), "cannot be used with list fact", id="number-op-on-list"),
    pytest.param(_when({"fact": "intent.groups", "op": "intersects", "value": ["aliens"]}), "unknown tag(s) ['aliens']", id="unknown-intent-tag"),
    pytest.param(_when({"fact": "wx.codes", "op": "intersects", "value": ["storm"]}), "integer weather codes", id="non-integer-weather-code"),
    pytest.param(_when({"fact": "wx.max.uv_index", "op": "gte"}), "when.all.0.value: Field required", id="missing-value"),
    pytest.param({"when": {"all": [LEAF], "any": [LEAF]}}, "each condition must be exactly one of", id="two-combinators"),
    pytest.param({"when": {"all": []}}, "when.all: List should have at least 1 item", id="empty-all"),
    pytest.param({"when": {"at_least": {"n": 3, "of": [LEAF]}}}, "n=3 is larger than the number of conditions", id="at-least-n-too-big"),
    pytest.param({"when": {"not": {**LEAF, "fact": "wx.max.nope"}}}, "when.not: unknown weather variable", id="invalid-leaf-inside-not"),
    pytest.param({"severity": "extreme"}, "unknown severity 'extreme'", id="unknown-severity"),
    pytest.param({"scope": "worldwide"}, "scope: Input should be 'global' or 'activity'", id="unknown-scope"),
    pytest.param({"applies_to": {"activities": ["skydiving"]}}, "unknown activity tag(s) ['skydiving']", id="unknown-activity"),
    pytest.param({"applies_to": {"activities": ["running"], "groups": ["teens"]}}, "unknown group tag(s) ['teens']", id="unknown-group"),
    pytest.param({"scope": "global"}, "scope 'global' requires applies_to.activities", id="global-without-pseudo-tag"),
    pytest.param({"applies_to": {"activities": ["any_outdoor"]}}, "can only be used with scope 'global'", id="pseudo-tag-on-activity-scope"),
    pytest.param({"guidance": []}, "guidance: List should have at least 1 item", id="empty-guidance"),
    pytest.param({"cite": ["wx.codes"]}, "cite entry 'wx.codes' must be a numeric fact", id="cite-non-numeric"),
    pytest.param({"cite": ["wx.max.nope"]}, "cite: unknown weather variable 'nope'", id="cite-unknown-variable"),
    pytest.param({"guidence": ["typo"]}, "guidence: Extra inputs are not permitted", id="typo-in-key"),
    pytest.param({"version": 0}, "version: Input should be greater than or equal to 1", id="bad-version"),
]


@pytest.mark.parametrize(("changes", "expected_message"), INVALID_SOPS)
def test_invalid_sop_fails_with_clear_message(policy_dir, changes, expected_message):
    root = policy_dir(minimal_sop(**changes))

    with pytest.raises(PolicyError) as excinfo:
        load_policy(root)

    assert "SOP-TEST-01.yaml" in str(excinfo.value)
    assert expected_message in str(excinfo.value)


def test_missing_required_field_is_reported(policy_dir):
    sop = minimal_sop()
    del sop["when"]
    with pytest.raises(PolicyError, match=r"SOP-TEST-01.yaml: when: Field required"):
        load_policy(policy_dir(sop))


def test_file_name_must_match_id(policy_dir):
    root = policy_dir()
    write_sop(root, minimal_sop(), file_name="my-new-sop.yaml")
    with pytest.raises(PolicyError, match=r"my-new-sop.yaml: file name does not match id 'SOP-TEST-01'"):
        load_policy(root)


def test_duplicate_ids_are_rejected(policy_dir):
    root = policy_dir(minimal_sop())
    write_sop(root, minimal_sop(), file_name="SOP-TEST-01.yml")
    with pytest.raises(PolicyError, match="duplicate SOP id 'SOP-TEST-01'"):
        load_policy(root)


def test_malformed_yaml_is_reported(policy_dir):
    root = policy_dir(("SOP-TEST-01.yaml", "id: SOP-TEST-01\nwhen: [unclosed\n"))
    with pytest.raises(PolicyError, match="SOP-TEST-01.yaml: invalid YAML"):
        load_policy(root)


def test_non_mapping_yaml_is_reported(policy_dir):
    root = policy_dir(("SOP-TEST-01.yaml", "- just\n- a list\n"))
    with pytest.raises(PolicyError, match="expected a mapping at the top level"):
        load_policy(root)


def test_all_problems_are_reported_together(policy_dir):
    root = policy_dir(
        minimal_sop("SOP-TEST-01", severity="extreme"),
        minimal_sop("SOP-TEST-02", applies_to={"activities": ["skydiving"]}),
    )
    with pytest.raises(PolicyError) as excinfo:
        load_policy(root)
    assert len(excinfo.value.problems) == 2
    assert "SOP-TEST-01.yaml" in str(excinfo.value)
    assert "SOP-TEST-02.yaml" in str(excinfo.value)


def test_empty_sop_folder_is_rejected(policy_dir):
    with pytest.raises(PolicyError, match="no SOP files found"):
        load_policy(policy_dir())


def test_invalid_vocabulary_is_rejected(policy_dir):
    root = policy_dir(minimal_sop())
    vocab_path = root / "vocabulary.yaml"
    vocab_path.write_text(vocab_path.read_text(encoding="utf-8").replace("  other_outdoor:", "  other_thing:"), encoding="utf-8")
    with pytest.raises(PolicyError, match="activities must include 'other_outdoor'"):
        load_policy(root)


# ---------------------------------------------------------------------------
# Adding an SOP needs no code change
# ---------------------------------------------------------------------------


def test_new_sop_file_is_matched_without_code_changes(policy_dir, weather):
    cold_run = minimal_sop(
        "SOP-ACT-04",
        title="Cold start for an early run",
        severity="low",
        applies_to={"activities": ["running"], "groups": []},
        when={"all": [{"fact": "wx.min.apparent_temperature", "op": "lte", "value": 8}]},
        guidance=["Warm up indoors and wear layers."],
        cite=["wx.min.apparent_temperature"],
    )
    root = policy_dir(cold_run, include_shipped=True)

    policy = load_policy(root)
    result = match_sops(policy, QueryProfile(activities=["running"]), weather({"wx.min.apparent_temperature": 5.0}))

    assert len(policy.sops) == 13
    assert result.primary is not None and result.primary.id == "SOP-ACT-04"
    assert "apparent_temperature" in required_weather_variables(policy)


# ---------------------------------------------------------------------------
# PolicyStore: reload on change, keep last good policy on a broken edit
# ---------------------------------------------------------------------------


def _touch_later(path: Path) -> None:
    """Make sure the mtime changes even on file systems with coarse timestamps."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


def test_policy_store_picks_up_new_sop(policy_dir):
    root = policy_dir(minimal_sop("SOP-TEST-01"))
    store = PolicyStore(root)
    assert store.get().ids == ["SOP-TEST-01"]

    write_sop(root, minimal_sop("SOP-TEST-02"))
    assert store.get().ids == ["SOP-TEST-01", "SOP-TEST-02"]


def test_policy_store_keeps_last_good_policy_on_broken_edit(policy_dir):
    root = policy_dir(minimal_sop("SOP-TEST-01"))
    store = PolicyStore(root)
    good = store.get()

    broken = write_sop(root, minimal_sop("SOP-TEST-01", severity="extreme"))
    _touch_later(broken)
    assert store.get() is good
    assert store.last_error is not None and "unknown severity" in str(store.last_error)

    fixed = write_sop(root, minimal_sop("SOP-TEST-01", severity="high"))
    _touch_later(fixed)
    assert store.get().get("SOP-TEST-01").severity == "high"
    assert store.last_error is None


def test_policy_store_raises_if_first_load_fails(policy_dir):
    root = policy_dir(minimal_sop(severity="extreme"))
    with pytest.raises(PolicyError):
        PolicyStore(root).get()


def test_policy_store_reuses_cached_policy_when_nothing_changed():
    store = PolicyStore(POLICY_DIR)
    assert store.get() is store.get()
