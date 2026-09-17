"""Deterministic SOP matching: applicability, condition evaluation, ranking.

Pipeline for one question (``match_sops``):

1. Applicability - does the SOP's ``applies_to`` fit the user's activities/groups?
2. Evaluation    - evaluate ``when`` with three-valued logic (TRUE/FALSE/UNKNOWN).
   A missing or null fact is UNKNOWN, never FALSE.
3. Fallback      - fallback SOPs are considered only if no regular SOP matched
   AND no applicable regular SOP was UNKNOWN (we can't say "no thresholds
   crossed" if we couldn't check some of them).
4. Ranking       - sort by (severity desc, global scope first, specificity desc, id asc).

No function here refers to a specific SOP id or threshold.
"""

import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.policy.loader import PolicySet
from app.policy.models import (
    ANY_KNOWN_ACTIVITY,
    INTENT_FACTS,
    LIST_OPERATORS,
    SOP,
    UNRECOGNIZED_ACTIVITY,
    AllOf,
    AnyOf,
    AtLeast,
    Condition,
    Leaf,
    NotOf,
    is_number,
    iter_leaves,
)

Facts = Mapping[str, Any]
"""Flat fact namespace, e.g. {"wx.max.uv_index": 9.1, "wx.codes": [3, 95], "intent.groups": ["children"]}."""


# ---------------------------------------------------------------------------
# Three-valued condition evaluation
# ---------------------------------------------------------------------------


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating a condition.

    ``evidence`` holds true statements about the facts that explain the verdict
    (empty when UNKNOWN). ``unknown_facts`` lists facts that were missing/null
    anywhere in the tree, even if the verdict did not depend on them.
    """

    truth: Truth
    evidence: tuple[str, ...] = ()
    unknown_facts: tuple[str, ...] = ()


_NUMBER_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "in": lambda actual, expected: actual in expected,
}

_OPERATOR_SYMBOLS = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "eq": "==",
    "in": "in",
    "intersects": "includes any of",
    "not_intersects": "includes none of",
}


def evaluate_condition(condition: Condition, facts: Facts) -> EvalResult:
    """Evaluate a condition tree against facts using Kleene three-valued logic.

    * ``all`` is TRUE if every child is TRUE, FALSE if any child is FALSE, else UNKNOWN.
    * ``any`` is TRUE if any child is TRUE, FALSE if every child is FALSE, else UNKNOWN.
    * ``not`` swaps TRUE/FALSE and keeps UNKNOWN.
    * ``at_least n`` is TRUE if >= n children are TRUE, FALSE if even counting
      every UNKNOWN child as TRUE would not reach n, else UNKNOWN.

    ``all`` and ``any`` are ``at_least`` with n = len(children) and n = 1.

    Raises:
        TypeError: if a fact has the wrong type for its operator (an upstream bug,
            not a missing value).
    """
    if isinstance(condition, Leaf):
        return _evaluate_leaf(condition, facts)
    if isinstance(condition, NotOf):
        return _negate(evaluate_condition(condition.negated, facts))
    if isinstance(condition, AllOf):
        results = [evaluate_condition(c, facts) for c in condition.all]
        return _at_least(len(results), results)
    if isinstance(condition, AnyOf):
        results = [evaluate_condition(c, facts) for c in condition.any]
        return _at_least(1, results)
    if isinstance(condition, AtLeast):
        results = [evaluate_condition(c, facts) for c in condition.at_least.of]
        return _at_least(condition.at_least.n, results)
    raise TypeError(f"unsupported condition type: {type(condition).__name__}")


def _evaluate_leaf(leaf: Leaf, facts: Facts) -> EvalResult:
    actual = facts.get(leaf.fact)
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return EvalResult(Truth.UNKNOWN, unknown_facts=(leaf.fact,))

    if leaf.op in LIST_OPERATORS:
        if not isinstance(actual, (list, tuple, set, frozenset)):
            raise TypeError(f"fact '{leaf.fact}' must be a list for operator '{leaf.op}', got {actual!r}")
        overlaps = bool(set(actual) & set(leaf.value))
        holds = overlaps if leaf.op == "intersects" else not overlaps
    else:
        if not is_number(actual):
            raise TypeError(f"fact '{leaf.fact}' must be a number for operator '{leaf.op}', got {actual!r}")
        holds = _NUMBER_COMPARISONS[leaf.op](actual, leaf.value)

    verdict = "meets" if holds else "does not meet"
    statement = f"{leaf.fact} = {_format(actual)}, {verdict} {_OPERATOR_SYMBOLS[leaf.op]} {_format(leaf.value)}"
    return EvalResult(Truth.TRUE if holds else Truth.FALSE, evidence=(statement,))


def _negate(result: EvalResult) -> EvalResult:
    flipped = {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE, Truth.UNKNOWN: Truth.UNKNOWN}
    return EvalResult(flipped[result.truth], result.evidence, result.unknown_facts)


def _at_least(n: int, results: Sequence[EvalResult]) -> EvalResult:
    unknown_facts = tuple(dict.fromkeys(f for r in results for f in r.unknown_facts))
    true = [r for r in results if r.truth is Truth.TRUE]
    false = [r for r in results if r.truth is Truth.FALSE]
    unknown_count = len(results) - len(true) - len(false)

    if len(true) >= n:
        return EvalResult(Truth.TRUE, _join_evidence(true), unknown_facts)
    if len(true) + unknown_count >= n:
        return EvalResult(Truth.UNKNOWN, (), unknown_facts)
    return EvalResult(Truth.FALSE, _join_evidence(false), unknown_facts)


def _join_evidence(results: Iterable[EvalResult]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(line for r in results for line in r.evidence))


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (set, frozenset)):
        value = sorted(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


class QueryProfile(BaseModel):
    """The parts of the user's question the policy engine needs (vocabulary tags)."""

    model_config = ConfigDict(frozen=True)

    activities: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    def as_facts(self) -> dict[str, list[str]]:
        """Expose the profile as ``intent.*`` facts for conditions."""
        return {fact: list(getattr(self, attr)) for fact, attr in INTENT_FACTS.items()}


@dataclass(frozen=True)
class Applicability:
    applies: bool
    reason: str


def check_applicability(sop: SOP, profile: QueryProfile) -> Applicability:
    """Decide whether an SOP is about this kind of question, before looking at weather.

    Rules:
      * No activities at all -> not an outdoor-activity question -> nothing applies.
      * scope 'global' + any_outdoor        -> applies to any activity.
      * scope 'global' + any_known_activity -> needs an activity other than other_outdoor.
      * scope 'activity'                    -> needs an overlap with applies_to.activities.
      * non-empty applies_to.groups         -> additionally needs an overlap with profile.groups.
    """
    activities = set(profile.activities)
    if not activities:
        return Applicability(False, "question has no outdoor activity")

    targets = sop.applies_to.activities
    if sop.scope == "global":
        if targets == [ANY_KNOWN_ACTIVITY] and not activities - {UNRECOGNIZED_ACTIVITY}:
            return Applicability(False, "only unrecognised activities")
    elif not activities & set(targets):
        return Applicability(False, f"activities {sorted(activities)} not in {targets}")

    groups = sop.applies_to.groups
    if groups and not set(profile.groups) & set(groups):
        return Applicability(False, f"groups {sorted(profile.groups)} not in {groups}")
    return Applicability(True, "applies")


# ---------------------------------------------------------------------------
# Matching, fallback and ranking
# ---------------------------------------------------------------------------


class SOPMatch(BaseModel):
    """An SOP whose conditions were TRUE, with the evidence that made it TRUE."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: int
    title: str
    category: str
    severity: str
    severity_rank: int
    scope: str
    specificity: int
    fallback: bool
    guidance: list[str]
    cite: list[str]
    evidence: list[str]


class UnevaluableSOP(BaseModel):
    """An applicable SOP whose verdict was UNKNOWN because facts were missing."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    severity: str
    severity_rank: int
    unknown_facts: list[str]


class MatchResult(BaseModel):
    """Everything downstream nodes need to answer, cite and explain."""

    model_config = ConfigDict(frozen=True)

    matched: list[SOPMatch]            # all matches, ranked
    primary: SOPMatch | None           # matched[0]
    secondary: list[SOPMatch]          # next matches, capped at max_secondary
    unevaluable: list[UnevaluableSOP]  # ranked by severity, then id
    fallback_used: bool


def specificity(sop: SOP) -> int:
    """How narrowly an SOP is targeted: applies_to constraints + number of condition leaves."""
    constraints = (1 if sop.scope == "activity" else 0) + (1 if sop.applies_to.groups else 0)
    return constraints + sum(1 for _ in iter_leaves(sop.when))


def ranking_key(match: SOPMatch) -> tuple[int, int, int, str]:
    """Sort key: severity desc, global scope first, specificity desc, id asc."""
    return (-match.severity_rank, 0 if match.scope == "global" else 1, -match.specificity, match.id)


def rank_matches(matches: Iterable[SOPMatch]) -> list[SOPMatch]:
    """Deterministically order matches; the input order never affects the result."""
    return sorted(matches, key=ranking_key)


def match_sops(
    policy: PolicySet,
    profile: QueryProfile,
    weather_facts: Facts,
    *,
    max_secondary: int = 2,
) -> MatchResult:
    """Find, rank and explain the SOPs that apply to a question.

    Args:
        policy: validated policy set.
        profile: the user's activities, groups and concerns (vocabulary tags).
        weather_facts: ``wx.*`` facts for the requested window. Missing or None
            values are treated as UNKNOWN.
        max_secondary: how many matches after the primary to return in ``secondary``.

    Raises:
        ValueError: if ``weather_facts`` tries to supply ``intent.*`` facts.
    """
    if max_secondary < 0:
        raise ValueError("max_secondary must be >= 0")
    overlap = sorted(k for k in weather_facts if k in INTENT_FACTS)
    if overlap:
        raise ValueError(f"weather_facts must not contain intent facts: {overlap}")
    facts = {**weather_facts, **profile.as_facts()}

    regular = [sop for sop in policy.sops if not sop.fallback]
    fallbacks = [sop for sop in policy.sops if sop.fallback]

    matched, unevaluable = _evaluate_sops(regular, policy, profile, facts)
    fallback_used = False
    if not matched and not unevaluable:
        matched, fallback_unevaluable = _evaluate_sops(fallbacks, policy, profile, facts)
        unevaluable += fallback_unevaluable
        fallback_used = bool(matched)

    ranked = rank_matches(matched)
    return MatchResult(
        matched=ranked,
        primary=ranked[0] if ranked else None,
        secondary=ranked[1 : 1 + max_secondary],
        unevaluable=sorted(unevaluable, key=lambda u: (-u.severity_rank, u.id)),
        fallback_used=fallback_used,
    )


def _evaluate_sops(
    sops: Iterable[SOP], policy: PolicySet, profile: QueryProfile, facts: Facts
) -> tuple[list[SOPMatch], list[UnevaluableSOP]]:
    matched: list[SOPMatch] = []
    unevaluable: list[UnevaluableSOP] = []
    for sop in sops:
        if not check_applicability(sop, profile).applies:
            continue
        result = evaluate_condition(sop.when, facts)
        rank = policy.severity_rank(sop.severity)
        if result.truth is Truth.TRUE:
            matched.append(
                SOPMatch(
                    id=sop.id,
                    version=sop.version,
                    title=sop.title,
                    category=sop.category,
                    severity=sop.severity,
                    severity_rank=rank,
                    scope=sop.scope,
                    specificity=specificity(sop),
                    fallback=sop.fallback,
                    guidance=list(sop.guidance),
                    cite=list(sop.cite),
                    evidence=list(result.evidence),
                )
            )
        elif result.truth is Truth.UNKNOWN:
            unevaluable.append(
                UnevaluableSOP(
                    id=sop.id,
                    title=sop.title,
                    severity=sop.severity,
                    severity_rank=rank,
                    unknown_facts=list(result.unknown_facts),
                )
            )
    return matched, unevaluable
