"""Pydantic models for the policy layer.

* ``Vocabulary`` - the closed set of tags, weather variables and severities that
  SOPs may reference (``policy/vocabulary.yaml``).
* ``Condition``  - a small recursive condition language: a leaf
  ``{fact, op, value}`` or one of ``all`` / ``any`` / ``not`` / ``at_least``.
* ``SOP``        - one Standard Operating Procedure (``policy/sops/<id>.yaml``).

Validation happens in two layers:

1. Structural - types, required fields, and no unknown keys (``extra="forbid"``),
   so a typo such as ``guidence:`` fails loudly instead of being ignored.
2. Semantic - fact paths, tags and severities must exist in the vocabulary.
   The vocabulary is passed in through Pydantic's validation context:
   ``SOP.model_validate(data, context={"vocabulary": vocabulary})``.

Nothing in this module knows about any individual SOP.
"""

from collections.abc import Iterator
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

# ---------------------------------------------------------------------------
# Generic names shared by the loader and the engine
# ---------------------------------------------------------------------------

ANY_OUTDOOR = "any_outdoor"
"""Pseudo-activity for global SOPs: any outdoor activity, including unrecognised ones."""

ANY_KNOWN_ACTIVITY = "any_known_activity"
"""Pseudo-activity for global SOPs: at least one activity the vocabulary recognises."""

UNRECOGNIZED_ACTIVITY = "other_outdoor"
"""Vocabulary tag for an outdoor activity with no specific tag (e.g. scuba diving)."""

PSEUDO_ACTIVITIES = frozenset({ANY_OUTDOOR, ANY_KNOWN_ACTIVITY})

WEATHER_AGGREGATES = ("max", "min", "sum")
WEATHER_CODES_FACT = "wx.codes"
WEATHER_CODE_VARIABLE = "weather_code"
"""The Open-Meteo variable behind ``wx.codes``."""
INTENT_FACTS = {
    "intent.activities": "activities",
    "intent.groups": "groups",
    "intent.concerns": "concerns",
}

Operator = Literal["gt", "gte", "lt", "lte", "eq", "in", "intersects", "not_intersects"]
NUMBER_OPERATORS = frozenset({"gt", "gte", "lt", "lte", "eq", "in"})
LIST_OPERATORS = frozenset({"intersects", "not_intersects"})

FactKind = Literal["number", "tag_list", "code_list"]


def is_number(value: Any) -> bool:
    """True for int/float values (bool is excluded even though it subclasses int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class StrictModel(BaseModel):
    """Base model: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TagInfo(StrictModel):
    description: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)


class ConcernInfo(TagInfo):
    supported: bool


class WeatherVariable(StrictModel):
    unit: str
    description: str = Field(min_length=1)
    label: str | None = None
    """Short user-facing name, e.g. 'wind gusts'."""
    aggregates: list[str]
    """Aggregates that are meaningful over a time window (subset of max/min/sum)."""

    @model_validator(mode="after")
    def _check_aggregates(self) -> "WeatherVariable":
        unknown = [a for a in self.aggregates if a not in WEATHER_AGGREGATES]
        if unknown:
            raise ValueError(f"unknown aggregate(s) {unknown}; allowed: {list(WEATHER_AGGREGATES)}")
        if len(set(self.aggregates)) != len(self.aggregates):
            raise ValueError("aggregates must not contain duplicates")
        return self


class Vocabulary(StrictModel):
    """Closed vocabulary that SOPs (and later the intent extractor) must use."""

    version: int = Field(ge=1)
    severity_ranks: dict[str, int] = Field(min_length=1)
    activities: dict[str, TagInfo] = Field(min_length=1)
    groups: dict[str, TagInfo] = Field(default_factory=dict)
    concerns: dict[str, ConcernInfo] = Field(default_factory=dict)
    weather_variables: dict[str, WeatherVariable] = Field(min_length=1)
    banned_phrases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Vocabulary":
        if len(set(self.severity_ranks.values())) != len(self.severity_ranks):
            raise ValueError("severity_ranks must give every severity a distinct rank")
        if UNRECOGNIZED_ACTIVITY not in self.activities:
            raise ValueError(f"activities must include '{UNRECOGNIZED_ACTIVITY}'")
        reserved = PSEUDO_ACTIVITIES & self.activities.keys()
        if reserved:
            raise ValueError(f"activity names {sorted(reserved)} are reserved for applies_to")
        return self

    def tags_for_fact(self, fact: str) -> set[str]:
        """Allowed tag values for an ``intent.*`` fact."""
        return set(getattr(self, INTENT_FACTS[fact]))


def classify_fact(fact: str, vocabulary: Vocabulary) -> FactKind:
    """Return the kind of a fact path, or raise ``ValueError`` if it is not valid.

    Valid fact paths:
      * ``wx.<max|min|sum>.<weather_variable>`` -> number
      * ``wx.codes``                            -> list of WMO weather codes
      * ``intent.activities|groups|concerns``   -> list of vocabulary tags
    """
    if fact == WEATHER_CODES_FACT:
        if WEATHER_CODE_VARIABLE not in vocabulary.weather_variables:
            raise ValueError(f"'{fact}' requires '{WEATHER_CODE_VARIABLE}' in weather_variables in vocabulary.yaml")
        return "code_list"
    if fact in INTENT_FACTS:
        return "tag_list"
    parts = fact.split(".")
    if len(parts) == 3 and parts[0] == "wx":
        _, aggregate, variable = parts
        if aggregate not in WEATHER_AGGREGATES:
            raise ValueError(
                f"unknown aggregate '{aggregate}' in fact '{fact}'; "
                f"allowed: {list(WEATHER_AGGREGATES)}"
            )
        if variable not in vocabulary.weather_variables:
            raise ValueError(
                f"unknown weather variable '{variable}' in fact '{fact}'; "
                "add it to weather_variables in vocabulary.yaml"
            )
        enabled = vocabulary.weather_variables[variable].aggregates
        if aggregate not in enabled:
            raise ValueError(
                f"aggregate '{aggregate}' is not enabled for '{variable}' in fact '{fact}'; enabled: {enabled}"
            )
        return "number"
    raise ValueError(
        f"unknown fact '{fact}'; allowed: wx.<max|min|sum>.<variable>, "
        f"{WEATHER_CODES_FACT}, {', '.join(INTENT_FACTS)}"
    )


def _vocabulary_from(info: ValidationInfo) -> Vocabulary:
    vocabulary = (info.context or {}).get("vocabulary")
    if not isinstance(vocabulary, Vocabulary):
        raise ValueError("validation context must provide a 'vocabulary'")
    return vocabulary


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class Leaf(StrictModel):
    """``{fact, op, value}`` - compare one fact against a constant."""

    fact: str
    op: Operator
    value: Any

    @model_validator(mode="after")
    def _check_against_vocabulary(self, info: ValidationInfo) -> "Leaf":
        vocabulary = _vocabulary_from(info)
        kind = classify_fact(self.fact, vocabulary)

        if kind == "number":
            if self.op not in NUMBER_OPERATORS:
                raise ValueError(
                    f"operator '{self.op}' cannot be used with numeric fact '{self.fact}'; "
                    f"use one of {sorted(NUMBER_OPERATORS)}"
                )
            if self.op == "in":
                if not (isinstance(self.value, list) and self.value and all(map(is_number, self.value))):
                    raise ValueError(f"operator 'in' requires a non-empty list of numbers, got {self.value!r}")
            elif not is_number(self.value):
                raise ValueError(f"operator '{self.op}' requires a number, got {self.value!r}")
            return self

        if self.op not in LIST_OPERATORS:
            raise ValueError(
                f"operator '{self.op}' cannot be used with list fact '{self.fact}'; "
                f"use one of {sorted(LIST_OPERATORS)}"
            )
        if not (isinstance(self.value, list) and self.value):
            raise ValueError(f"operator '{self.op}' requires a non-empty list, got {self.value!r}")
        if kind == "code_list":
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in self.value):
                raise ValueError(f"'{self.fact}' values must be integer weather codes, got {self.value!r}")
        else:
            allowed = vocabulary.tags_for_fact(self.fact)
            unknown = [v for v in self.value if not isinstance(v, str) or v not in allowed]
            if unknown:
                raise ValueError(f"unknown tag(s) {unknown} for '{self.fact}'; allowed: {sorted(allowed)}")
        return self


class AllOf(StrictModel):
    """``all: [...]`` - true when every child is true."""

    all: list["Condition"] = Field(min_length=1)


class AnyOf(StrictModel):
    """``any: [...]`` - true when at least one child is true."""

    any: list["Condition"] = Field(min_length=1)


class NotOf(StrictModel):
    """``not: {...}`` - negates one child."""

    negated: "Condition" = Field(alias="not")


class AtLeastSpec(StrictModel):
    n: int = Field(ge=1)
    of: list["Condition"] = Field(min_length=1)

    @model_validator(mode="after")
    def _n_within_range(self) -> "AtLeastSpec":
        if self.n > len(self.of):
            raise ValueError(f"n={self.n} is larger than the number of conditions ({len(self.of)})")
        return self


class AtLeast(StrictModel):
    """``at_least: {n, of: [...]}`` - true when at least ``n`` children are true."""

    at_least: AtLeastSpec


CONDITION_SHAPE_HELP = (
    "each condition must be exactly one of: {fact, op, value} | {all: [...]} | "
    "{any: [...]} | {not: {...}} | {at_least: {n, of: [...]}}"
)
_COMBINATOR_KEYS = ("all", "any", "not", "at_least")


def _condition_shape(raw: Any) -> str | None:
    """Pick the condition model from the keys present (Pydantic discriminator)."""
    if isinstance(raw, dict):
        if len(raw) == 1 and next(iter(raw)) in _COMBINATOR_KEYS:
            return next(iter(raw))
        return "leaf" if "fact" in raw else None
    return {Leaf: "leaf", AllOf: "all", AnyOf: "any", NotOf: "not", AtLeast: "at_least"}.get(type(raw))


Condition = Annotated[
    Union[
        Annotated[Leaf, Tag("leaf")],
        Annotated[AllOf, Tag("all")],
        Annotated[AnyOf, Tag("any")],
        Annotated[NotOf, Tag("not")],
        Annotated[AtLeast, Tag("at_least")],
    ],
    Discriminator(
        _condition_shape,
        custom_error_type="invalid_condition",
        custom_error_message=CONDITION_SHAPE_HELP,
    ),
]

for _model in (AllOf, AnyOf, NotOf, AtLeastSpec, AtLeast):
    _model.model_rebuild()

_CONDITION_ADAPTER: TypeAdapter[Condition] = TypeAdapter(Condition)


def parse_condition(raw: Any, vocabulary: Vocabulary) -> Condition:
    """Validate a raw condition (e.g. a dict from YAML) against the vocabulary."""
    return _CONDITION_ADAPTER.validate_python(raw, context={"vocabulary": vocabulary})


def child_conditions(condition: Condition) -> list[Condition]:
    """Direct children of a condition (empty for a leaf)."""
    if isinstance(condition, AllOf):
        return list(condition.all)
    if isinstance(condition, AnyOf):
        return list(condition.any)
    if isinstance(condition, NotOf):
        return [condition.negated]
    if isinstance(condition, AtLeast):
        return list(condition.at_least.of)
    return []


def iter_leaves(condition: Condition) -> Iterator[Leaf]:
    """Yield every leaf in a condition tree, depth-first."""
    if isinstance(condition, Leaf):
        yield condition
        return
    for child in child_conditions(condition):
        yield from iter_leaves(child)


# ---------------------------------------------------------------------------
# SOP
# ---------------------------------------------------------------------------


class AppliesTo(StrictModel):
    """Who/what an SOP is about. Activities and groups are each "any-of"."""

    activities: list[str] = Field(min_length=1)
    groups: list[str] = Field(default_factory=list)


class SOP(StrictModel):
    """One Standard Operating Procedure, loaded from ``policy/sops/<id>.yaml``."""

    id: str = Field(pattern=r"^SOP-[A-Z0-9]+(-[A-Z0-9]+)*$")
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    severity: str
    scope: Literal["global", "activity"]
    applies_to: AppliesTo
    fallback: bool = False
    when: Condition
    guidance: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    cite: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @model_validator(mode="after")
    def _check_against_vocabulary(self, info: ValidationInfo) -> "SOP":
        vocabulary = _vocabulary_from(info)

        if self.severity not in vocabulary.severity_ranks:
            raise ValueError(
                f"unknown severity '{self.severity}'; allowed: {list(vocabulary.severity_ranks)}"
            )

        activities = self.applies_to.activities
        if self.scope == "global":
            if len(activities) != 1 or activities[0] not in PSEUDO_ACTIVITIES:
                raise ValueError(
                    "scope 'global' requires applies_to.activities to be exactly "
                    f"[{ANY_OUTDOOR}] or [{ANY_KNOWN_ACTIVITY}]"
                )
        else:
            misplaced = PSEUDO_ACTIVITIES & set(activities)
            if misplaced:
                raise ValueError(f"{sorted(misplaced)} can only be used with scope 'global'")
            unknown = [a for a in activities if a not in vocabulary.activities]
            if unknown:
                raise ValueError(f"unknown activity tag(s) {unknown} in applies_to.activities")

        unknown_groups = [g for g in self.applies_to.groups if g not in vocabulary.groups]
        if unknown_groups:
            raise ValueError(f"unknown group tag(s) {unknown_groups} in applies_to.groups")

        for fact in self.cite:
            try:
                kind = classify_fact(fact, vocabulary)
            except ValueError as exc:
                raise ValueError(f"cite: {exc}") from None
            if kind != "number":
                raise ValueError(f"cite entry '{fact}' must be a numeric fact wx.<max|min|sum>.<variable>")
        return self
