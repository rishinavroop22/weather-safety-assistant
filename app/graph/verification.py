"""Deterministic checks on the composer's draft before anyone sees it.

The composer may only use what it was given in the ``CompositionRequest``.
A draft fails if:

1. it is missing or empty,
2. it does not cite the selected SOP, or cites / mentions any SOP that was not selected,
3. it uses a placeholder that is not a reportable fact, ``{location}`` or ``{window}``,
4. it writes a number that fails the field-aware number rules below,
5. it contains a banned phrase from the vocabulary.

Number rules (placeholders and SOP ids are removed first). Each literal number is
classified by its context and checked only against sources of the same kind:

* Time/date (``17:00``, ``2026-09-17``): must appear in the window or guidance text.
* Weather value - followed by a weather unit (``°C``, ``%``, ``km/h``, ``mm``, ``m``
  and spelled-out aliases): must equal a reported fact (rounding allowed) or a policy
  threshold of a variable with that unit. A weather label just before the number
  (e.g. "gusts") narrows the variables further.
* No unit: accepted if the same number appears in the authored SOP/window text with
  the same following word ("30 minutes"), or with the same preceding word ("SPF 30")
  when no weather label is nearby. Otherwise, if a weather label is nearby
  ("wind speed is 30"), it must match that variable's fact or threshold.
  Anything else is rejected.

So "SPF 30" in guidance never makes "wind speed is 30 km/h" acceptable. This is a
guard, not a semantic classifier.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.graph.rendering import CONTEXT_PLACEHOLDERS, PLACEHOLDER_PATTERN
from app.graph.state import VerificationResult
from app.llm import ComposedAnswer, CompositionRequest
from app.policy import Vocabulary

SOP_ID_PATTERN = re.compile(r"\bSOP-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")
TEMPORAL_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_.:])(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d+(?:\.\d+)?)")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
CLAUSE_BOUNDARY = re.compile(r"[;!?\n]|\.(?=\s|$)")
NEXT_WORD_PATTERN = re.compile(r"\s*([A-Za-z][A-Za-z'\-]*)")

# Spelled-out forms of the vocabulary's weather units (generic language, not SOP-specific).
UNIT_ALIASES: tuple[tuple[str, str], ...] = (
    (r"°\s*C|°|degrees?(?:\s+(?:celsius|c))?|celsius", "°C"),
    (r"%|percent|per\s+cent", "%"),
    (r"km/h|kmph|kph|km\s+per\s+hour|kilomet(?:re|er)s?\s+(?:per|an)\s+hour|km", "km/h"),
    (r"mm|millimet(?:re|er)s?", "mm"),
    (r"met(?:re|er)s?|m", "m"),
)
UNIT_PATTERN = re.compile(
    r"\s*(?:" + "|".join(f"(?P<unit{i}>{pattern})" for i, (pattern, _) in enumerate(UNIT_ALIASES)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
LABEL_WINDOW_WORDS = 5                      # how far back a weather label counts as "nearby"
LABEL_STOPWORDS = {"of", "chance", "code", "relative", "weather"}


@dataclass(frozen=True)
class NumberMention:
    token: str
    unit: str | None            # canonical weather unit written right after the number
    prev_word: str | None
    next_word: str | None
    nearby_words: tuple[str, ...]


def verify_draft(
    draft: ComposedAnswer | None,
    composition: CompositionRequest | None,
    *,
    vocabulary: Vocabulary,
    composer_error: str | None = None,
) -> VerificationResult:
    """Return pass/fail with every violation found (not just the first)."""
    if composition is None:
        return VerificationResult(passed=False, violations=["no selected SOP and weather facts to ground an answer"])
    if draft is None:
        reason = f" ({composer_error})" if composer_error else ""
        return VerificationResult(passed=False, violations=[f"composer produced no valid answer{reason}"])
    text = draft.text.strip()
    if not text:
        return VerificationResult(passed=False, violations=["composer returned empty text"])

    violations: list[str] = []
    allowed_ids = set(composition.sop_ids)

    if composition.primary.id not in draft.cited_sop_ids:
        violations.append(f"selected SOP {composition.primary.id} is not cited")
    violations += [f"cites SOP {sop_id} that was not selected" for sop_id in draft.cited_sop_ids if sop_id not in allowed_ids]
    violations += [f"mentions SOP {sop_id} that was not selected" for sop_id in SOP_ID_PATTERN.findall(text) if sop_id not in allowed_ids]

    for name in PLACEHOLDER_PATTERN.findall(text):
        if name not in composition.facts and name not in CONTEXT_PLACEHOLDERS:
            violations.append(f"unknown placeholder {{{name}}}")

    prose = SOP_ID_PATTERN.sub(" ", PLACEHOLDER_PATTERN.sub(" ", text))
    violations += check_numbers(prose, composition, vocabulary)

    lowered = text.casefold()
    violations += [f"banned phrase '{phrase}'" for phrase in vocabulary.banned_phrases if phrase.casefold() in lowered]

    violations = list(dict.fromkeys(violations))
    return VerificationResult(passed=not violations, violations=violations)


def check_numbers(prose: str, composition: CompositionRequest, vocabulary: Vocabulary) -> list[str]:
    """Apply the field-aware number rules; return one violation per rejected number."""
    authored_texts = [composition.location_name, composition.window_description]
    for sop in [composition.primary, *composition.secondary]:
        authored_texts += [sop.title, *sop.guidance]
    authored = [m for text in authored_texts for m in find_numbers(text)]
    authored_times = {m.token for m in authored if TEMPORAL_PATTERN.fullmatch(m.token)}
    authored_next = {(float(m.token), m.next_word) for m in authored if not TEMPORAL_PATTERN.fullmatch(m.token) and m.next_word}
    authored_prev = {(m.prev_word, float(m.token)) for m in authored if not TEMPORAL_PATTERN.fullmatch(m.token) and m.prev_word}
    keywords = label_keywords(vocabulary)

    violations: list[str] = []
    for mention in find_numbers(prose):
        token = mention.token
        if TEMPORAL_PATTERN.fullmatch(token):
            if token not in authored_times:
                violations.append(f"unsupported number {token} (time/date not in the window or guidance)")
            continue

        nearby_variables = {var for word in mention.nearby_words for var in keywords.get(_singular(word), ())}

        if mention.unit is not None:
            same_unit = {var for var, info in vocabulary.weather_variables.items() if info.unit == mention.unit}
            candidates = (nearby_variables & same_unit) or same_unit
            if not matches_weather(token, candidates, composition):
                violations.append(f"unsupported number {token} ({mention.unit}: not a reported weather value or policy threshold)")
            continue

        value = float(token)
        if (value, mention.next_word) in authored_next:
            continue
        if not nearby_variables and (mention.prev_word, value) in authored_prev:
            continue
        if nearby_variables:
            if not matches_weather(token, nearby_variables, composition):
                labels = ", ".join(sorted(nearby_variables))
                violations.append(f"unsupported number {token} (does not match the {labels} value or threshold)")
            continue
        violations.append(f"unsupported number {token} (not a weather value and not in the SOP guidance)")
    return violations


def matches_weather(token: str, variables: Iterable[str], composition: CompositionRequest) -> bool:
    """True if ``token`` equals a reported fact (rounded to the precision written) or a policy threshold
    for one of ``variables``."""
    variables = set(variables)
    decimals = len(token.split(".", 1)[1]) if "." in token else 0
    value = float(token)
    for fact, fact_value in composition.facts.items():
        if fact.split(".")[2] in variables and round(float(fact_value), decimals) == value:
            return True
    for fact, thresholds in composition.policy_thresholds.items():
        if fact.split(".")[2] in variables and value in thresholds:
            return True
    return False


def find_numbers(text: str) -> list[NumberMention]:
    """Every number in ``text`` with the context the rules need."""
    mentions: list[NumberMention] = []
    segment_start = 0
    for match in NUMBER_PATTERN.finditer(text):
        before = text[segment_start:match.start()]
        boundaries = list(CLAUSE_BOUNDARY.finditer(before))
        if boundaries:
            before = before[boundaries[-1].end():]
        words = [word.lower() for word in WORD_PATTERN.findall(before)]

        after = text[match.end():]
        unit_match = UNIT_PATTERN.match(after)
        next_match = NEXT_WORD_PATTERN.match(after)
        mentions.append(
            NumberMention(
                token=match.group(),
                unit=_canonical_unit(unit_match),
                prev_word=words[-1] if words else None,
                next_word=next_match.group(1).lower() if next_match else None,
                nearby_words=tuple(words[-LABEL_WINDOW_WORDS:]),
            )
        )
        segment_start = match.end()
    return mentions


def label_keywords(vocabulary: Vocabulary) -> dict[str, set[str]]:
    """Words from weather variable labels -> variables, e.g. ``gust`` -> {wind_gusts_10m}."""
    keywords: dict[str, set[str]] = {}
    for variable, info in vocabulary.weather_variables.items():
        for word in (info.label or variable.replace("_", " ")).lower().split():
            if word not in LABEL_STOPWORDS:
                keywords.setdefault(_singular(word), set()).add(variable)
    return keywords


def _canonical_unit(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    for i, (_, unit) in enumerate(UNIT_ALIASES):
        if match.group(f"unit{i}") is not None:
            return unit
    return None


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
