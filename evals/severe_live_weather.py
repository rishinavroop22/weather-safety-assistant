"""Severe live-weather eval: is a severe answer grounded in the real numbers of that request?

    python -m evals.severe_live_weather                # LLM roles: real if configured, else stub
    python -m evals.severe_live_weather --llm stub     # deterministic stub LLM roles
    python -m evals.severe_live_weather --llm real     # require the real LLM

Two parts, reported separately and never merged:

1. LIVE: scan a fixed list of cities with the real Open-Meteo API, using the production
   weather client, fact builder and policy engine, and pick a city whose live forecast
   triggers a high or critical SOP. Then run the full graph for that question and check the
   answer against that request's own API data. If no city qualifies right now the live case
   is SKIPPED, never PASS: severe weather can't be manufactured.
2. FIXTURE: replay the recorded real Open-Meteo response in
   evals/fixtures/open_meteo_bhopal_20260917.json (it contains thunderstorm codes) through
   the same graph and assertions. This validates the grounding path on any day, but it is
   labelled FIXTURE, not live.

No weather value, threshold or SOP id is hardcoded here. Results are saved to
evals/results/severe_live_weather_<UTC timestamp>.json.
"""

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph import build_graph, run_turn
from app.graph.rendering import WEATHER_SECTION_HEADER, fact_label, format_fact_value
from app.graph.verification import SOP_ID_PATTERN, check_numbers
from app.llm import (
    ChatJSONClient,
    LLMAnswerComposer,
    LLMConfig,
    LLMConfigError,
    LLMIntentParser,
    ScriptedIntentParser,
    TemplateAnswerComposer,
)
from app.policy import PolicySet, PolicyStore, QueryProfile, match_sops
from app.weather import FixtureWeatherProvider, OpenMeteoClient, WeatherLayerError, WeatherProvider, WindowPassedError, build_facts

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "policy"
RESULTS_DIR = ROOT / "evals" / "results"
SEVERE_FIXTURE = ROOT / "evals" / "fixtures" / "open_meteo_bhopal_20260917.json"
FIXTURE_CITY = "Bhopal"
MIN_SEVERITY = "high"

# Climates where storms, heavy rain, fog, gusts or heat are common. Only names: no weather data.
CITIES: tuple[str, ...] = (
    "Mumbai", "Kolkata", "Chennai", "Guwahati", "Bhubaneswar", "Delhi", "Dhaka", "Yangon",
    "Bangkok", "Manila", "Ho Chi Minh City", "Kuala Lumpur", "Singapore", "Jakarta", "Hong Kong",
    "Miami", "Houston", "Manaus", "Lagos", "Kinshasa", "Riyadh", "Phoenix",
)
# (activity tag, day) -> question. Cycling and hiking between them reach every high/critical SOP
# that doesn't need a vulnerable group (storm, rain system, heat, wet roads, fog, gusts).
QUESTIONS: dict[tuple[str, str], str] = {
    ("cycling", "today"): "Is it safe to cycle in {city} today?",
    ("cycling", "tomorrow"): "Is it safe to cycle in {city} tomorrow?",
    ("hiking", "today"): "Is it safe to go hiking near {city} today?",
    ("hiking", "tomorrow"): "Is it safe to go hiking near {city} tomorrow?",
}


@dataclass
class Candidate:
    city: str
    activity: str
    day: str
    sop_id: str
    severity: str
    rank: int

    @property
    def message(self) -> str:
        return QUESTIONS[(self.activity, self.day)].format(city=self.city)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_for_severe_weather(provider: WeatherProvider, policy: PolicySet, cities: Sequence[str]) -> tuple[list[dict[str, Any]], Candidate | None]:
    """Evaluate each city's live forecast with the production policy engine; return the most severe case."""
    from app.policy import required_weather_variables

    variables = required_weather_variables(policy)
    minimum = policy.severity_rank(MIN_SEVERITY)
    rows: list[dict[str, Any]] = []
    best: Candidate | None = None
    for city in cities:
        try:
            location = provider.geocode(city)
            forecast = provider.forecast(location, variables)
        except WeatherLayerError as exc:
            rows.append({"city": city, "error": f"{type(exc).__name__}: {exc.kind}"})
            continue
        for activity, day in QUESTIONS:
            try:
                facts = build_facts(forecast, day=day, part_of_day="whole_day", vocabulary=policy.vocabulary)
            except (WindowPassedError, WeatherLayerError) as exc:
                rows.append({"city": city, "activity": activity, "day": day, "error": type(exc).__name__})
                continue
            primary = match_sops(policy, QueryProfile(activities=[activity]), facts.policy_facts()).primary
            rank = policy.severity_rank(primary.severity) if primary else -1
            rows.append({
                "city": city, "location": location.display_name, "activity": activity, "day": day,
                "primary_sop": primary.id if primary else None, "severity": primary.severity if primary else None,
            })
            if primary and rank >= minimum and (best is None or rank > best.rank):
                best = Candidate(city, activity, day, primary.id, primary.severity, rank)
    return rows, best


# ---------------------------------------------------------------------------
# Assertions on the graph's own final state
# ---------------------------------------------------------------------------


def grounding_assertions(state: dict[str, Any], policy: PolicySet) -> list[dict[str, Any]]:
    """Deterministic checks that the answer is a severe, SOP-cited answer built from this request's API data."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    outcome = state.get("outcome")
    match, facts, composition, raw = state.get("match_result"), state.get("weather_facts"), state.get("composition"), state.get("raw_weather")
    answer = state.get("final_answer") or ""
    check("policy_answer_produced", outcome in ("answered", "answered_fallback") and composition is not None,
          f"outcome={outcome}, failure={getattr(state.get('failure'), 'kind', None)}")
    if not (match and match.primary and facts and composition and raw):
        return checks

    primary = match.primary
    vocabulary = policy.vocabulary
    body = answer.split(f"\n\n{WEATHER_SECTION_HEADER}", 1)[0]
    check("severity_high_or_critical", policy.severity_rank(primary.severity) >= policy.severity_rank(MIN_SEVERITY),
          f"{primary.id} severity={primary.severity}")

    rerun = match_sops(policy, QueryProfile(activities=state["request"].activities, groups=state["request"].groups,
                                            concerns=state["request"].concerns), facts.policy_facts()).primary
    check("sop_reproducible_from_request_facts", rerun is not None and rerun.id == primary.id,
          f"graph={primary.id}, policy engine re-run={rerun.id if rerun else None}")

    check("sop_id_in_answer_text", primary.id in body, f"looking for {primary.id} in the answer text")
    check("sop_cited_in_policy_line", f"Policy applied: {primary.id}" in answer, f"looking for 'Policy applied: {primary.id}'")

    index = {t: i for i, t in enumerate(raw.hourly_times)}
    mismatches = []
    for fact, value in composition.facts.items():
        _, aggregate, variable = fact.split(".")
        hourly = [raw.hourly_series(variable)[index[h]] for h in facts.window.hours]
        expected = {"max": max, "min": min, "sum": lambda v: round(sum(v), 2)}[aggregate](hourly)
        if expected != value:
            mismatches.append(f"{fact}: answer fact {value} != API payload {expected}")
    check("weather_values_equal_api_payload", composition.facts and not mismatches,
          "; ".join(mismatches) or f"{len(composition.facts)} cited values recomputed from this request's raw Open-Meteo hourly data")

    missing = [f"{fact_label(f, vocabulary)}: {format_fact_value(f, v, vocabulary)}" for f, v in composition.facts.items()
               if f"- {fact_label(f, vocabulary)}: {format_fact_value(f, v, vocabulary)}" not in answer]
    check("weather_values_shown_in_answer", not missing, "missing: " + "; ".join(missing) if missing else "every cited value is shown with its unit")

    violations = check_numbers(SOP_ID_PATTERN.sub(" ", body), composition, vocabulary)
    check("no_unsupported_numbers_or_thresholds", not violations, "; ".join(violations) or "every number in the answer text is a weather value, policy threshold, window time or authored guidance number")
    return checks


# ---------------------------------------------------------------------------
# Running a case
# ---------------------------------------------------------------------------


def _roles(llm_mode: str, message: str, intent: dict[str, Any], store: PolicyStore) -> tuple[Any, Any, str]:
    if llm_mode == "real":
        client = ChatJSONClient(LLMConfig.from_env())
        return LLMIntentParser(client, lambda: store.get().vocabulary), LLMAnswerComposer(client), f"real LLM ({client.config.model})"
    return ScriptedIntentParser({message: intent}), TemplateAnswerComposer(), "stub LLM roles (scripted intent + template composer)"


def run_case(provider: WeatherProvider, store: PolicyStore, message: str, intent: dict[str, Any], llm_mode: str) -> dict[str, Any]:
    parser, composer, label = _roles(llm_mode, message, intent, store)
    graph = build_graph(policy_store=store, weather_provider=provider, intent_parser=parser, answer_composer=composer)
    state = run_turn(graph, f"eval-{uuid.uuid4().hex}", message)
    assertions = grounding_assertions(state, store.get())
    facts, match = state.get("weather_facts"), state.get("match_result")
    return {
        "llm_roles": label,
        "request": message,
        "location": state["location"].display_name if state.get("location") else None,
        "window": facts.window.label if facts else None,
        "timezone": facts.timezone if facts else None,
        "weather_source": facts.source_url if facts else None,
        "retrieved_at_utc": facts.fetched_at_utc.isoformat(timespec="seconds") if facts and facts.fetched_at_utc else None,
        "weather_facts": facts.values if facts else None,
        "matched_sops": [{"id": m.id, "severity": m.severity, "evidence": m.evidence} for m in match.matched] if match else [],
        "outcome": state.get("outcome"),
        "failure": state["failure"].kind if state.get("failure") else None,
        "verification": state["verification"].model_dump() if state.get("verification") else None,
        "answer": state.get("final_answer"),
        "assertions": assertions,
        "passed": bool(assertions) and all(a["passed"] for a in assertions),
    }


def _llm_unavailable(case: dict[str, Any]) -> bool:
    return case["failure"] in ("llm_unavailable", "invalid_intent")


def run_live(store: PolicyStore, provider: WeatherProvider, cities: Sequence[str], llm_mode: str) -> dict[str, Any]:
    rows, best = scan_for_severe_weather(provider, store.get(), cities)
    result: dict[str, Any] = {"cities_scanned": list(cities), "scan": rows, "attempts": []}
    if best is None:
        result.update(status="SKIPPED", reason=f"no scanned city currently has a {MIN_SEVERITY}-or-higher SOP match for cycling or hiking today/tomorrow")
        return result

    result["selected"] = {"city": best.city, "activity": best.activity, "day": best.day, "scan_sop": best.sop_id, "scan_severity": best.severity}
    intent = {"intent": "activity_safety", "activities": [best.activity], "location": best.city, "day": best.day, "part_of_day": "whole_day"}
    modes = ["real", "stub"] if llm_mode == "auto" else [llm_mode]
    for mode in modes:
        case = run_case(provider, store, best.message, intent, mode)
        result["attempts"].append(case)
        if mode == "real" and _llm_unavailable(case) and "stub" in modes:
            continue   # provider unavailable: record the attempt, then check grounding with stub roles
        break
    final = result["attempts"][-1]
    if llm_mode != "stub" and _llm_unavailable(result["attempts"][0]) and len(result["attempts"]) == 1:
        result.update(status="NOT RUN", reason=f"real LLM unavailable ({result['attempts'][0]['failure']})")
    else:
        result.update(status="PASS" if final["passed"] else "FAIL",
                      reason="all grounding assertions passed" if final["passed"] else "failed: " + ", ".join(a["name"] for a in final["assertions"] if not a["passed"]))
    return result


def run_fixture_replay(store: PolicyStore, fixture: Path = SEVERE_FIXTURE, llm_mode: str = "stub") -> dict[str, Any]:
    provider = FixtureWeatherProvider.from_file(fixture)
    message = f"Is it safe to cycle in {FIXTURE_CITY} today?"
    intent = {"intent": "activity_safety", "activities": ["cycling"], "location": FIXTURE_CITY, "day": "today", "part_of_day": "whole_day"}
    case = run_case(provider, store, message, intent, llm_mode)
    return {
        "fixture": str(fixture.relative_to(ROOT)) if fixture.is_relative_to(ROOT) else str(fixture),
        "note": "recorded real Open-Meteo response replayed offline; this is NOT a live-weather result",
        "case": case,
        "status": "PASS" if case["passed"] else "FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", choices=["auto", "real", "stub"], default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    llm_mode = args.llm
    if llm_mode in ("auto", "real"):
        try:
            LLMConfig.from_env()
        except LLMConfigError as exc:
            if llm_mode == "real":
                print(exc)
                return 1
            llm_mode = "stub"

    store = PolicyStore(POLICY_DIR)
    started = datetime.now(timezone.utc)
    with OpenMeteoClient() as client:
        live = run_live(store, client, CITIES, llm_mode)
    fixture = run_fixture_replay(store)
    report = {"run_at_utc": started.isoformat(timespec="seconds"), "llm_mode_requested": args.llm, "live": live, "fixture_replay": fixture}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"severe_live_weather_{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"LIVE: {live['status']} - {live['reason']}")
    if "selected" in live:
        print(f"  selected: {live['selected']}")
        for attempt in live["attempts"]:
            print(f"  attempt [{attempt['llm_roles']}] outcome={attempt['outcome']} failure={attempt['failure']} passed={attempt['passed']}")
            for a in attempt["assertions"]:
                print(f"    {'ok ' if a['passed'] else 'BAD'} {a['name']}: {a['detail']}")
    print(f"FIXTURE REPLAY: {fixture['status']} ({fixture['fixture']}, {fixture['case']['llm_roles']})")
    for a in fixture["case"]["assertions"]:
        print(f"    {'ok ' if a['passed'] else 'BAD'} {a['name']}: {a['detail']}")
    print(f"Saved {path.relative_to(ROOT)}")
    return 0 if live["status"] != "FAIL" and fixture["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
