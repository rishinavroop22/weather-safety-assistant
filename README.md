# Weather-Advisory Support Bot

A LangGraph chat bot that answers outdoor-activity safety questions using live
Open-Meteo weather, and only gives advice that comes from written SOPs.

> **Status: Phase 2 - policy layer + Open-Meteo weather layer.** SOP files,
> validation, the deterministic matching engine, geocoding, forecast fetching
> and fact building are done and tested. LangGraph, LLM, API and frontend come
> in later phases.

## Setup

Requires Python 3.11+.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` when later phases need an LLM key. `.env` is git-ignored.

## Run

```bash
python -m app.policy               # validate policy/ and list the loaded SOPs
python -m pytest                   # run the unit tests (no network needed)
python -m evals.smoke_open_meteo   # optional live Open-Meteo check; saves a fixture
```

## Weather layer (`app/weather/`)

```
city ─► OpenMeteoClient.geocode() ─► Location (name, admin1, country, lat, lon, timezone)
     ─► OpenMeteoClient.forecast(location, required_weather_variables(policy))
     ─► RawForecast (untouched JSON + parsed local times)
     ─► build_facts(forecast, day, part_of_day, vocabulary)
     ─► WeatherFacts.values  = {"wx.max.uv_index": 7.3, "wx.codes": [1, 95], ...}  → match_sops()
```

- The request uses `timezone=auto`; "now" is Open-Meteo's `current.time` in local time, never the server clock.
- Windows: now = current hour +2h; morning 06–11; afternoon 12–16; evening 17–20; night 21–23;
  whole day = now–23 (today) or 06–21 (tomorrow). A window that is fully over raises `WindowPassedError`.
- Aggregates per variable come from `aggregates:` in `vocabulary.yaml`. A null hour inside the window
  makes that variable's facts `None`, which the policy engine treats as UNKNOWN. Nothing is filled with zero.
- Failures raise `LocationError` / `WeatherError` with `kind`
  (`not_found | timeout | http_error | network_error | invalid_response`), a user-safe `message` and a log-only `detail`.
- Anything with `geocode()` and `forecast()` is a `WeatherProvider`; `FixtureWeatherProvider` replays saved responses.

## Policy layout

```
policy/
  vocabulary.yaml      allowed activity/group/concern tags, weather variables, severities
  sops/<SOP-ID>.yaml   one SOP per file; the file name must equal the id
```

**Why YAML, one file per SOP:** people who aren't engineers can read and edit it,
diffs are clean, it's validated strictly by Pydantic on load, and adding a rule means adding a file.

### SOP fields

| Field | Meaning |
|---|---|
| `id`, `version`, `title`, `category` | identity; `category` is a free snake_case label |
| `severity` | one of `vocabulary.yaml` `severity_ranks` (info, low, moderate, high, critical) |
| `scope` | `activity` = listed activities only; `global` = applies across activities |
| `applies_to.activities` | any-of. For `global`: exactly `[any_outdoor]` or `[any_known_activity]` |
| `applies_to.groups` | any-of, optional extra requirement (children, elderly, pets) |
| `fallback` | `true` = only considered when nothing else matched and nothing was unknown |
| `when` | condition tree (below) |
| `guidance` | the authored advice; the only advice the bot may give for this SOP |
| `cite` | numeric facts whose values the answer must show |
| `rationale` | why the rule exists (for reviewers) |

### Condition language

```yaml
when:
  all:                                   # also: any, not, at_least
    - fact: wx.max.wind_gusts_10m        # wx.<max|min|sum>.<variable>
      op: gte                            # gt gte lt lte eq in
      value: 45
    - fact: intent.groups                # intent.activities|groups|concerns, wx.codes
      op: intersects                     # intersects, not_intersects
      value: [children]
    - at_least:
        n: 2
        of: [ ... ]
```

## How matching works

1. **Applicability.** `applies_to` is checked against the user's activities and groups.
2. **Evaluation with three-valued logic.** Each result is TRUE, FALSE or UNKNOWN. A missing or null weather value is
   **UNKNOWN**, never FALSE. `all`/`any`/`not`/`at_least` follow Kleene logic.
   Each match records **evidence**, e.g. `wx.max.uv_index = 9.1, meets >= 7`.
3. **Fallback.** Fallback SOPs run only if no regular SOP matched **and** no
   applicable regular SOP was UNKNOWN. Otherwise "no thresholds crossed" could be false.
4. **Ranking.** Sort by severity (highest first), then global scope before activity scope,
   then specificity (applies_to constraints + condition leaves), then id alphabetically.
   The first result is the primary SOP; the next two are secondary.

### Adding an SOP (no code change)

1. Copy an existing file in `policy/sops/` to `policy/sops/SOP-NEW-01.yaml` and set `id: SOP-NEW-01`.
2. Edit `applies_to`, `when`, `guidance`, `cite`.
3. Run `python -m app.policy`. It reports any invalid field by file and path.

A new *activity/group/concern tag* or a new *Open-Meteo variable* is also a
config-only change (add it to `vocabulary.yaml`). A new **kind of fact**, such as air
quality or official weather alerts, needs code, because the data has to be fetched.
