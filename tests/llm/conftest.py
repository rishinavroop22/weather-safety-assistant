"""Real-LLM tests: skipped unless LLM_API_KEY / LLM_BASE_URL / LLM_MODEL are configured.

Weather is faked so the expected SOP is known; the intent parser and composer are real.
Every test records what the model actually produced to evals/results/llm_test_results.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.llm import ChatJSONClient, LLMAnswerComposer, LLMConfig, LLMConfigError, LLMIntentParser
from app.policy import PolicyStore
from tests.conftest import POLICY_DIR

RESULTS_FILE = Path(__file__).resolve().parents[2] / "evals" / "results" / "llm_test_results.jsonl"


def _load_config() -> tuple[LLMConfig | None, str | None]:
    try:
        return LLMConfig.from_env(), None
    except LLMConfigError as exc:
        return None, str(exc)


CONFIG, CONFIG_ERROR = _load_config()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if CONFIG is not None:
        return
    skip = pytest.mark.skip(reason=f"real-LLM tests need credentials: {CONFIG_ERROR}")
    for item in items:
        if Path(str(item.fspath)).parent == Path(__file__).parent:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def llm_client() -> ChatJSONClient:
    assert CONFIG is not None
    return ChatJSONClient(CONFIG)


@pytest.fixture(scope="session")
def llm_roles(llm_client: ChatJSONClient) -> tuple[LLMIntentParser, LLMAnswerComposer]:
    store = PolicyStore(POLICY_DIR)
    return LLMIntentParser(llm_client, lambda: store.get().vocabulary), LLMAnswerComposer(llm_client)


@pytest.fixture(scope="session")
def _results_file() -> Path:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("", encoding="utf-8")
    return RESULTS_FILE


@pytest.fixture
def record(request: pytest.FixtureRequest, _results_file: Path):
    """Append one JSON line describing what the real model did in this test."""

    def write(**data: Any) -> None:
        entry = {
            "test": request.node.nodeid,
            "model": CONFIG.model if CONFIG else None,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **data,
        }
        with _results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=_jsonable, ensure_ascii=False) + "\n")

    return write


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """The traceable parts of a graph run, for the results file."""
    match = state.get("match_result")
    return {
        "parsed_intent": state.get("parsed_intent"),
        "request": state.get("request"),
        "location": state["location"].display_name if state.get("location") else None,
        "window": state["weather_facts"].window.label if state.get("weather_facts") else None,
        "primary_sop": match.primary.id if match and match.primary else None,
        "secondary_sops": [m.id for m in match.secondary] if match else [],
        "draft": state.get("draft"),
        "composer_error": state.get("composer_error"),
        "verification": state.get("verification"),
        "failure": state.get("failure"),
        "outcome": state.get("outcome"),
        "trace": state.get("trace"),
        "final_answer": state.get("final_answer"),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)
