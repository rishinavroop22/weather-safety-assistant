"""Build the production graph: real policy files, live Open-Meteo, real LLM roles."""

from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from app.graph import build_graph
from app.llm import ChatJSONClient, LLMAnswerComposer, LLMConfig, LLMIntentParser
from app.policy import PolicyStore
from app.weather import OpenMeteoClient

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[1] / "policy"


def build_production_graph(
    policy_dir: Path | str = DEFAULT_POLICY_DIR,
    llm_config: LLMConfig | None = None,
    policy_store: PolicyStore | None = None,
) -> CompiledStateGraph:
    """Wire the graph for real use.

    Args:
        policy_store: pass one to share it with the caller (e.g. the API, for display
            formatting); otherwise one is created for ``policy_dir``.

    Raises:
        LLMConfigError: LLM environment variables are missing or invalid.
        PolicyError: the policy files are invalid (checked at startup, not on the first request).
    """
    config = llm_config or LLMConfig.from_env()
    policy_store = policy_store or PolicyStore(policy_dir)
    policy_store.get()
    client = ChatJSONClient(config)
    return build_graph(
        policy_store=policy_store,
        weather_provider=OpenMeteoClient(),
        intent_parser=LLMIntentParser(client, lambda: policy_store.get().vocabulary),
        answer_composer=LLMAnswerComposer(client),
    )
