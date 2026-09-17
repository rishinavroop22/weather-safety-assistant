"""Real LLM implementations of the two roles, behind the Phase 3 interfaces.

Each role makes exactly one chat-completions call and returns raw JSON; the
graph validates it (``validate_intent`` / ``ComposedAnswer``) and, for the
composer, verifies it deterministically before anything reaches the user.
"""

from collections.abc import Callable
from typing import Any

from app.llm.client import ChatJSONClient
from app.llm.interfaces import CompositionRequest, ConversationContext
from app.llm.prompts import (
    COMPOSED_ANSWER_SCHEMA,
    COMPOSER_SYSTEM_PROMPT,
    composer_user_prompt,
    intent_json_schema,
    intent_system_prompt,
    intent_user_prompt,
)
from app.policy import Vocabulary

INTENT_MAX_TOKENS = 300
COMPOSER_MAX_TOKENS = 700


class LLMIntentParser:
    """Free text -> ParsedIntent-shaped dict, constrained to the current vocabulary.

    Args:
        client: the chat client.
        vocabulary: callable returning the current vocabulary (e.g.
            ``lambda: policy_store.get().vocabulary``), so vocabulary edits apply live.
    """

    def __init__(self, client: ChatJSONClient, vocabulary: Callable[[], Vocabulary]) -> None:
        self.client = client
        self.vocabulary = vocabulary

    def parse(self, message: str, context: ConversationContext | None) -> dict[str, Any]:
        vocabulary = self.vocabulary()
        return self.client.complete_json(
            system=intent_system_prompt(vocabulary),
            user=intent_user_prompt(message, context),
            schema_name="parsed_intent",
            schema=intent_json_schema(vocabulary),
            max_tokens=INTENT_MAX_TOKENS,
        )


class LLMAnswerComposer:
    """CompositionRequest -> ComposedAnswer-shaped dict. Wording only."""

    def __init__(self, client: ChatJSONClient) -> None:
        self.client = client

    def compose(self, request: CompositionRequest) -> dict[str, Any]:
        return self.client.complete_json(
            system=COMPOSER_SYSTEM_PROMPT,
            user=composer_user_prompt(request),
            schema_name="composed_answer",
            schema=COMPOSED_ANSWER_SCHEMA,
            max_tokens=COMPOSER_MAX_TOKENS,
        )
