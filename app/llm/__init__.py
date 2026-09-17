"""LLM roles behind narrow interfaces: intent extraction and answer wording."""

from app.llm.client import ChatJSONClient, LLMError
from app.llm.config import LLMConfig, LLMConfigError
from app.llm.interfaces import (
    AnswerComposer,
    ComposedAnswer,
    CompositionRequest,
    ConversationContext,
    IntentParser,
    IntentType,
    ParsedIntent,
)
from app.llm.roles import LLMAnswerComposer, LLMIntentParser
from app.llm.stubs import ScriptedIntentParser, TemplateAnswerComposer

__all__ = [
    "AnswerComposer",
    "ChatJSONClient",
    "ComposedAnswer",
    "CompositionRequest",
    "ConversationContext",
    "IntentParser",
    "IntentType",
    "LLMConfig",
    "LLMAnswerComposer",
    "LLMConfigError",
    "LLMError",
    "LLMIntentParser",
    "ParsedIntent",
    "ScriptedIntentParser",
    "TemplateAnswerComposer",
]
