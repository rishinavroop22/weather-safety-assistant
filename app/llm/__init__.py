"""LLM roles behind narrow interfaces (real providers arrive in Phase 4)."""

from app.llm.interfaces import (
    AnswerComposer,
    ComposedAnswer,
    CompositionRequest,
    ConversationContext,
    IntentParser,
    IntentType,
    ParsedIntent,
)
from app.llm.stubs import ScriptedIntentParser, TemplateAnswerComposer

__all__ = [
    "AnswerComposer",
    "ComposedAnswer",
    "CompositionRequest",
    "ConversationContext",
    "IntentParser",
    "IntentType",
    "ParsedIntent",
    "ScriptedIntentParser",
    "TemplateAnswerComposer",
]
