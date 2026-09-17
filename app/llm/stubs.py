"""Deterministic stand-ins for the LLM roles (tests and Phase 3 demos).

They are not language understanding: ``ScriptedIntentParser`` only knows the
exact messages it was given. Phase 4 replaces both with real LLM implementations.
"""

from collections.abc import Mapping
from typing import Any

from app.llm.interfaces import ComposedAnswer, CompositionRequest, ConversationContext, ParsedIntent

ScriptEntry = ParsedIntent | dict[str, Any] | Exception


class ScriptedIntentParser:
    """Returns a pre-written intent for each known message (matched case-insensitively).

    Unknown messages are treated as off topic. A script entry that is an
    exception is raised, to simulate a failing model.
    """

    def __init__(self, script: Mapping[str, ScriptEntry]) -> None:
        self._script = {self._key(message): entry for message, entry in script.items()}
        self.calls: list[tuple[str, ConversationContext | None]] = []

    def parse(self, message: str, context: ConversationContext | None) -> ParsedIntent | dict[str, Any]:
        self.calls.append((message, context))
        entry = self._script.get(self._key(message), {"intent": "off_topic"})
        if isinstance(entry, Exception):
            raise entry
        return entry

    @staticmethod
    def _key(message: str) -> str:
        return " ".join(message.split()).casefold()


class TemplateAnswerComposer:
    """Restates the selected SOPs' authored guidance. Writes no numbers of its own."""

    def __init__(self) -> None:
        self.requests: list[CompositionRequest] = []

    def compose(self, request: CompositionRequest) -> ComposedAnswer:
        self.requests.append(request)
        primary = request.primary
        lines = [f"For {{location}}, {{window}}: {primary.title} ({primary.id})."]
        lines += [f"- {point}" for point in primary.guidance]
        for sop in request.secondary:
            lines.append(f"Also relevant: {sop.title} ({sop.id}).")
            lines += [f"- {point}" for point in sop.guidance]
        return ComposedAnswer(text="\n".join(lines), cited_sop_ids=request.sop_ids)
