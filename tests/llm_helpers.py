"""A mocked OpenAI-compatible endpoint for testing the real LLM roles without a key."""

import json
from collections.abc import Callable
from typing import Any

import httpx

from app.llm import ChatJSONClient, LLMAnswerComposer, LLMConfig, LLMIntentParser
from app.policy import PolicyStore
from tests.conftest import POLICY_DIR

TEST_KEY = "sk-test-SECRET-123"
Reply = dict[str, Any] | str | httpx.Response | Exception | Callable[[dict[str, Any]], Any]


def llm_config(**overrides: Any) -> LLMConfig:
    fields = {"api_key": TEST_KEY, "base_url": "https://llm.test/v1", "model": "test-model", "timeout_seconds": 5.0}
    fields.update(overrides)
    return LLMConfig(**fields)


def completion(content: Any, finish_reason: str = "stop", **message: Any) -> httpx.Response:
    """A chat-completions response whose message content is ``content`` (dicts are JSON-encoded)."""
    text = json.dumps(content) if isinstance(content, dict) else content
    body = {
        "id": "cmpl-test",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": text, **message}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(200, json=body)


class MockLLM:
    """Routes each request by its schema name ('parsed_intent' / 'composed_answer') to queued replies."""

    def __init__(self, **replies: Reply | list[Reply]) -> None:
        self.replies = {name: (value if isinstance(value, list) else [value]) for name, value in replies.items()}
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        name = body["response_format"].get("json_schema", {}).get("name", "json_object")
        queue = self.replies.get(name) or self.replies.get("any")
        if not queue:
            raise AssertionError(f"unexpected LLM call for {name}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(reply) and not isinstance(reply, httpx.Response):
            reply = reply(body)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return completion(reply)

    def calls(self, name: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.requests if name is None or r["body"]["response_format"].get("json_schema", {}).get("name") == name]

    def client(self, **config: Any) -> ChatJSONClient:
        return ChatJSONClient(llm_config(**config), httpx.Client(transport=httpx.MockTransport(self.handler)))

    def roles(self, policy_dir=POLICY_DIR) -> tuple[LLMIntentParser, LLMAnswerComposer]:
        client = self.client()
        store = PolicyStore(policy_dir)
        return LLMIntentParser(client, lambda: store.get().vocabulary), LLMAnswerComposer(client)
