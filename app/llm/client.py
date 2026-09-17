"""Minimal OpenAI-compatible chat-completions client that returns a JSON object.

One POST to ``{LLM_BASE_URL}/chat/completions`` per call, no retries (a retry
would silently double cost and latency). Every failure becomes an ``LLMError``
with a machine-readable ``kind``; nothing is ever fabricated.
"""

import json
import logging
import re
import time
from typing import Any

import httpx

from app.llm.config import LLMConfig

logger = logging.getLogger(__name__)

CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class LLMError(Exception):
    """An LLM call failed.

    ``kind``: timeout | network_error | rate_limited | model_unavailable | http_error
              | invalid_response | empty_response
    ``detail`` is safe to log (the API key is scrubbed).
    """

    def __init__(self, kind: str, detail: str, status_code: int | None = None) -> None:
        super().__init__(f"LLM call failed ({kind})")
        self.kind = kind
        self.detail = detail
        self.status_code = status_code


class ChatJSONClient:
    """Sends a system + user message and returns the model's JSON object."""

    def __init__(self, config: LLMConfig, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self._http = http_client or httpx.Client(timeout=config.timeout_seconds)

    def complete_json(self, *, system: str, user: str, schema_name: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """Return the parsed JSON object produced by the model.

        With ``response_format=json_schema`` the provider enforces ``schema``;
        with ``json_object`` the schema is appended to the system prompt instead.
        The caller still validates the result with Pydantic.

        Raises:
            LLMError: on any transport, HTTP, or output problem.
        """
        if self.config.response_format == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}
            system = f"{system}\n\nReturn only a JSON object matching this JSON schema:\n{json.dumps(schema)}"

        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        started = time.perf_counter()
        try:
            response = self._http.post(
                f"{self.config.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise self._error("timeout", f"{type(exc).__name__} after {self.config.timeout_seconds}s") from None
        except httpx.TransportError as exc:
            raise self._error("network_error", f"{type(exc).__name__}: {exc}") from None
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not response.is_success:
            raise self._http_error(response)

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise self._error("invalid_response", "response is not a chat completion") from None

        usage = data.get("usage") or {}
        logger.info(
            "LLM %s: model=%s %.0fms prompt_tokens=%s completion_tokens=%s",
            schema_name, self.config.model, elapsed_ms, usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )

        if message.get("refusal"):
            raise self._error("invalid_response", "model refused")
        if choice.get("finish_reason") == "length":
            raise self._error("invalid_response", "output truncated (finish_reason=length)")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise self._error("empty_response", "model returned no content")

        text = content.strip()
        fenced = CODE_FENCE.match(text)
        if fenced:
            text = fenced.group(1)
        try:
            parsed = json.loads(text)
        except ValueError:
            raise self._error("invalid_response", f"content is not valid JSON ({len(text)} chars)") from None
        if not isinstance(parsed, dict):
            raise self._error("invalid_response", "content is JSON but not an object")
        return parsed

    def _http_error(self, response: httpx.Response) -> LLMError:
        reason = _provider_message(response)
        status = response.status_code
        if status == 429:
            kind = "rate_limited"
        elif status == 404 or ("model" in reason.lower() and ("not found" in reason.lower() or "does not exist" in reason.lower())):
            kind = "model_unavailable"
        else:
            kind = "http_error"
        return self._error(kind, f"HTTP {status}: {reason}", status)

    def _error(self, kind: str, detail: str, status_code: int | None = None) -> LLMError:
        safe = detail.replace(self.config.api_key, "***") if self.config.api_key else detail
        logger.warning("LLM call failed: kind=%s model=%s detail=%s", kind, self.config.model, safe)
        return LLMError(kind, safe, status_code)


def _provider_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, list) and body and isinstance(body[0], dict):   # some providers wrap errors in a list
            body = body[0]
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
    except ValueError:
        pass
    return (response.text or response.reason_phrase)[:300]
