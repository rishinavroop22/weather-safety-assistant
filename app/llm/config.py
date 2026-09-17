"""LLM configuration from environment variables (optionally loaded from ``.env``).

Required:  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
Optional:  LLM_TIMEOUT_SECONDS (default 30), LLM_RESPONSE_FORMAT (json_schema | json_object),
           LLM_REASONING_EFFORT (sent as ``reasoning_effort`` only when set, e.g. "none" for
           reasoning models whose hidden thinking would otherwise use up the output budget)

Nothing here hardcodes a provider, URL or model. The API key never appears in
``repr``, logs or error messages.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

REQUIRED_VARIABLES = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
ResponseFormat = Literal["json_schema", "json_object"]


class LLMConfigError(RuntimeError):
    """LLM configuration is missing or invalid (message never includes secret values)."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    response_format: ResponseFormat = "json_schema"
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, load_dotenv_file: bool = True) -> "LLMConfig":
        """Read configuration from ``env`` (default: ``os.environ``, after loading ``.env``).

        Raises:
            LLMConfigError: listing missing variable names or the invalid setting.
        """
        if env is None:
            if load_dotenv_file:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            env = os.environ

        missing = [name for name in REQUIRED_VARIABLES if not env.get(name, "").strip()]
        if missing:
            raise LLMConfigError(f"Missing LLM configuration: {', '.join(missing)}. Set them in .env (see .env.example).")

        base_url = env["LLM_BASE_URL"].strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise LLMConfigError("LLM_BASE_URL must start with http:// or https://")

        try:
            timeout = float(env.get("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be a number") from None
        if timeout <= 0:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be positive")

        response_format = env.get("LLM_RESPONSE_FORMAT", "json_schema").strip() or "json_schema"
        if response_format not in ("json_schema", "json_object"):
            raise LLMConfigError("LLM_RESPONSE_FORMAT must be 'json_schema' or 'json_object'")

        return cls(
            api_key=env["LLM_API_KEY"].strip(),
            base_url=base_url,
            model=env["LLM_MODEL"].strip(),
            timeout_seconds=timeout,
            response_format=response_format,  # type: ignore[arg-type]
            reasoning_effort=env.get("LLM_REASONING_EFFORT", "").strip() or None,
        )
