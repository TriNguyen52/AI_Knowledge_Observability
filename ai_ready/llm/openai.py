"""OpenAI LLM provider implementation.

Uses the OpenAI SDK to call GPT models.
Requires OPENAI_API_KEY environment variable or api_key parameter.
"""

from __future__ import annotations

import time
from typing import Any

from ai_ready.llm.base import LLMMessage, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI-backed LLM provider.

    Calls OpenAI's chat completions API via the openai SDK.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._kwargs = kwargs
        self._client = None

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._client is None:
            self._init_client()

        model_name = model or self._default_model
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]

        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=model_name,
            messages=message_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**self._kwargs, **kwargs},
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=model_name,
            provider=self.name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    @property
    def default_model(self) -> str:
        return self._default_model
