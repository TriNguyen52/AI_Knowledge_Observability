"""Anthropic LLM provider implementation.

Uses the Anthropic SDK to call Claude models.
Requires ANTHROPIC_API_KEY environment variable or api_key parameter.
"""

from __future__ import annotations

import time
from typing import Any

from ai_ready.llm.base import LLMMessage, LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic-backed LLM provider.

    Calls Anthropic's messages API via the anthropic SDK.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._kwargs = kwargs
        self._client = None

    def _init_client(self) -> None:
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
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

        # Anthropic separates system message from conversation messages
        system_content = ""
        conv_messages = []
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                conv_messages.append({"role": m.role, "content": m.content})

        start = time.monotonic()
        response = self._client.messages.create(
            model=model_name,
            system=system_content if system_content else None,
            messages=conv_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**self._kwargs, **kwargs},
        )
        latency_ms = (time.monotonic() - start) * 1000

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = response.usage

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.name,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    @property
    def default_model(self) -> str:
        return self._default_model
