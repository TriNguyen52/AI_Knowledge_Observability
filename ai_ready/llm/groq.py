"""Groq LLM provider implementation.

Uses the Groq SDK to call Groq-hosted models (llama-3.3-70b, mixtral, etc.).
Groq provides fast inference for open-source models.
"""

from __future__ import annotations

import time
from typing import Any

from ai_ready.llm.base import LLMMessage, LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """Groq-backed LLM provider.

    Calls Groq's chat completions API via the groq SDK.
    Requires GROQ_API_KEY environment variable or api_key parameter.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "llama-3.3-70b-versatile",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._kwargs = kwargs
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Groq client lazily."""
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )
        except Exception as e:
            # Client may fail if no API key — defer error to chat() call
            self._client = None
            self._init_error = e

    def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Groq."""
        if self._client is None:
            self._init_client()
            if self._client is None:
                raise RuntimeError(f"Groq client initialization failed: {getattr(self, '_init_error', 'unknown')}")

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

    def stream_chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """Stream a chat completion from Groq."""
        if self._client is None:
            self._init_client()
            if self._client is None:
                raise RuntimeError(f"Groq client initialization failed: {getattr(self, '_init_error', 'unknown')}")

        model_name = model or self._default_model
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]

        stream = self._client.chat.completions.create(
            model=model_name,
            messages=message_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **{**self._kwargs, **kwargs},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def default_model(self) -> str:
        return self._default_model
