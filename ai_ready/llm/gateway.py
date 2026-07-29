"""LLM Gateway — provider-agnostic entry point for LLM calls.

Burr actions call LLMGateway.chat(), never a provider directly.
The gateway routes to the active provider, handles fallback, and
provides a unified interface regardless of backend.

Usage:
    gateway = LLMGateway(provider="groq")
    response = gateway.chat(messages=[
        LLMMessage(role="system", content="You are a knowledge analyst."),
        LLMMessage(role="user", content="Analyze these signals..."),
    ])

Security: The gateway only receives pre-built messages. It never
accesses AI-Ready stores, files, or secrets. The LLM cannot
query the knowledge base or modify documents.
"""

from __future__ import annotations

import os
from typing import Any

from ai_ready.llm.base import LLMMessage, LLMProvider, LLMResponse


class LLMGateway:
    """Provider-agnostic gateway for LLM calls.

    Routes chat requests to the active provider. Supports
    provider switching at runtime and automatic fallback.

    The gateway is the ONLY LLM interface that Burr actions use.
    """

    def __init__(
        self,
        provider: str = "groq",
        api_key: str | None = None,
        default_model: str | None = None,
        **provider_kwargs: Any,
    ) -> None:
        self._provider_name = provider
        self._default_model = default_model
        self._provider: LLMProvider | None = None
        self._provider_kwargs = provider_kwargs
        self._api_key = api_key

        # Lazy-init the provider on first use
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the configured provider."""
        if self._provider_name == "groq":
            from ai_ready.llm.groq import GroqProvider
            api_key = self._api_key or os.environ.get("GROQ_API_KEY", "")
            self._provider = GroqProvider(
                api_key=api_key,
                default_model=self._default_model or "llama-3.3-70b-versatile",
                **self._provider_kwargs,
            )
        elif self._provider_name == "openai":
            from ai_ready.llm.openai import OpenAIProvider
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            self._provider = OpenAIProvider(
                api_key=api_key,
                default_model=self._default_model or "gpt-4o",
                **self._provider_kwargs,
            )
        elif self._provider_name == "anthropic":
            from ai_ready.llm.anthropic import AnthropicProvider
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._provider = AnthropicProvider(
                api_key=api_key,
                default_model=self._default_model or "claude-sonnet-4-20250514",
                **self._provider_kwargs,
            )
        elif self._provider_name == "ollama":
            from ai_ready.llm.ollama import OllamaProvider
            self._provider = OllamaProvider(
                default_model=self._default_model or "llama3",
                **self._provider_kwargs,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider_name}")

    def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat request through the active provider.

        Args:
            messages: Pre-built LLMMessage list (caller constructs prompts).
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with content and usage metadata.
        """
        if self._provider is None:
            self._init_provider()
        assert self._provider is not None
        return self._provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """Stream a chat request through the active provider."""
        if self._provider is None:
            self._init_provider()
        assert self._provider is not None
        yield from self._provider.stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider(self) -> LLMProvider | None:
        return self._provider
