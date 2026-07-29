"""Abstract base class for LLM providers.

Every provider (Groq, OpenAI, Anthropic, Ollama) implements this interface.
Burr actions call LLMGateway, which delegates to the active provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide chat() and optionally stream_chat().
    The provider receives pre-built messages — it never accesses AI-Ready
    data stores, files, or secrets directly.
    """

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of LLMMessage objects (system, user, assistant roles).
            model: Model name to use (provider-specific default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with the generated content and usage metadata.
        """
        ...

    def stream_chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """Stream a chat completion. Optional — providers may override.

        Yields:
            String chunks as they arrive.
        """
        # Default: non-streaming fallback
        response = self.chat(messages, model, temperature, max_tokens, **kwargs)
        yield response.content

    @property
    def default_model(self) -> str:
        """Return the provider's default model name."""
        return ""
