"""Ollama LLM provider implementation.

Uses the Ollama HTTP API to call locally-hosted models.
No API key required — Ollama runs locally.
"""

from __future__ import annotations

import time
from typing import Any

from ai_ready.llm.base import LLMMessage, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Ollama-backed LLM provider for local models.

    Calls Ollama's chat API via HTTP. No API key needed.
    Requires Ollama to be running locally (default: http://localhost:11434).
    """

    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "llama3",
        **kwargs: Any,
    ) -> None:
        self._host = host
        self._default_model = default_model
        self._kwargs = kwargs

    def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        import json
        import urllib.request

        model_name = model or self._default_model
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]

        payload = json.dumps({
            "model": model_name,
            "messages": message_dicts,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **self._kwargs,
                **kwargs,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.monotonic()
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=model_name,
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            raw=data,
        )

    @property
    def default_model(self) -> str:
        return self._default_model
