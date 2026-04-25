"""Gemini-backed chat completion client (google-genai unified SDK).

Public surface
--------------
``get_llm_client()``
    Factory returning a process-wide singleton implementing the
    :class:`LLMClient` protocol.

``LLMClient`` (Protocol)
    Has a single method: ``complete(prompt, model=None) -> LLMResponse``.

``LLMResponse``
    Frozen dataclass: ``text``, ``input_tokens``, ``output_tokens``, ``model``.

The client lazy-imports ``shared.config.get_settings()`` so importing this
module does NOT require ``GEMINI_API_KEY``. ValidationError is raised at
:func:`get_llm_client` call time if ``.env`` is missing — that is the
documented contract per RESEARCH.md Pattern 5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from google import genai


@dataclass(frozen=True)
class LLMResponse:
    """Result of a single chat completion."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(Protocol):
    """Minimal LLM client surface used across all tiers."""

    def complete(
        self, prompt: str, model: str | None = None
    ) -> LLMResponse:  # pragma: no cover — Protocol
        ...


class _GeminiLLMClient:
    """Wraps ``google.genai.Client`` with usage-metadata extraction."""

    def __init__(self) -> None:
        from .config import get_settings

        settings = get_settings()
        self._default_model = settings.default_chat_model
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def complete(self, prompt: str, model: str | None = None) -> LLMResponse:
        chosen_model = model or self._default_model
        response = self._client.models.generate_content(
            model=chosen_model, contents=prompt
        )
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        text = response.text or ""
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=chosen_model,
        )


# Process-wide singleton — populated lazily by ``get_llm_client()``.
_client_cache: _GeminiLLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a process-wide Gemini-backed :class:`LLMClient`.

    Raises ``pydantic.ValidationError`` if ``GEMINI_API_KEY`` is unset (via
    the lazy ``shared.config.get_settings()`` factory). Callers should treat
    this as the validation gate rather than swallowing it.
    """
    global _client_cache
    if _client_cache is None:
        _client_cache = _GeminiLLMClient()
    return _client_cache
