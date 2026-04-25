"""Gemini-backed embedding client (google-genai unified SDK).

Public surface
--------------
``get_embedding_client()``
    Factory returning a process-wide singleton implementing the
    :class:`EmbeddingClient` protocol.

``EmbeddingClient.embed(text, model=None)``
    Accepts a string or list of strings, always returns ``list[list[float]]``
    (a list of vectors, one per input).

The client lazy-imports ``shared.config.get_settings()`` so importing this
module does NOT require ``GEMINI_API_KEY``. ValidationError is raised at
:func:`get_embedding_client` call time if ``.env`` is missing — that is the
documented contract per RESEARCH.md Pattern 5.
"""
from __future__ import annotations

from typing import Protocol

from google import genai


class EmbeddingClient(Protocol):
    """Minimal embedding client surface used across all tiers."""

    def embed(
        self, text: str | list[str], model: str | None = None
    ) -> list[list[float]]:  # pragma: no cover — Protocol
        ...


class _GeminiEmbeddingClient:
    """Wraps ``google.genai.Client.models.embed_content``."""

    def __init__(self) -> None:
        from .config import get_settings

        settings = get_settings()
        self._default_model = settings.default_embedding_model
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def embed(
        self, text: str | list[str], model: str | None = None
    ) -> list[list[float]]:
        chosen_model = model or self._default_model
        contents = [text] if isinstance(text, str) else list(text)
        response = self._client.models.embed_content(
            model=chosen_model, contents=contents
        )
        # Always return list[list[float]] for predictable downstream handling.
        return [list(e.values) for e in response.embeddings]


# Process-wide singleton — populated lazily by ``get_embedding_client()``.
_client_cache: _GeminiEmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Return a process-wide Gemini-backed :class:`EmbeddingClient`.

    Raises ``pydantic.ValidationError`` if ``GEMINI_API_KEY`` is unset (via
    the lazy ``shared.config.get_settings()`` factory).
    """
    global _client_cache
    if _client_cache is None:
        _client_cache = _GeminiEmbeddingClient()
    return _client_cache
