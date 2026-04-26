"""Settings for the rag-architecture-patterns repo.

Per RESEARCH.md Pattern 5, ``GEMINI_API_KEY`` is REQUIRED at validation time.
We use a lazy factory (``get_settings()``) so that simply importing this
module — or any sibling that re-imports ``shared.config`` — does NOT raise
when ``.env`` is absent. Only callers that actually need a key (the LLM /
embedding clients, the smoke test) invoke ``get_settings()``; that call
raises ``pydantic.ValidationError`` if ``GEMINI_API_KEY`` is missing, which
IS the validation contract.

Importantly: do NOT define a module-level ``settings = Settings()`` singleton
— that would break ``tests/smoke_test.py::test_imports`` in fresh checkouts
without ``.env``.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime configuration loaded from ``.env`` + process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # REQUIRED — ValidationError raised by get_settings() if absent. Used by
    # the legacy shared.llm Gemini client and the Phase 127 smoke test.
    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")

    # Optional — only needed for direct-OpenAI use cases (legacy) or curation.
    openai_api_key: SecretStr | None = Field(None, alias="OPENAI_API_KEY")
    s2_api_key: SecretStr | None = Field(None, alias="S2_API_KEY")

    # OpenRouter unified gateway (https://openrouter.ai). REQUIRED for Tier 1+
    # because Tier 1 routes BOTH embeddings and chat completions through it.
    # Optional at validation time so importing shared.config does not require
    # OPENROUTER_API_KEY in environments that only run Phase 127's smoke test.
    openrouter_api_key: SecretStr | None = Field(None, alias="OPENROUTER_API_KEY")

    # Defaults align with the Gemini-first stack the smoke test exercises.
    default_chat_model: str = Field("gemini-2.5-flash", alias="DEFAULT_CHAT_MODEL")
    default_embedding_model: str = Field(
        "gemini-embedding-001", alias="DEFAULT_EMBEDDING_MODEL"
    )
    dataset_root: str = Field("dataset", alias="DATASET_ROOT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide ``Settings`` singleton (lazy).

    Lazy construction is intentional: it lets ``from shared import config``
    succeed at import time even with no ``.env`` in fresh checkouts, while
    preserving the required-field contract — ``ValidationError`` raised here
    propagates to callers that actually need keys.
    """
    return Settings()
