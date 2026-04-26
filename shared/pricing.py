"""Per-model USD/1M-token pricing table for the rag-architecture-patterns repo.

Values are sourced from 127-RESEARCH.md (current as of 2026-04-25) and locked
here so downstream cost tracking is reproducible (D-13 stability).

Schema
------
``PRICES[model]`` is a dict with keys:

* ``input``   — USD per 1M input tokens.
* ``output``  — USD per 1M output tokens (0.0 for embedding-only models).
* ``cache``   — USD per 1M cached input tokens (chat/Gemini models only).

``PRICING_DATE`` is the ISO 8601 date these values were captured. Update both
fields together when a vendor publishes new prices.
"""
from __future__ import annotations

# ISO 8601 date the prices were last verified against vendor documentation.
PRICING_DATE: str = "2026-04-25"

# USD per 1,000,000 tokens.
PRICES: dict[str, dict[str, float]] = {
    # Google Gemini (https://ai.google.dev/pricing)
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cache": 0.03},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cache": 0.125},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
    # OpenAI (https://openai.com/api/pricing/)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    # OpenRouter (https://openrouter.ai/models) — slugs use {provider}/{model}.
    # Prices match the underlying-provider rates that OpenRouter passes through;
    # verify against https://openrouter.ai/<provider>/<model> before adding new.
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "openai/text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cache": 0.03},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cache": 0.125},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
}
