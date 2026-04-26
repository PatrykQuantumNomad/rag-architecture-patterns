"""Tier 1: chat completion routed through OpenRouter.

Local to Tier 1 — does NOT modify ``shared/llm.py`` (Gemini-only by Phase 127
contract; smoke test depends on that contract). Tier 1 picks its own chat
provider so users can compare models (e.g. ``--model anthropic/claude-haiku-4.5``
or ``--model openai/gpt-4o-mini``) without touching the shared layer.

The OpenAI Python SDK is fully compatible with OpenRouter when pointed at
``https://openrouter.ai/api/v1``; the request/response shape mirrors OpenAI's
native chat completions API.

Cost is recorded into ``shared.cost_tracker.CostTracker`` via
``record_llm(model, input_tokens, output_tokens)``; the model slug must be
present in ``shared.pricing.PRICES`` (extend that table when adopting new
OpenRouter models — see PRICING_DATE for the verification cadence).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

from .embed_openai import OPENROUTER_BASE_URL

if TYPE_CHECKING:
    from shared.cost_tracker import CostTracker


# Default OpenRouter chat model. Picked for narrative continuity with
# Phase 127 (Gemini 2.5 Flash) and lowest cost among general-purpose models.
# Override at the CLI with ``--model``.
DEFAULT_CHAT_MODEL: str = "google/gemini-2.5-flash"


@dataclass(frozen=True)
class ChatResponse:
    """Result of a single OpenRouter chat completion.

    Mirrors the field shape of ``shared.llm.LLMResponse`` so callers (e.g.
    ``main.cmd_query``) can swap providers without touching their use sites.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str


def build_chat_client() -> OpenAI:
    """Construct an OpenAI SDK client pointed at OpenRouter for chat completions.

    Raises ``SystemExit`` if ``OPENROUTER_API_KEY`` is unset (Pitfall 10).
    Shares construction with the embedding client — both go through the same
    OpenRouter base URL with the same key, but kept separate functions for
    clarity and so test seams remain narrow.
    """
    from shared.config import get_settings

    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit(
            "OPENROUTER_API_KEY required for Tier 1 chat completion. "
            "Copy .env.example to .env and set your key from "
            "https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=settings.openrouter_api_key.get_secret_value(),
    )


def complete(
    client: OpenAI,
    prompt: str,
    model: str,
    tracker: "CostTracker",
) -> ChatResponse:
    """Run a single chat completion and record the cost.

    Parameters
    ----------
    client
        An OpenAI SDK client built via ``build_chat_client``.
    prompt
        The fully-formed prompt string (Tier 1 uses ``prompt.build_prompt``).
    model
        OpenRouter model slug (e.g. ``"google/gemini-2.5-flash"``,
        ``"anthropic/claude-haiku-4.5"``, ``"openai/gpt-4o-mini"``). Must be
        present in ``shared.pricing.PRICES`` for cost tracking.
    tracker
        Receives ``record_llm(model, input_tokens, output_tokens)``.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = resp.usage
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    tracker.record_llm(model, input_tokens, output_tokens)
    return ChatResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )
