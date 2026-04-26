"""Tier 1: embedding wrapper routed through OpenRouter.

Filename retained from Plan 128-03 for compatibility, but the client is now
the OpenAI Python SDK pointed at the OpenRouter gateway
(https://openrouter.ai/api/v1) with ``OPENROUTER_API_KEY``. We keep the
underlying model id ``openai/text-embedding-3-small`` so the on-disk
ChromaDB index dimensions (1536) and Tier 5's reuse contract are unchanged.

Local to Tier 1 — does NOT modify ``shared/embeddings.py`` (Gemini-only by
Phase 127 contract; smoke test depends on that contract).

Cost is recorded into ``shared.cost_tracker.CostTracker`` from inside
``embed_batch`` — pre-computing embeddings (rather than registering an
OpenAIEmbeddingFunction on the ChromaDB collection) keeps every API call
auditable (128-RESEARCH.md Pitfall 5).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from shared.cost_tracker import CostTracker

# OpenRouter gateway — the OpenAI SDK is fully compatible with this base URL.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# OpenRouter model slug (provider/model). Underlying provider is OpenAI; the
# 1536-dim vector shape is what the ChromaDB index was built against.
EMBED_MODEL: str = "openai/text-embedding-3-small"
EMBED_DIMS: int = 1536


def build_openai_client() -> OpenAI:
    """Construct an OpenAI SDK client pointed at OpenRouter.

    Raises ``SystemExit`` (clear, friendly message) if ``OPENROUTER_API_KEY``
    is unset — see 128-RESEARCH.md Pitfall 10. We use ``SystemExit`` (not
    ``RuntimeError``) so the CLI exits cleanly without a stack trace at the
    user level.

    The factory name is retained from Plan 128-03; the construction is what
    changed (base_url + key source). Tier-1 callers keep importing
    ``build_openai_client`` unchanged.
    """
    from shared.config import get_settings

    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise SystemExit(
            "OPENROUTER_API_KEY required for Tier 1 (chat + embeddings). "
            "Copy .env.example to .env and set your key from "
            "https://openrouter.ai/keys"
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=settings.openrouter_api_key.get_secret_value(),
    )


def embed_batch(
    client: OpenAI, texts: list[str], tracker: "CostTracker"
) -> list[list[float]]:
    """Embed ``texts`` in a single OpenRouter request and record the cost.

    Returns a ``list[list[float]]`` of length ``len(texts)``, each 1536-dim.

    The caller is responsible for chunking the overall corpus into <=100-text
    batches (128-RESEARCH.md Pitfall 6) — this function does exactly ONE API
    request per call. OpenRouter passes the request through to OpenAI; the
    response shape (``data[i].embedding`` + ``usage.prompt_tokens``) matches
    the native OpenAI embeddings API exactly, so the SDK's parsing works
    without modification.
    """
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    tracker.record_embedding(EMBED_MODEL, int(resp.usage.prompt_tokens))
    return [list(d.embedding) for d in resp.data]
