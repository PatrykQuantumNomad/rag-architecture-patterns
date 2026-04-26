"""Tier 1: OpenAI embedding wrapper (text-embedding-3-small).

Local to Tier 1 — does NOT modify shared/embeddings.py (Gemini-only by
Phase 127 contract; smoke test depends on that contract).

Cost is recorded into shared.cost_tracker.CostTracker from inside
embed_batch — pre-computing embeddings (rather than registering an
OpenAIEmbeddingFunction on the ChromaDB collection) keeps every API
call auditable (128-RESEARCH.md Pitfall 5).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from shared.cost_tracker import CostTracker

# Locked by 128-RESEARCH.md "Standard Stack" — 1536d default, $0.02/1M tokens.
EMBED_MODEL: str = "text-embedding-3-small"
EMBED_DIMS: int = 1536


def build_openai_client() -> OpenAI:
    """Construct an OpenAI client from shared.config settings.

    Raises SystemExit (clear, friendly message) if OPENAI_API_KEY is unset
    — see 128-RESEARCH.md Pitfall 10. We use SystemExit (not RuntimeError)
    so the CLI exits cleanly without a stack-trace at the user level.
    """
    from shared.config import get_settings

    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit(
            "OPENAI_API_KEY required for Tier 1 (embeddings). "
            "Copy .env.example to .env and set your key from "
            "https://platform.openai.com/api-keys"
        )
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def embed_batch(
    client: OpenAI, texts: list[str], tracker: "CostTracker"
) -> list[list[float]]:
    """Embed `texts` in a single OpenAI request and record the cost.

    Returns a list[list[float]] of length len(texts), each 1536-dim.

    The caller is responsible for chunking the overall corpus into
    <=100-text batches (128-RESEARCH.md Pitfall 6) — this function does
    exactly ONE API request per call.
    """
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # OpenAI returns usage.prompt_tokens reliably for embeddings.
    tracker.record_embedding(EMBED_MODEL, int(resp.usage.prompt_tokens))
    return [list(d.embedding) for d in resp.data]
