"""Tier 3: LightRAG initialization with OpenRouter routing.

This module wires LightRAG's ``llm_model_func`` and ``embedding_func`` to
OpenRouter — the same gateway Tier 1 uses post-Phase-128. Embedding dim
is locked at 1536 to match Tier 1's index size and avoid HKUDS issue
#2119 (embedding-dimension mismatch corrupts retrieval at first ingest).

Constants here are the source of truth for Tier 3:

* ``WORKING_DIR``        — graph + KV stores live under ``lightrag_storage/tier-3-graph/``
* ``OPENROUTER_BASE``    — same gateway as Tier 1 (Phase 128 pivot)
* ``DEFAULT_LLM_MODEL``  — ``google/gemini-2.5-flash`` (cheapest viable; ≥128K context)
* ``DEFAULT_EMBED_MODEL`` — ``openai/text-embedding-3-small`` (1536 dim — locked)
* ``EMBED_DIMS``         — 1536 (NEVER change without ``--reset``; Pitfall 4)
* ``EMBED_MAX_TOKENS``   — 8192

Pitfall notes (Phase 129 RESEARCH):

* **Pitfall 4 — Embedding dim is HARDCODED at module level.** LightRAG indexes
  vectors at first ingest; if the embedding dimension changes between runs,
  retrieval silently corrupts (HKUDS issue #2119). Do NOT expose
  ``--embed-model`` as a CLI flag. ``--reset`` is the only path to a new dim.
* **Pitfall 5 — LLM context floor.** Tier 3 entity extraction requires ≥32K
  context. ``google/gemini-2.5-flash`` (128K) and ``openai/gpt-4o-mini`` (128K)
  both qualify. Smaller models (Llama-3-8B 8K) will 400 on the gleaning step.
* **Pitfall 8 — Working dir isolation.** ``lightrag_storage/tier-3-graph/`` is
  exclusive to Tier 3. Other tiers (Phase 130) MUST use their own dir.
* **Pitfall 9 — Pin lightrag-hku exactly.** ``[tier-3]`` extras pin
  ``lightrag-hku==1.4.15`` (Plan 129-01). Re-run
  ``scripts/probe_lightrag_token_tracker.py`` after any version bump and
  re-validate the ``CostAdapter`` shape.

The ``OPENROUTER_API_KEY`` is read from ``os.environ`` INSIDE the closures
(not captured at import time) so this module imports cleanly without an API
key — see ``test_rag.py`` for the no-key import contract.

Probe-validated cost-tracking integration (Phase 129 Plan 03 Task 1):
    Both ``openai_complete_if_cache`` AND ``openai_embed`` on
    lightrag-hku==1.4.15 accept ``token_tracker`` kwargs and call
    ``token_tracker.add_usage(token_counts)``. ``build_rag(llm_token_tracker=...)``
    threads a single ``CostAdapter`` instance through both, capturing both
    LLM and embedding cost without any monkey-patching.
"""
from __future__ import annotations

import os
from typing import Any

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "lightrag_storage/tier-3-graph"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIMS = 1536  # Pitfall 4 — locked at first ingest; --reset to change.
EMBED_MAX_TOKENS = 8192  # Conservative ceiling — gemini-2.5-flash supports far more.

_MIN_CONTEXT_NOTE = (
    "Tier 3 LLMs need >=32K context (Pitfall 5). "
    "google/gemini-2.5-flash and openai/gpt-4o-mini both qualify; "
    "smaller models (Llama-3-8B 8K context) will 400 on entity extraction."
)


def _make_llm_func(model: str | None, token_tracker: Any | None):
    """Build an async LLM closure that routes through OpenRouter.

    The closure reads ``OPENROUTER_API_KEY`` from ``os.environ`` LAZILY (per
    call) so module-level import does not require a key. Caller-supplied
    ``model`` overrides the default; ``$TIER3_LLM_MODEL`` is the second
    fallback before ``DEFAULT_LLM_MODEL``.
    """

    async def _llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        return await openai_complete_if_cache(
            model or os.environ.get("TIER3_LLM_MODEL", DEFAULT_LLM_MODEL),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )

    return _llm_model_func


def _make_embed_func(token_tracker: Any | None):
    """Build an async embedding closure that routes through OpenRouter.

    Probe in Plan 03 Task 1 confirmed ``openai_embed`` ALSO accepts a
    ``token_tracker`` kwarg on lightrag-hku==1.4.15, so we thread the same
    ``CostAdapter`` instance used for LLM calls — capturing embedding cost
    too. The embed-side ``add_usage`` payload omits ``completion_tokens``;
    the adapter dispatches on its absence to ``record_embedding``.
    """

    async def _embed_func(texts: list[str]):
        return await openai_embed(
            texts,
            model=DEFAULT_EMBED_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            token_tracker=token_tracker,
        )

    return _embed_func


def build_rag(
    working_dir: str = WORKING_DIR,
    llm_token_tracker: Any | None = None,
    model: str | None = None,
) -> LightRAG:
    """Build a ``LightRAG`` instance with OpenRouter-routed LLM + embed funcs.

    The constructor itself does NOT make any network calls — it just stores
    the closures and prepares the on-disk ``working_dir`` lazily. First
    network activity is on ``await rag.ainsert(...)`` (ingest, Plan 05) or
    ``await rag.aquery(...)`` (query, Plan 05).

    Parameters
    ----------
    working_dir
        Where LightRAG persists graph + KV stores. Defaults to
        ``lightrag_storage/tier-3-graph`` (Pitfall 8 — exclusive to Tier 3).
    llm_token_tracker
        Optional ``CostAdapter`` (or anything with ``.add_usage(dict)``).
        Threaded into BOTH the LLM closure and the embedding closure so
        every LightRAG-driven LLM/embed call records cost. Pass ``None``
        in tests where you don't want cost recording.
    model
        Override the default LLM model. Otherwise the closure reads
        ``$TIER3_LLM_MODEL`` and falls back to ``DEFAULT_LLM_MODEL``.

    Notes
    -----
    Phase 128 narrative continuity: Tier 3 reuses Tier 1's OpenRouter key
    and embedding model (1536-dim ``openai/text-embedding-3-small``). The
    two tiers' on-disk indexes are stored in different directories
    (``chroma_db/tier-1-naive/`` vs ``lightrag_storage/tier-3-graph/``)
    so they do not collide.
    """
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=_make_llm_func(model=model, token_tracker=llm_token_tracker),
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIMS,
            max_token_size=EMBED_MAX_TOKENS,
            func=_make_embed_func(token_tracker=llm_token_tracker),
        ),
    )


__all__ = [
    "build_rag",
    "WORKING_DIR",
    "OPENROUTER_BASE",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBED_MODEL",
    "EMBED_DIMS",
    "EMBED_MAX_TOKENS",
]
