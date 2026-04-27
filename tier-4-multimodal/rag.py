"""Tier 4 — RAG-Anything builder + OpenRouter-routed LLM/vision/embed funcs.

Pattern 1 from 130-RESEARCH.md: three async closures that route through the
OpenRouter unified gateway (single ``OPENROUTER_API_KEY``) using the
LightRAG OpenAI-compatible helpers (``openai_complete_if_cache`` +
``openai_embed``). RAG-Anything composes a LightRAG instance internally, so
the closures here also feed Tier 3's cost-tracking adapter shape (Plan 03's
``CostAdapter`` precedent — see ``cost_adapter.py``).

Locked invariants
-----------------

* ``WORKING_DIR = "rag_anything_storage/tier-4-multimodal"`` — Tier 4 NEVER
  writes to ``lightrag_storage/`` (Tier 3's path). Mixing storage roots
  cross-contaminates entity/embedding indices (anti-pattern).
* ``EMBED_DIMS = 1536`` — pinned at module level, NOT a CLI flag. LightRAG
  indexes vectors at first ingest; switching dim silently corrupts retrieval
  (Pitfall 4, inherited from Tier 3 / HKUDS issue #2119). ``--reset`` is the
  only path to a new dim.
* ``DEFAULT_VISION_MODEL == DEFAULT_LLM_MODEL`` — Gemini 2.5 Flash is itself
  multimodal, so a single slug serves both text-only and vision passes.
* ``_vision_func`` honors BOTH ``messages=`` and ``image_data=`` shapes
  (Pitfall 3 — RAG-Anything dispatches differently for inline-image content
  list ingest vs. multimodal LLM cache reuse).
* All ``os.environ["OPENROUTER_API_KEY"]`` reads happen INSIDE the async
  closures so the module imports cleanly without the key set (lazy env
  read; mirrors Phase 129 Plan 03 ``tier-3-graph/rag.py``).
"""
from __future__ import annotations

import os

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ----------------------------------------------------------------------------
# Locked module-level constants (do NOT promote to CLI flags without --reset)
# ----------------------------------------------------------------------------

WORKING_DIR = "rag_anything_storage/tier-4-multimodal"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"
DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"  # multimodal — same slug
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIMS = 1536  # Pitfall 4 — pinned; index corruption on change
EMBED_MAX_TOKENS = 8192


# ----------------------------------------------------------------------------
# OpenRouter-routed async closures
# ----------------------------------------------------------------------------


async def _llm_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    model=None,
    **kwargs,
):
    """Text-only LLM closure routed through OpenRouter."""
    return await openai_complete_if_cache(
        model or os.environ.get("TIER4_LLM_MODEL", DEFAULT_LLM_MODEL),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def _vision_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    image_data=None,
    messages=None,
    model=None,
    **kwargs,
):
    """Vision LLM closure with the two-shape contract (Pitfall 3).

    RAG-Anything calls this in two distinct shapes:

    * ``messages=[...]`` — pre-baked OpenAI chat-completions message list
      (used during multimodal LLM cache reuse).
    * ``image_data=<data-url>`` — raw image data URL that we wrap into an
      OpenAI ``image_url`` content block ourselves (used during fresh
      content-list image ingest from ``insert_content_list``).

    If neither is provided, fall back to the text-only path so callers that
    accidentally invoke ``vision_model_func`` for a text prompt still work.
    """
    if messages:
        return await openai_complete_if_cache(
            model or DEFAULT_VISION_MODEL,
            "",
            messages=messages,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            **kwargs,
        )
    if image_data:
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ]
        return await openai_complete_if_cache(
            model or DEFAULT_VISION_MODEL,
            "",
            messages=msg,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            **kwargs,
        )
    # No image — degrade gracefully to the text-only LLM closure.
    return await _llm_func(prompt, system_prompt, history_messages, **kwargs)


async def _embed_func(texts):
    """Embedding closure routed through OpenRouter."""
    return await openai_embed(
        texts,
        model=DEFAULT_EMBED_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE,
    )


# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------


def build_rag(working_dir: str = WORKING_DIR) -> RAGAnything:
    """Construct a RAG-Anything instance with the locked Tier 4 contract.

    Notes
    -----
    * Constructor is local — no network calls. Safe to invoke without
      ``OPENROUTER_API_KEY`` set; the closures only read the env at call
      time.
    * RAG-Anything's constructor (via the underlying LightRAG it composes)
      creates ``working_dir`` and several storage files as a side effect at
      build time. Tests should pass a ``tmp_path`` to isolate this.
    """
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    return RAGAnything(
        config=config,
        llm_model_func=_llm_func,
        vision_model_func=_vision_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIMS,
            max_token_size=EMBED_MAX_TOKENS,
            func=_embed_func,
        ),
    )
