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
* Cost capture: ``build_rag(llm_token_tracker=adapter)`` threads the same
  adapter through BOTH the LLM/vision closure path AND the embedding
  closure (Phase 129 Plan 03 Outcome A — lightrag-hku==1.4.15
  ``openai_complete_if_cache`` + ``openai_embed`` both accept
  ``token_tracker=`` and call ``add_usage(...)``). Plan 130-03 wires this
  via ``main.py``.
"""
from __future__ import annotations

import os
from typing import Any

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
# OpenRouter-routed async closures (factory variants accept token_tracker)
# ----------------------------------------------------------------------------


def _make_llm_func(model: str | None, token_tracker: Any | None):
    """Build the text-only LLM closure with optional ``token_tracker``.

    Mirrors Phase 129 Plan 03 ``tier-3-graph/rag.py::_make_llm_func`` —
    factory pattern lets the model + tracker be late-bound at ``build_rag``
    time without baking them into module-level globals.
    """

    async def _llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        return await openai_complete_if_cache(
            model or os.environ.get("TIER4_LLM_MODEL", DEFAULT_LLM_MODEL),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )

    return _llm_func


def _make_vision_func(model: str | None, token_tracker: Any | None):
    """Build the vision LLM closure with the two-shape contract (Pitfall 3).

    RAG-Anything calls vision in two distinct shapes:

    * ``messages=[...]`` — pre-baked OpenAI chat-completions message list
      (used during multimodal LLM cache reuse).
    * ``image_data=<data-url>`` — raw image data URL that we wrap into an
      OpenAI ``image_url`` content block ourselves (used during fresh
      content-list image ingest from ``insert_content_list``).

    If neither is provided, fall back to the text-only path so callers that
    accidentally invoke ``vision_model_func`` for a text prompt still work.
    """
    text_only = _make_llm_func(model=model, token_tracker=token_tracker)

    async def _vision_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return await openai_complete_if_cache(
                model or DEFAULT_VISION_MODEL,
                "",
                messages=messages,
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=OPENROUTER_BASE,
                token_tracker=token_tracker,
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
                token_tracker=token_tracker,
                **kwargs,
            )
        # No image — degrade gracefully to the text-only LLM closure.
        return await text_only(prompt, system_prompt, history_messages, **kwargs)

    return _vision_func


def _make_embed_func(token_tracker: Any | None):
    """Build the embedding closure threaded with optional ``token_tracker``.

    Phase 129 Plan 03 probe-validated that lightrag-hku==1.4.15's
    ``openai_embed`` accepts ``token_tracker=`` and calls ``add_usage(...)``
    after each batch (with no ``completion_tokens`` key — the
    ``CostAdapter`` dispatches that case to ``record_embedding``).
    """

    async def _embed_func(texts):
        return await openai_embed(
            texts,
            model=DEFAULT_EMBED_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE,
            token_tracker=token_tracker,
        )

    return _embed_func


# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------


def build_rag(
    working_dir: str = WORKING_DIR,
    llm_token_tracker: Any | None = None,
    model: str | None = None,
) -> RAGAnything:
    """Construct a RAG-Anything instance with the locked Tier 4 contract.

    Parameters
    ----------
    working_dir
        Where RAG-Anything (and its embedded LightRAG) persists graph + KV
        stores. Defaults to ``rag_anything_storage/tier-4-multimodal``.
    llm_token_tracker
        Optional ``CostAdapter`` (or anything with ``.add_usage(dict)``).
        Threaded into BOTH the LLM/vision closures AND the embedding
        closure so every LightRAG-driven LLM/embed call records cost.
        Pass ``None`` in tests where you don't want cost recording.
    model
        Override the default LLM/vision model slug. Otherwise the closure
        reads ``$TIER4_LLM_MODEL`` and falls back to ``DEFAULT_LLM_MODEL``.

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
        llm_model_func=_make_llm_func(model=model, token_tracker=llm_token_tracker),
        vision_model_func=_make_vision_func(
            model=model, token_tracker=llm_token_tracker
        ),
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
    "DEFAULT_VISION_MODEL",
    "DEFAULT_EMBED_MODEL",
    "EMBED_DIMS",
    "EMBED_MAX_TOKENS",
]
