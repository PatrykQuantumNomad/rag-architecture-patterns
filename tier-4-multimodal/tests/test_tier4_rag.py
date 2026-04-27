"""Non-live unit tests for tier-4-multimodal/rag.py constants + builder.

No ``OPENROUTER_API_KEY`` required — closures read env LAZILY at call time
(verified during Plan 02 Task 1 smoke imports). The ``build_rag`` constructor
DOES create files on disk under ``working_dir`` (RAGAnything composes a
LightRAG instance which writes a graph .graphml + 3 vdb_*.json placeholders
at construct time). Tests pass ``tmp_path`` to isolate that side effect.

Pinned invariants asserted here
-------------------------------
* ``WORKING_DIR == "rag_anything_storage/tier-4-multimodal"`` — Tier 4 NEVER
  writes to ``lightrag_storage/`` (Tier 3's path).
* ``EMBED_DIMS == 1536`` — Pitfall 4 (LightRAG indexes vectors at first
  ingest; dim change silently corrupts retrieval).
* All 3 model slugs are OpenRouter-shape (``provider/model``).
"""
from __future__ import annotations

from tier_4_multimodal.rag import (
    build_rag,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_VISION_MODEL,
    EMBED_DIMS,
    WORKING_DIR,
)


def test_locked_constants():
    """All 5 module constants pinned to their decision-locked values."""
    assert WORKING_DIR == "rag_anything_storage/tier-4-multimodal"
    assert DEFAULT_LLM_MODEL == "google/gemini-2.5-flash"
    assert DEFAULT_VISION_MODEL == "google/gemini-2.5-flash"
    assert DEFAULT_EMBED_MODEL == "openai/text-embedding-3-small"
    assert EMBED_DIMS == 1536
    # Sanity-check OpenRouter slug shape on all 3 model strings.
    for slug in (DEFAULT_LLM_MODEL, DEFAULT_VISION_MODEL, DEFAULT_EMBED_MODEL):
        assert "/" in slug, f"{slug!r} should be an OpenRouter provider/model slug"


def test_build_rag_constructs(tmp_path):
    """RAGAnything builds without API calls (constructor is local).

    No ``OPENROUTER_API_KEY`` required — env reads are inside the closures.
    The constructor creates ``working_dir`` as a side effect (LightRAG
    writes graph + vdb placeholder files at construct time).
    """
    working = tmp_path / "tier-4-test"
    rag = build_rag(working_dir=str(working))
    assert rag is not None
    assert working.exists(), "build_rag should create working_dir on disk"
