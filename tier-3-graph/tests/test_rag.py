"""Non-live unit tests for ``tier_3_graph.rag`` and ``tier_3_graph.cost_adapter``.

These tests do NOT require an API key and do NOT make network calls. They
verify:

1. Locked constants (Pitfall 4 — embedding dim is the highest-stakes value;
   if it ever changes accidentally, downstream re-ingest is required).
2. ``build_rag`` constructs a ``LightRAG`` instance whose embedding dim
   matches ``EMBED_DIMS=1536``.
3. ``CostAdapter.add_usage`` dispatches LLM-shape vs embed-shape token
   payloads correctly into ``CostTracker.record_llm`` /
   ``CostTracker.record_embedding``.
4. The module imports cleanly without ``OPENROUTER_API_KEY`` set (the
   closures read the env LAZILY).

Live ingest+query end-to-end is exercised in ``test_main_live.py`` (Plan 05).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


# Plan 128-02 follow-on convention: tests use ``importlib.util`` paths via the
# ``tier_3_graph`` shim, NOT a hyphenated direct import. The shim handles
# loading rag.py / cost_adapter.py from ``tier-3-graph/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_locked_constants_match_research_pattern_4() -> None:
    """RESEARCH Pattern 4 + Pitfall 4 — these constants are the source of truth.

    If a future plan needs to change any of them, it MUST also update the
    plan's frontmatter ``files_modified`` to include ``rag.py`` and
    document the cost / corruption implications.
    """
    from tier_3_graph.rag import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLM_MODEL,
        EMBED_DIMS,
        EMBED_MAX_TOKENS,
        OPENROUTER_BASE,
        WORKING_DIR,
    )

    assert WORKING_DIR == "lightrag_storage/tier-3-graph"
    assert OPENROUTER_BASE == "https://openrouter.ai/api/v1"
    assert DEFAULT_LLM_MODEL == "google/gemini-2.5-flash"
    assert DEFAULT_EMBED_MODEL == "openai/text-embedding-3-small"
    assert EMBED_DIMS == 1536, (
        "Pitfall 4: EMBED_DIMS is locked at first ingest. Changing it "
        "silently corrupts retrieval (HKUDS issue #2119). Use --reset to "
        "rebuild the index against a new dim."
    )
    assert EMBED_MAX_TOKENS == 8192


def test_module_imports_without_api_key() -> None:
    """The module MUST import even when no OPENROUTER_API_KEY is set.

    The closures inside ``_make_llm_func`` / ``_make_embed_func`` read the
    env LAZILY — only at call time, not at import time. A user who only
    runs ``--reset`` (no ingest, no query) should not be required to have
    a key.

    Strategy: load ``tier-3-graph/rag.py`` via ``importlib.util`` directly
    (NOT through the ``tier_3_graph`` shim), so we observe the bare-module
    import contract independent of whatever sys.modules state earlier tests
    may have left behind. ``importlib.reload`` does NOT play nicely with
    shim-loaded modules (the loader was a one-shot
    ``spec_from_file_location`` and reloading raises ``ModuleNotFoundError:
    spec not found``).
    """
    import importlib.util as _iutil

    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        rag_src = _REPO_ROOT / "tier-3-graph" / "rag.py"
        spec = _iutil.spec_from_file_location("_tier3_rag_smoke", rag_src)
        assert spec is not None and spec.loader is not None
        mod = _iutil.module_from_spec(spec)
        spec.loader.exec_module(mod)  # Must not raise without an API key.
        assert hasattr(mod, "build_rag")
        assert mod.EMBED_DIMS == 1536
    finally:
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved


def test_build_rag_constructs_with_locked_embedding_dim() -> None:
    """``build_rag`` returns a LightRAG instance whose embedding dim is 1536."""
    from tier_3_graph.rag import EMBED_DIMS, build_rag

    with tempfile.TemporaryDirectory() as td:
        rag = build_rag(working_dir=td)

    assert rag.embedding_func.embedding_dim == EMBED_DIMS == 1536
    assert rag.llm_model_func is not None
    assert rag.embedding_func is not None


def test_cost_adapter_dispatches_llm_payload_to_record_llm() -> None:
    """LLM-shape payload (has completion_tokens) routes to record_llm."""
    from shared.cost_tracker import CostTracker

    from tier_3_graph.cost_adapter import CostAdapter

    tracker = CostTracker("tier-3")
    adapter = CostAdapter(
        tracker,
        llm_model="google/gemini-2.5-flash",
        embed_model="openai/text-embedding-3-small",
    )

    # Probe-validated LLM shape on lightrag-hku==1.4.15.
    adapter.add_usage(
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    assert tracker.total_usd() > 0


def test_cost_adapter_dispatches_embed_payload_to_record_embedding() -> None:
    """Embed-shape payload (NO completion_tokens) routes to record_embedding."""
    from shared.cost_tracker import CostTracker

    from tier_3_graph.cost_adapter import CostAdapter

    tracker = CostTracker("tier-3")
    adapter = CostAdapter(
        tracker,
        llm_model="google/gemini-2.5-flash",
        embed_model="openai/text-embedding-3-small",
    )

    # Probe-validated embed shape on lightrag-hku==1.4.15.
    adapter.add_usage({"prompt_tokens": 200, "total_tokens": 200})

    assert tracker.total_usd() > 0


def test_cost_adapter_handles_none_usage_safely() -> None:
    """``add_usage(None)`` is a no-op (LightRAG may emit no-usage payloads on cache hits)."""
    from shared.cost_tracker import CostTracker

    from tier_3_graph.cost_adapter import CostAdapter

    tracker = CostTracker("tier-3")
    adapter = CostAdapter(
        tracker,
        llm_model="google/gemini-2.5-flash",
        embed_model="openai/text-embedding-3-small",
    )

    adapter.add_usage(None)
    assert tracker.total_usd() == 0


def test_cost_adapter_handles_object_style_usage() -> None:
    """Object payloads (e.g. ``response.usage``) work alongside dict payloads."""
    from shared.cost_tracker import CostTracker

    from tier_3_graph.cost_adapter import CostAdapter

    class _MockUsage:
        prompt_tokens = 200
        completion_tokens = 10
        total_tokens = 210

    tracker = CostTracker("tier-3")
    adapter = CostAdapter(
        tracker,
        llm_model="google/gemini-2.5-flash",
        embed_model="openai/text-embedding-3-small",
    )
    adapter.add_usage(_MockUsage())
    assert tracker.total_usd() > 0
