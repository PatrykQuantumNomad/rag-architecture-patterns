"""Tier 3: live end-to-end test (real OpenRouter API).

Marked ``@pytest.mark.live`` — skipped automatically by pytest unless run
with ``-m live`` AND ``OPENROUTER_API_KEY`` is present in the environment
(``tier3_live_keys_ok`` fixture handles the skip).

Cost-bounded e2e
================

A full-corpus ingest costs ~$1 (~420 chunks × per-chunk LLM entity
extraction). This test hard-caps the input at the **2 smallest papers**
on disk to keep cost in the ~$0.05–0.15 range and latency in the ~60–180s
range — measurable but cheap enough to repeat per phase verification.

The test exercises every Phase 129 ROADMAP success criterion for Tier 3
in a single run:

  1. ``python tier-3-graph/main.py`` runs end-to-end
     (we call ``ingest_corpus`` + ``run_query`` directly — the same code
     paths invoked by the CLI entrypoint after argparse + the cost-surprise
     gate, which we bypass programmatically with explicit ``--yes``-style
     setup since pytest non-interactivity already short-circuits the prompt
     via the EOFError abort path).
  2. The graph persists to disk and can be inspected
     (we assert ``graph_chunk_entity_relation.graphml`` exists with
     non-trivial size after ingest — the file is created at constructor
     init as an empty NetworkX placeholder, but a successful entity
     extraction grows it well past 1KB).
  3. Cost and latency are recorded
     (we assert ``tracker.total_usd() > 0`` after the query — captured by
     the ``CostAdapter`` threaded through ``build_rag(llm_token_tracker=)``
     against both the LLM closure AND the embedding closure on
     lightrag-hku==1.4.15, per Plan 03 probe).

Isolation
=========

* ``working_dir`` is set to ``tmp_path / "lightrag_test"`` so the test
  NEVER touches the user's production graph at
  ``lightrag_storage/tier-3-graph/`` (Pitfall 8 — collection collisions;
  Pitfall 10 — destructive operations on user state).
* ``CostTracker("tier-3-test")`` writes its persisted JSON to
  ``evaluation/results/costs/tier-3-test-<ts>.json`` (separate from the
  production ``tier-3-<ts>.json`` files).
* The 2 papers chosen are the smallest by file size on disk so the cost
  estimate stays predictable. If a future paper-set rotation introduces
  papers smaller than the current "smallest 2", the test still picks the
  smallest 2 — bounded cost is preserved.

This test is the **Phase 129 close-out gate** for Tier 3: when it passes
against real OpenRouter, Tier 3 is shipped. Plan 07's checkpoint hands off
to a human-supervised invocation by the orchestrator; the executor merely
WRITES this file.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

# Hard cap: full corpus ingest is ~$1. 2 papers ≈ ~$0.05–0.15.
# Do NOT raise without re-projecting cost against the latest 129-RESEARCH.md
# numbers — first live run may surface $0.30+ per paper for graph-heavy text.
SUBSET_PAPERS = 2


@pytest.mark.live
def test_tier3_end_to_end_2papers(tier3_live_keys_ok, tmp_path):
    """End-to-end: 2-paper ingest + graphml exists + hybrid query + cost > 0."""
    from shared.config import get_settings
    from shared.cost_tracker import CostTracker
    from shared.loader import DatasetLoader
    from tier_3_graph.cost_adapter import CostAdapter
    from tier_3_graph.ingest import ingest_corpus
    from tier_3_graph.query import run_query
    from tier_3_graph.rag import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLM_MODEL,
        build_rag,
    )

    # 1. Forward OPENROUTER_API_KEY into os.environ so LightRAG's openai-compat
    #    closures can read it lazily (rag.py reads os.environ inside the
    #    closures, NOT pydantic-settings — see Plan 03 SUMMARY).
    settings = get_settings()
    assert settings.openrouter_api_key is not None, (
        "tier3_live_keys_ok should have skipped without an OpenRouter key"
    )
    os.environ["OPENROUTER_API_KEY"] = (
        settings.openrouter_api_key.get_secret_value()
    )

    # 2. Pick the SUBSET_PAPERS smallest papers by file size on disk so the
    #    test is cost-bounded regardless of which papers got curated. Skip
    #    the test if fewer than SUBSET_PAPERS PDFs are present (e.g., a
    #    fresh checkout without LFS pulled).
    loader = DatasetLoader()
    all_papers = loader.papers()
    dataset_root = Path(get_settings().dataset_root)
    candidates: list[tuple[dict, int]] = []
    for p in all_papers:
        pdf_path = dataset_root / "papers" / p["filename"]
        if pdf_path.exists():
            candidates.append((p, pdf_path.stat().st_size))
    if len(candidates) < SUBSET_PAPERS:
        pytest.skip(
            f"Need >={SUBSET_PAPERS} PDFs in dataset/papers/ for live test; "
            "run Phase 127 Plan 04 to populate"
        )
    candidates.sort(key=lambda x: x[1])
    subset = [p for p, _ in candidates[:SUBSET_PAPERS]]
    assert len(subset) == SUBSET_PAPERS

    # 3. Build a CostTracker + adapter pair scoped to "tier-3-test" so the
    #    persisted JSON is namespaced separately from production runs.
    tracker = CostTracker("tier-3-test")
    adapter = CostAdapter(
        tracker,
        llm_model=DEFAULT_LLM_MODEL,
        embed_model=DEFAULT_EMBED_MODEL,
    )

    # 4. Working dir LIVES IN tmp_path so the production graph at
    #    lightrag_storage/tier-3-graph/ is never touched by this test
    #    (Pitfall 8 + Pitfall 10).
    test_working_dir = tmp_path / "lightrag_test"

    # 5. Use a dummy console object that absorbs prints — the live test
    #    output otherwise floods pytest's captured stdout. ingest_corpus
    #    + render_query_result both expect a .print(...) method.
    class _SilentConsole:
        def print(self, *args, **kwargs) -> None:
            return None

    console = _SilentConsole()

    async def _run() -> None:
        rag = build_rag(
            working_dir=str(test_working_dir),
            llm_token_tracker=adapter,
            model=DEFAULT_LLM_MODEL,
        )
        await rag.initialize_storages()
        try:
            ingested = await ingest_corpus(
                rag,
                subset,
                console=console,
                tracker=tracker,
                model=DEFAULT_LLM_MODEL,
            )
            assert ingested == SUBSET_PAPERS, (
                f"Expected {SUBSET_PAPERS} papers ingested, got {ingested}"
            )

            # ROADMAP success criterion 2: graph persists to disk.
            graphml = test_working_dir / "graph_chunk_entity_relation.graphml"
            assert graphml.exists(), (
                "graph_chunk_entity_relation.graphml not created after ingest"
            )
            # Constructor creates an empty placeholder; successful entity
            # extraction grows it well past 1KB. This is the same heuristic
            # main.py uses for first-ingest detection (Plan 05).
            assert graphml.stat().st_size > 1024, (
                f"graphml too small ({graphml.stat().st_size}B) — entity "
                "extraction likely failed"
            )

            # ROADMAP success criterion 1 (CLI runs end-to-end): query path.
            #
            # Use a generic question, not the multi-hop DPR/RAG probe — those
            # specific papers are NOT guaranteed to be in the 2-smallest
            # subset, so a multi-hop assertion would be flaky. We just want
            # to confirm the query path produces a non-empty answer; Phase
            # 131 will benchmark answer quality.
            answer, in_tok, out_tok = await run_query(
                rag,
                "What is the main contribution of these documents?",
                mode="hybrid",
                model=DEFAULT_LLM_MODEL,
                tracker=tracker,
            )
            assert isinstance(answer, str) and len(answer) > 0, (
                "Empty answer from rag.aquery"
            )

            # ROADMAP success criterion 3: cost is recorded.
            #
            # CostAdapter routes BOTH LLM (entity extraction during ingest +
            # the per-query LLM call) AND embedding usage into the tracker
            # via lightrag-hku==1.4.15's token_tracker callbacks (probe-
            # validated in Plan 03). total_usd() must be > 0 because we
            # actually called the API.
            assert tracker.total_usd() > 0, (
                "CostTracker should record non-zero cost after live run"
            )
        finally:
            # Always finalize storages so background tasks don't leak past
            # the test (Pitfall 8 — async lifecycle).
            await rag.finalize_storages()

    asyncio.run(_run())
