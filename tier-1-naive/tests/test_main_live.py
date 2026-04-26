"""Tier 1: live end-to-end test (real OpenAI + Gemini APIs).

Marked ``@pytest.mark.live`` — skipped automatically by pytest unless run
with ``-m live`` and both API keys are present in the environment.

Cost: ~$0.001 per run (2 papers, ~20 chunks, 1 query). The test uses a
``tmp_path`` ChromaDB to avoid polluting the canonical
``chroma_db/tier-1-naive/`` directory used by Tier 5 (Phase 130).

This test is the **phase verification gate**: it exercises all three Phase
128 ROADMAP success criteria in a single run:

  1. ``python tier-1-naive/main.py`` ingests + answers a sample query
     (we call ``cmd_ingest`` and ``cmd_query`` directly, the same code
     paths invoked by the CLI entrypoint).
  2. ChromaDB index persists to disk and can be reused (we re-open the
     collection from ``tmp_chroma`` after the run and assert ``count() > 0``).
  3. Cost and latency are printed for the demo query (we record the rich
     console and assert ``"Cost:"`` and ``"Latency:"`` appear in the output).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shared.cost_tracker import CostTracker


@pytest.mark.live
def test_ingest_and_query_end_to_end_2papers(
    tier1_live_keys, tmp_path, monkeypatch, capsys
):
    """Ingest 2 papers, query, assert cost+answer+chunks+latency printed."""
    from rich.console import Console

    from shared.loader import DatasetLoader
    from tier_1_naive import main as tier1_main
    from tier_1_naive import store as tier1_store

    # 1. Subset DatasetLoader.papers() to the first 2 entries that actually
    #    have a PDF on disk. Done via monkeypatch so we don't touch the
    #    manifest file. Skip the test if fewer than 2 PDFs are present
    #    (e.g., a fresh checkout without LFS pulled).
    real_loader = DatasetLoader()
    all_papers = real_loader.papers()
    dataset_root = Path("dataset")
    subset: list[dict] = []
    for p in all_papers:
        pdf_path = dataset_root / "papers" / p["filename"]
        if pdf_path.exists():
            subset.append(p)
        if len(subset) >= 2:
            break
    if len(subset) < 2:
        pytest.skip(
            "Need >=2 PDFs in dataset/papers/ for live test; "
            "run Phase 127 Plan 04 to populate"
        )

    def _patched_papers(self):
        return subset

    monkeypatch.setattr(DatasetLoader, "papers", _patched_papers)

    # 2. Redirect ChromaDB persistence to tmp_path so we don't pollute
    #    the canonical chroma_db/tier-1-naive/ used by Tier 5 (Pitfall 8).
    tmp_chroma = tmp_path / "chroma"
    monkeypatch.setattr(tier1_store, "CHROMA_PATH", tmp_chroma)

    console = Console(record=True)
    tracker = CostTracker("tier-1-test")

    # 3. Run ingest. Should embed ~10-30 chunks across the 2 papers.
    rc_ingest = tier1_main.cmd_ingest(reset=False, tracker=tracker, console=console)
    assert rc_ingest == 0
    embed_cost = tracker.total_usd()
    assert embed_cost > 0, "Ingest must record an OpenAI embedding cost"

    # 4. Run a query. Use the canonical DEFAULT_QUERY so the test mirrors
    #    what `python tier-1-naive/main.py` does end-to-end.
    rc_query = tier1_main.cmd_query(
        query=tier1_main.DEFAULT_QUERY,
        top_k=tier1_main.DEFAULT_TOP_K,
        tracker=tracker,
        console=console,
    )
    assert rc_query == 0

    # 5. Assert the run produced (a) more cost (LLM call on top of embed),
    #    (b) console output mentioning "Cost:" and "Latency:" — both come
    #    from render_query_result + the explicit latency footer line.
    assert tracker.total_usd() > embed_cost, (
        "Query must add LLM cost on top of embed cost"
    )
    output = console.export_text()
    assert "Cost:" in output, "render_query_result must print Cost: line"
    assert "Latency:" in output, "main.cmd_query must print Latency: footer"
    # Confirm at least one chunk row was rendered (doc_id pattern paper_id#p<page>).
    assert "#p" in output, "Chunk table must render doc_id like <paper>#p<page>"

    # 6. Confirm CHROMA_PATH was honored — the on-disk index lives in tmp.
    assert tmp_chroma.exists(), "tmp_chroma directory must exist after ingest"
    # Confirm collection persisted (re-open and count > 0). This is the
    # ROADMAP success criterion #2: "ChromaDB index persists to disk and
    # can be reused by Tier 5".
    coll = tier1_store.open_collection(reset=False, path=tmp_chroma)
    assert coll.count() > 0, (
        "Collection must persist to disk and be re-readable from tmp_chroma"
    )
