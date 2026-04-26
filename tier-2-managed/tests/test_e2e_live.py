"""Tier 2: live end-to-end test (real Gemini File Search API).

Marked ``@pytest.mark.live`` — skipped automatically by pytest unless run
with ``-m live`` and ``GEMINI_API_KEY`` is present in the environment.

Filename note (Rule 3 deviation from plan body): Plan 129-06 specified
``test_main_live.py`` to mirror Tier 1's filename, but pytest's rootdir
mode (no ``__init__.py`` in tier ``tests/`` dirs, per Phase 128 Plan 02
follow-on) requires unique basenames across the whole repo. With Tier 1
already owning ``test_main_live.py`` and Tier 3 (Plan 07, concurrent
wave) also adding one, this file is renamed to ``test_e2e_live.py`` so
collection succeeds — same precedent as the Phase 128 Plan 02 follow-on
that DELETED ``tier-1-naive/tests/__init__.py``: pytest collection
ergonomics override per-tier filename consistency.

Cost: ~$0.02-0.05 per run with a 3-paper subset (Pitfall 2 — keep the
test small to dodge documented 503 TPM saturation at full-corpus scale).
The test creates a UNIQUE timestamped store per run rather than reusing
the canonical ``rag-arch-patterns-tier-2`` store, so multiple test runs
don't pollute each other's state. The store is deleted in ``finally:``,
so a failed test does NOT leak storage in the user's GAI account.

This test is the Tier 2 verification gate: it exercises all three Phase
129 ROADMAP success criteria for Tier 2 in a single run:

  1. ``python tier-2-managed/main.py`` runs end-to-end (we drive
     ``upload_with_retry`` + ``tier2_query`` + ``to_display_chunks`` —
     the same code paths invoked by ``cmd_ingest`` + ``cmd_query`` in
     ``main.py``).
  2. The store handle persists (we create a unique store, upload, and
     confirm the documents land via ``list_existing_documents``).
  3. Cost + latency are tracked (we record LLM tokens via
     ``CostTracker.record_llm`` from ``response.usage_metadata`` and
     assert ``tracker.total_usd() > 0``).

Pitfall 6 reminder: ``grounding_metadata`` may be ``None`` if the model
concludes the corpus does not contain the answer. We log the chunk count
for visibility but do NOT hard-assert ``> 0`` — Pitfall 6 says that's a
flaky assertion. The cost-and-answer assertions are the load-bearing
checks; the chunk count is observational.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


SUBSET_PAPERS = 3  # Pitfall 2 — keep test runtime short and dodge TPM saturation.


@pytest.mark.live
def test_tier2_end_to_end(tier2_live_keys_ok, tmp_path):
    """End-to-end: create unique store, upload 3 papers, query, assert cost > 0, cleanup."""
    from google import genai
    from google.genai import types

    from shared.config import get_settings
    from shared.cost_tracker import CostTracker
    from shared.loader import DatasetLoader
    from tier_2_managed.query import (
        DEFAULT_MODEL,
        query as tier2_query,
        to_display_chunks,
    )
    from tier_2_managed.store import (
        delete_store,
        list_existing_documents,
        upload_with_retry,
    )

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    # Pitfall 1: file_search_stores arrived in google-genai 1.49. Fail fast
    # with a friendly hint if the SDK is too old (the test would otherwise
    # crash with a less-helpful AttributeError later).
    assert hasattr(client, "file_search_stores"), (
        "google-genai >= 1.49 required for File Search; "
        "run `uv pip install -e \".[tier-2]\"` to upgrade"
    )

    # Create a UNIQUE store for this test run. Don't pollute the user's
    # canonical ``rag-arch-patterns-tier-2`` store — that one is owned by
    # the CLI's idempotent ingest flow.
    unique_name = (
        f"rag-arch-tier-2-test-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    store = client.file_search_stores.create(
        config=types.CreateFileSearchStoreConfig(display_name=unique_name)
    )
    store_name = store.name

    try:
        # Pick the 3 smallest PDFs by file size — fastest to upload + index.
        # Skip the test if the dataset doesn't have enough papers (e.g., a
        # fresh checkout where Phase 127 Plan 04 hasn't run).
        loader = DatasetLoader()
        papers = loader.papers()
        dataset_root = Path(settings.dataset_root)
        candidates = [
            (p, (dataset_root / "papers" / p["filename"]).stat().st_size)
            for p in papers
            if (dataset_root / "papers" / p["filename"]).exists()
        ]
        candidates.sort(key=lambda x: x[1])
        subset = [p for p, _ in candidates[:SUBSET_PAPERS]]
        if len(subset) < SUBSET_PAPERS:
            pytest.skip(
                f"Need >={SUBSET_PAPERS} PDFs in dataset/papers/ for live test; "
                "run Phase 127 Plan 04 to populate"
            )

        # Sequential uploads with 30s/60s/120s backoff (Pitfall 2).
        for p in subset:
            pdf_path = dataset_root / "papers" / p["filename"]
            upload_with_retry(client, store_name, str(pdf_path), p["filename"])

        # Confirm uploads landed. ``list_existing_documents`` returns the set
        # of display_names already in the store; we expect exactly the 3 we
        # just uploaded.
        existing = list_existing_documents(client, store_name)
        assert len(existing) == SUBSET_PAPERS, (
            f"expected {SUBSET_PAPERS} docs, got {len(existing)}"
        )

        # Run a generic query that works regardless of which 3 papers got
        # picked. The canonical ``DEFAULT_QUERY`` (single-hop-001) only
        # works if Lewis 2020 happens to be in the smallest-3 subset; using
        # a generic prompt keeps the test corpus-agnostic.
        question = (
            "Summarize the main contribution of the provided documents "
            "in two sentences."
        )
        tracker = CostTracker("tier-2-test")
        t0 = time.monotonic()
        resp = tier2_query(client, store_name, question, model=DEFAULT_MODEL)
        latency = time.monotonic() - t0

        # Load-bearing assertions: non-empty answer, cost > 0.
        assert resp is not None
        assert getattr(resp, "text", None), "Empty answer text"

        # Pitfall 6: grounding_metadata may be None if the model thinks the
        # corpus is irrelevant. We log the count for visibility (so the
        # SUMMARY can record the empirical Open Q3 answer — does flash
        # surface a score field?) but do NOT hard-assert > 0.
        chunks = to_display_chunks(resp)
        scores = [c["score"] for c in chunks]
        any_nonzero_score = any(s > 0 for s in scores)
        print(
            f"\nTier 2 live e2e: chunks={len(chunks)}, "
            f"latency={latency:.2f}s, "
            f"any_nonzero_score={any_nonzero_score}"
        )

        # Record LLM cost from response.usage_metadata. Defensive against
        # paths where usage_metadata is omitted (cached responses, certain
        # tool-use paths) — see Plan 129-04 SUMMARY decision.
        usage = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        if in_tok or out_tok:
            tracker.record_llm(DEFAULT_MODEL, in_tok, out_tok)

        # The cost-tracking assertion is the source-of-truth Phase 129
        # ROADMAP success criterion. ``usage_metadata`` should always be
        # present on a real ``generate_content`` response; if it's not,
        # the test fails loudly here rather than silently passing with $0.
        total_usd = tracker.total_usd()
        print(f"Tier 2 live e2e: cost=${total_usd:.6f} ({in_tok}in/{out_tok}out)")
        assert total_usd > 0, (
            "CostTracker should record non-zero LLM cost from "
            "response.usage_metadata"
        )
    finally:
        # Cleanup is non-negotiable — even on exception, delete the test
        # store so we don't accumulate orphan stores in the user's GAI
        # account. ``delete_store`` swallows server-side errors silently
        # (already-deleted, network blip) and unconditionally clears the
        # ``.store_id`` sidecar; we don't want that local-side effect for
        # the test, but the test created its own unique store rather than
        # using the cached handle, so there is no sidecar to clear.
        delete_store(client, store_name)
