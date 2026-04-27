"""Tier 4 live e2e test — 3-paper + 5-image subset.

Filename matches Phase 129 Plan 07 convention (``test_tierN_e2e_live.py``)
for pytest rootdir basename uniqueness — no ``__init__.py`` per Phase 128
Plan 02 follow-on rule means basenames must be unique across tier dirs.

Marked ``@pytest.mark.live`` — skipped automatically by pytest unless run
with ``-m live`` AND ``OPENROUTER_API_KEY`` is present in the environment
(``tier4_live_keys_ok`` fixture handles the skip).

Cost-bounded e2e
================

A full-corpus ingest costs ~$1-2 (LLM entity extraction across 100 papers
+ 581-image vision pass). This test hard-caps the input at the **3 smallest
papers** + **5 first images** to keep cost in the ~$0.10-0.30 range and
total wall time in the ~5-30 min range — measurable but cheap enough to
repeat per phase verification.

Cold-start caveat
=================

The first invocation on a fresh machine will trigger MineRU's layout/OCR
model download (~3-5 GB; ~5-15 min). Subsequent runs reuse the cache at
``$MINERU_CACHE_DIR`` (defaults to ``~/.mineru/``) and start in seconds.

Exercises every Tier 4 ROADMAP success criterion in a single run:

  1. ``python tier-4-multimodal/main.py`` runs end-to-end (we call
     ``ingest_pdfs`` + ``ingest_standalone_images`` + ``run_query``
     directly — same code paths as the CLI entrypoint after argparse +
     cost-surprise gate, which we bypass programmatically since pytest
     non-interactivity already short-circuits the prompt via the EOFError
     abort path).
  2. Multimodal ingest persists to disk and can be inspected
     (assert ``graph_chunk_entity_relation.graphml`` exists with non-trivial
     size after ingest — file is created at constructor init as an empty
     NetworkX placeholder, but successful entity extraction grows it well
     past 1KB).
  3. Cost and latency are recorded
     (assert ``tracker.total_usd() > 0`` after the query — captured by the
     ``CostAdapter`` threaded through ``build_rag(llm_token_tracker=)``
     against both the LLM/vision closures AND the embedding closure on
     lightrag-hku==1.4.15, per Phase 129 Plan 03 probe).

Isolation
=========

* ``working_dir`` is set under ``tmp_path`` so the test NEVER touches the
  user's production graph at ``rag_anything_storage/tier-4-multimodal/``
  (Pitfall 8 — collection collisions; Pitfall 10 — destructive ops on user
  state).
* ``CostTracker("tier-4-test")`` writes its persisted JSON to
  ``evaluation/results/costs/tier-4-test-<ts>.json`` (separate from the
  production ``tier-4-<ts>.json`` files).
* The 3 papers are the smallest by file size on disk and the 5 images are
  the first 5 entries from ``dataset/manifests/figures.json`` — bounded
  cost is preserved regardless of paper-set rotation.
* Symlinks (or a copy on systems without symlink permissions) are used to
  avoid duplicating image bytes; the ``figures.json`` subset manifest is
  written to ``tmp_path / dataset_subset / manifests / figures.json`` and
  ``ingest_standalone_images`` reads from this synthetic dataset root.

This test is the **Phase 130 Tier 4 close-out gate**: when it passes
against real OpenRouter, Tier 4 is shipped + ROADMAP must-haves are
empirically verified.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import pytest


# Hard caps: full-corpus ingest is ~$1-2. 3 papers + 5 images ≈ ~$0.10-0.30.
# Do NOT raise without re-projecting cost against the latest 130-RESEARCH.md
# numbers — first live run may surface higher per-paper / per-image cost
# for graph-heavy text + uncaptioned figures.
SUBSET_PAPERS = 3
SUBSET_IMAGES = 5

QUESTION = (
    "Summarize the main contribution of the provided documents in 2-3 sentences. "
    "If figures are referenced, mention what the figure depicts."
)


@pytest.mark.live
def test_tier4_end_to_end_subset(tier4_live_keys_ok, tmp_path):
    """End-to-end Tier 4 — 3 PDFs + 5 images via hybrid mode, cost > 0, KG > 1KB."""
    from shared.config import get_settings
    from shared.cost_tracker import CostTracker
    from shared.loader import DatasetLoader
    from tier_4_multimodal.cost_adapter import CostAdapter
    from tier_4_multimodal.ingest_images import ingest_standalone_images
    from tier_4_multimodal.ingest_pdfs import ingest_pdfs
    from tier_4_multimodal.query import run_query
    from tier_4_multimodal.rag import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLM_MODEL,
        build_rag,
    )

    # 1. Forward OPENROUTER_API_KEY into os.environ so the LightRAG closures
    #    (composed inside RAG-Anything) see it on first call. pydantic-settings
    #    populates settings.openrouter_api_key but does NOT mutate os.environ.
    settings = get_settings()
    assert settings.openrouter_api_key is not None, (
        "tier4_live_keys_ok should have skipped without an OpenRouter key"
    )
    os.environ["OPENROUTER_API_KEY"] = (
        settings.openrouter_api_key.get_secret_value()
    )

    # 2. Pick the SUBSET_PAPERS smallest PDFs by file size on disk so the
    #    test stays cost-bounded regardless of which papers got curated.
    #    Skip the test entirely if fewer than SUBSET_PAPERS PDFs are present
    #    (e.g., a fresh checkout without LFS pulled).
    dataset_root = Path(settings.dataset_root)
    loader = DatasetLoader(dataset_root)
    all_papers = loader.papers()
    sized: list[tuple[dict, int]] = []
    for p in all_papers:
        pdf = dataset_root / "papers" / p["filename"]
        if pdf.exists():
            sized.append((p, pdf.stat().st_size))
    if len(sized) < SUBSET_PAPERS:
        pytest.skip(
            f"Need >={SUBSET_PAPERS} PDFs in dataset/papers/ for live test; "
            "run Phase 127 Plan 04 to populate"
        )
    sized.sort(key=lambda x: x[1])
    papers_subset = [p for p, _ in sized[:SUBSET_PAPERS]]
    assert len(papers_subset) == SUBSET_PAPERS

    # 3. Build a synthetic dataset_subset/ tree under tmp_path with only
    #    SUBSET_IMAGES symlinked images + a trimmed figures.json manifest.
    #    Pitfall 4 (absolute img_path) is enforced by ingest_images.py
    #    itself — we just hand it a clean dataset_root.
    figures_path = dataset_root / "manifests" / "figures.json"
    if figures_path.exists():
        all_figs = json.loads(figures_path.read_text())[:SUBSET_IMAGES]
    else:
        all_figs = []
    fake_dataset = tmp_path / "dataset_subset"
    (fake_dataset / "manifests").mkdir(parents=True)
    (fake_dataset / "images").mkdir(parents=True)
    real_images = dataset_root / "images"
    subset_figs: list[dict] = []
    for f in all_figs:
        src = real_images / f["filename"]
        if not src.exists():
            continue
        dst = fake_dataset / "images" / f["filename"]
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        subset_figs.append(f)
    (fake_dataset / "manifests" / "figures.json").write_text(
        json.dumps(subset_figs)
    )

    # 4. Working dir LIVES IN tmp_path so the production graph at
    #    rag_anything_storage/tier-4-multimodal/ is never touched
    #    (Pitfall 8 + Pitfall 10).
    working_dir = str(tmp_path / "rag_anything_storage" / "tier-4-multimodal-test")

    # 5. Build a CostTracker + adapter pair scoped to "tier-4-test" so the
    #    persisted JSON is namespaced separately from production runs.
    tracker = CostTracker("tier-4-test")
    adapter = CostAdapter(
        tracker,
        llm_model=DEFAULT_LLM_MODEL,
        embed_model=DEFAULT_EMBED_MODEL,
    )

    # 6. Build the RAG-Anything instance via the factory pattern (Plan 03
    #    Rule 1 refactor) — adapter threaded through both LLM/vision and
    #    embedding closures via build_rag(llm_token_tracker=adapter).
    rag = build_rag(
        working_dir=working_dir,
        llm_token_tracker=adapter,
        model=DEFAULT_LLM_MODEL,
    )

    # 7. Use a dummy console object that absorbs prints — the live test
    #    output otherwise floods pytest's captured stdout. ingest_pdfs +
    #    ingest_standalone_images both expect a .print(...) method on the
    #    Rich Console contract; the bare minimum is a no-op .print.
    class _SilentConsole:
        def print(self, *args, **kwargs) -> None:
            return None

    console = _SilentConsole()

    async def _run() -> None:
        try:
            # --- Ingest pass (PDFs + images) ---
            t0 = time.monotonic()
            n_pdfs = await ingest_pdfs(
                rag,
                papers_subset,
                dataset_root,
                console,
                device="cpu",  # MPS/CUDA varies per host; cpu is the safe default
                parser="mineru",
            )
            assert n_pdfs == SUBSET_PAPERS, (
                f"Expected {SUBSET_PAPERS} PDFs ingested, got {n_pdfs}"
            )
            n_imgs = await ingest_standalone_images(rag, fake_dataset)
            ingest_latency = time.monotonic() - t0
            print(f"\nIngest latency: {ingest_latency:.2f}s")
            print(f"PDFs ingested: {n_pdfs}; images ingested: {n_imgs}")

            # --- ROADMAP success criterion 2: KG persists to disk ---
            graphml = Path(working_dir) / "graph_chunk_entity_relation.graphml"
            assert graphml.exists(), (
                f"graph_chunk_entity_relation.graphml not created at {graphml}"
            )
            # Constructor creates an empty placeholder; successful entity
            # extraction grows it well past 1KB. Same heuristic as Plan 03's
            # main.py first-ingest detection.
            graphml_size = graphml.stat().st_size
            assert graphml_size > 1024, (
                f"graphml suspiciously small ({graphml_size}B) — entity "
                "extraction likely failed"
            )
            print(f"graphml size: {graphml_size} bytes")

            # --- ROADMAP success criterion 1: query path produces an answer ---
            t0 = time.monotonic()
            answer = await run_query(rag, QUESTION, mode="hybrid")
            query_latency = time.monotonic() - t0
            print(f"Query latency: {query_latency:.2f}s")
            assert isinstance(answer, str) and len(answer.strip()) > 0, (
                "Empty answer from rag.aquery"
            )
            truncated = answer[:300] + ("..." if len(answer) > 300 else "")
            print(f"Answer (truncated 300 chars): {truncated}")

            # --- ROADMAP success criterion 3: cost is recorded ---
            #
            # CostAdapter routes BOTH LLM (entity extraction during ingest +
            # vision pass on images + per-query LLM) AND embedding usage
            # into the tracker via lightrag-hku==1.4.15's token_tracker
            # callbacks (probe-validated in Phase 129 Plan 03 — Outcome A
            # extends to Tier 4 because RAG-Anything composes a LightRAG
            # instance internally). total_usd() must be > 0 because we
            # actually called the API.
            total_cost = tracker.total_usd()
            assert total_cost > 0, (
                "CostTracker should record non-zero cost after live run"
            )
            print(f"Total cost: ${total_cost:.6f}")

            # Persist the per-run JSON for forensics + expected_output capture.
            persisted_path = tracker.persist()
            print(f"Cost JSON: {persisted_path}")
        finally:
            # Always finalize storages so background tasks don't leak past
            # the test (Pitfall 8 — async lifecycle).
            try:
                await rag.finalize_storages()
            except Exception:
                # finalize_storages may not be required if no storage was
                # initialized (e.g., test failed before first ingest call).
                # Suppress so the original assertion error propagates cleanly.
                pass

    asyncio.run(_run())
