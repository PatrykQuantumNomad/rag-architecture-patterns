"""Tier 3 (LightRAG) ingest path — PDF→text→rag.ainsert.

This module owns its own PDF extractor (Pitfall 11): Tier 3 imports ``fitz``
(PyMuPDF) directly so it has zero dependency on Tier 1's ingest module. PyMuPDF
ships in the ``[tier-3]`` extras (Plan 129-01).

Why a Tier-3-owned extractor instead of reusing Tier 1's chunker:

* LightRAG performs its OWN chunking and entity extraction; we hand it the
  full per-paper concatenated text and let it tokenize internally.
* Tier 1's ``extract_pages`` returns ``(page_no, text)`` tuples shaped for
  ChromaDB chunk metadata — wrong shape for LightRAG.
* Cross-tier coupling is bad architecture (Pitfall 11): each tier should be
  reproducible from its own ``requirements.txt`` without pulling sibling
  modules. Tier 3 owning ``import fitz`` honors that.

Idempotency contract (Pattern 5):

* ``rag.ainsert(text, ids=[paper_id])`` passes a stable ``paper_id`` as the
  document id. LightRAG's ``kv_store_doc_status.json`` records ingested
  ids and skips re-insertion on subsequent runs — so re-running ``--ingest``
  after a successful one-time graph build is near-free (no re-extraction).
* This is the dual-protection model: cost-surprise is gated at the CLI
  level (``--yes`` flag in ``main.py``) AND at the LightRAG storage level
  (id-based dedup here). Both must hold.

Cost projection (Pitfall 3):

The first ingest extracts entities + relationships from every paper at ~$0.01
per paper via ``google/gemini-2.5-flash``, so a 100-paper corpus costs ~$1.
We print this estimate BEFORE the loop starts so users can abort by Ctrl-C
even after passing ``--yes`` to the higher-level CLI gate. The running
``tracker.total_usd()`` is also printed after each insert so users can
monitor real spend vs the projection.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf — Tier 3 owns this dep (Pitfall 11)


def extract_full_text(pdf_path: str) -> str:
    """Concatenate all page text from a PDF with double-newline separators.

    LightRAG performs its own chunking, so we hand it the whole document as
    one string. Empty pages (e.g., blank section dividers, figure-only pages
    that PyMuPDF cannot rasterize to text) are skipped to avoid feeding
    LightRAG meaningless whitespace that costs tokens during entity
    extraction.

    Parameters
    ----------
    pdf_path
        Filesystem path to a PDF file.

    Returns
    -------
    str
        All extractable page text joined by ``"\\n\\n"``. Empty string if
        the PDF contains no extractable text (e.g., scanned image-only PDFs).
    """
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                parts.append(t)
    return "\n\n".join(parts)


async def ingest_corpus(
    rag,
    papers: list[dict],
    console,
    tracker,
    model: str,
) -> int:
    """Ingest all papers from the corpus into the LightRAG knowledge graph.

    Idempotent: passes stable ``ids=[paper_id]`` so LightRAG's
    ``kv_store_doc_status.json`` deduplicates re-runs (Pattern 5). Prints a
    cost projection BEFORE the loop starts (Pitfall 3 — cost-surprise) so
    users can Ctrl-C abort even after the higher-level ``--yes`` gate passed.

    Parameters
    ----------
    rag
        A ``LightRAG`` instance from ``tier_3_graph.rag.build_rag(...)`` whose
        storages have been initialized via ``await rag.initialize_storages()``.
    papers
        The ``dataset/manifests/papers.json`` list — each dict carries
        ``paper_id`` (idempotency key) and ``filename`` (PDF basename).
    console
        A ``rich.console.Console`` for status output.
    tracker
        A ``shared.cost_tracker.CostTracker`` whose running ``total_usd()`` is
        printed after each insert. Cost recording happens via the
        ``CostAdapter`` already wired through ``build_rag``; this function
        only READS the running total for display.
    model
        The OpenRouter LLM slug used for entity extraction. Printed in the
        cost projection so users see which model the estimate is based on.

    Returns
    -------
    int
        Number of papers actually ingested (PDFs that existed on disk).
    """
    from shared.config import get_settings

    dataset_root = Path(get_settings().dataset_root)

    # Count existing PDFs up front so the projection reflects what we will
    # actually attempt (vs the manifest size, which may include papers whose
    # PDFs were not downloaded yet).
    n_papers = sum(
        1
        for p in papers
        if (dataset_root / "papers" / p["filename"]).exists()
    )
    console.print(
        f"[cyan]Tier 3 ingest plan: ~{n_papers} papers, ~$1.00 estimated "
        f"entity-extraction cost via {model} (one-time; idempotent on re-run "
        f"via kv_store_doc_status).[/cyan]"
    )

    ingested = 0
    for p in papers:
        pdf_path = dataset_root / "papers" / p["filename"]
        if not pdf_path.exists():
            continue
        text = extract_full_text(str(pdf_path))
        # Stable id ⇒ LightRAG dedupes re-runs (Pattern 5).
        await rag.ainsert(text, ids=[p["paper_id"]])
        ingested += 1
        console.print(
            f"  inserted {p['paper_id']} ({ingested}/{n_papers}) — "
            f"running cost ${tracker.total_usd():.4f}"
        )

    console.print(
        f"[green]Tier 3 ingest complete: {ingested} papers in graph at "
        f"lightrag_storage/tier-3-graph/.[/green]"
    )
    return ingested


__all__ = ["extract_full_text", "ingest_corpus"]
