"""Tier 4 — PDF batch ingest via the MineRU pipeline (Pattern 2).

Iterates a papers manifest and calls
``rag.process_document_complete(file_path=..., doc_id=p["paper_id"])`` per
paper. The ``doc_id`` argument is non-optional in spirit: without it,
RAG-Anything generates a fresh content-hash-derived id every run, which
defeats its dedup mechanism and re-extracts the entire PDF on every ingest
(anti-pattern; burns MineRU runtime + LLM cost).

The MineRU model fetch (~3-5 GB) happens lazily on the first
``process_document_complete`` call. Plan 02 (this plan) does NOT exercise
that path — only Plan 05's live e2e test does. ``ingest_pdfs`` here is
authored as a pure async function so Plan 03's CLI can ``await`` it without
any module-import side effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rich.console import Console


async def ingest_pdfs(
    rag,
    papers: Iterable[dict],
    dataset_root: Path,
    console: Console,
    *,
    device: str = "cpu",
    parser: str = "mineru",
) -> int:
    """Ingest each ``papers[i]`` PDF via ``rag.process_document_complete``.

    Parameters
    ----------
    rag
        A ``RAGAnything`` instance from ``rag.py``'s ``build_rag()``.
    papers
        Iterable of paper dicts (each must carry ``filename`` + ``paper_id``;
        matches ``dataset/manifests/papers.json`` schema from Phase 127).
    dataset_root
        Absolute path to the ``dataset/`` directory; PDFs live under
        ``dataset/papers/<filename>``.
    console
        Rich ``Console`` for progress output.
    device
        MineRU compute device (``"cpu"`` default; ``"cuda"`` available with
        a CUDA-capable host but not exercised by this repo's CI).
    parser
        Parser backend (default ``"mineru"`` — the only supported value
        per RAG-Anything 1.2.10's contract).

    Returns
    -------
    int
        Count of PDFs actually ingested (file-existence-filtered).
    """
    papers = list(papers)
    n = sum(1 for p in papers if (dataset_root / "papers" / p["filename"]).exists())
    console.print(
        f"[cyan]Tier 4 PDF ingest: {n} papers via {parser} on {device}.[/cyan]"
    )
    ingested = 0
    for p in papers:
        pdf = dataset_root / "papers" / p["filename"]
        if not pdf.exists():
            continue
        # NOTE: ``parser`` is set on ``RAGAnythingConfig`` at build time
        # (rag.py); raganything==1.2.10's ``process_document_complete`` does
        # NOT accept ``parser=`` as a kwarg — it forwards **kwargs to
        # ``MineruParser._run_mineru_command`` which raises TypeError on
        # unexpected kwargs (Rule 1 bug discovered during Plan 130-05 live
        # test). ``device`` IS accepted (passed through to mineru CLI ``-d``).
        await rag.process_document_complete(
            file_path=str(pdf),
            output_dir=str(Path("tier-4-multimodal/output") / p["paper_id"]),
            parse_method="auto",
            device=device,
            doc_id=p["paper_id"],  # stable id — RAG-Anything dedups; without it, every run re-extracts (anti-pattern)
        )
        ingested += 1
    return ingested
