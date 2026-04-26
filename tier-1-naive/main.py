"""Tier 1: Naive RAG CLI entrypoint.

Usage
-----
Run an end-to-end demo (auto-ingests if needed, runs the canned query):

    python tier-1-naive/main.py

Run ingest only (idempotent — skips if collection already populated):

    python tier-1-naive/main.py --ingest

Run a specific query against an already-ingested collection:

    python tier-1-naive/main.py --query "Your question?" --top-k 5

Wipe the collection and re-ingest from scratch:

    python tier-1-naive/main.py --ingest --reset

Cost & latency are printed for every query via shared.display.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as `python tier-1-naive/main.py`
# (so `from shared...` and `from tier_1_naive...` both resolve).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.display import render_query_result  # noqa: E402
from shared.loader import DatasetLoader  # noqa: E402
from tier_1_naive.chat import (  # noqa: E402
    DEFAULT_CHAT_MODEL,
    build_chat_client,
    complete as chat_complete,
)
from tier_1_naive.embed_openai import (  # noqa: E402
    EMBED_MODEL,
    build_openai_client,
    embed_batch,
)
from tier_1_naive.ingest import chunk_page, extract_pages  # noqa: E402
from tier_1_naive.prompt import build_prompt  # noqa: E402
from tier_1_naive.retrieve import retrieve_top_k  # noqa: E402
from tier_1_naive.store import open_collection  # noqa: E402

# ---------------------------------------------------------------------------
# Locked constants — see 128-RESEARCH.md Examples A & data section.
# ---------------------------------------------------------------------------
EMBED_BATCH: int = 100  # Pitfall 6 — well under 2048-input cap
DEFAULT_TOP_K: int = 5
DEFAULT_QUERY: str = (
    "What is the core mechanism Lewis et al. 2020 introduce in the RAG paper "
    "for combining parametric and non-parametric memory?"
)  # = golden_qa.json::single-hop-001 — canonical Tier 1 success demo


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_ingest(reset: bool, tracker: CostTracker, console: Console) -> int:
    """Ingest dataset/papers/* into chroma_db/tier-1-naive/.

    Idempotent: if the collection is already populated and --reset is
    NOT passed, prints a message and returns without re-embedding
    (Pattern 4 / Pitfall 5).
    """
    coll = open_collection(reset=reset)
    if coll.count() > 0 and not reset:
        console.print(
            f"[yellow]Collection already populated ({coll.count()} chunks). "
            "Pass --reset to re-ingest.[/yellow]"
        )
        return 0

    loader = DatasetLoader()
    papers = loader.papers()
    if not papers:
        console.print(
            "[red]No papers in dataset/manifests/papers.json. "
            "Run `python scripts/curate_corpus.py` (Phase 127 Plan 04) first.[/red]"
        )
        return 2

    dataset_root = Path(get_settings().dataset_root)
    all_chunks: list[dict] = []
    console.print(f"[cyan]Extracting and chunking {len(papers)} papers...[/cyan]")
    for p in papers:
        pdf_path = dataset_root / "papers" / p["filename"]
        if not pdf_path.exists():
            console.print(f"[yellow]Skipping missing PDF: {pdf_path}[/yellow]")
            continue
        for page_no, text in extract_pages(str(pdf_path)):
            all_chunks.extend(chunk_page(text, paper_id=p["paper_id"], page=page_no))

    console.print(
        f"[cyan]Built {len(all_chunks)} chunks. "
        f"Estimated embed cost: ~${(len(all_chunks) * 512 / 1_000_000) * 0.02:.4f} "
        f"({EMBED_MODEL} @ $0.02/1M tokens via OpenRouter, conservative).[/cyan]"
    )

    oai = build_openai_client()
    n_batches = (len(all_chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i : i + EMBED_BATCH]
        vectors = embed_batch(oai, [c["document"] for c in batch], tracker)
        coll.add(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=vectors,
        )
        console.print(
            f"  embedded batch {i // EMBED_BATCH + 1}/{n_batches} "
            f"(running cost: ${tracker.total_usd():.4f})"
        )

    console.print(
        f"[green]Ingested {len(all_chunks)} chunks from {len(papers)} papers "
        f"into chroma_db/tier-1-naive/.[/green]"
    )
    return 0


def cmd_query(
    query: str,
    top_k: int,
    model: str,
    tracker: CostTracker,
    console: Console,
) -> int:
    """Run a single retrieval-augmented query and render the result.

    Cost (embed + LLM) and end-to-end latency are printed via
    shared.display.render_query_result + a latency footer line. Both the
    embedding and the chat completion go through OpenRouter (single key,
    selectable chat model).
    """
    coll = open_collection(reset=False)
    if coll.count() == 0:
        console.print(
            "[red]Empty index. Run `python tier-1-naive/main.py --ingest` first, "
            "or invoke without flags to auto-ingest before querying.[/red]"
        )
        return 2

    t0 = time.monotonic()

    embed_client = build_openai_client()
    [qv] = embed_batch(embed_client, [query], tracker)
    res = retrieve_top_k(coll, query_vec=qv, k=top_k)

    prompt = build_prompt(query, res["documents"], res["metadatas"])
    chat_client = build_chat_client()
    answer = chat_complete(chat_client, prompt, model=model, tracker=tracker)

    latency_s = time.monotonic() - t0

    chunks = [
        {
            "doc_id": f"{m['paper_id']}#p{m['page']}",
            "score": s,
            "snippet": doc[:200] + ("…" if len(doc) > 200 else ""),
        }
        for m, s, doc in zip(
            res["metadatas"], res["similarities"], res["documents"]
        )
    ]

    render_query_result(
        query=query,
        chunks=chunks,
        answer=answer.text,
        cost_usd=tracker.total_usd(),
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        console_override=console,
    )
    console.print(f"[bold]Latency:[/bold] {latency_s:.2f}s")

    persisted = tracker.persist()
    console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier-1-naive",
        description=(
            "Tier 1 (Naive RAG) — ChromaDB + OpenRouter embeddings + "
            "OpenRouter chat (configurable via --model)."
        ),
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Run ingest (PDFs -> chunks -> embeddings -> ChromaDB).",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a query against the persisted index.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the collection before --ingest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_CHAT_MODEL,
        help=(
            "OpenRouter chat model slug (default %(default)s). Examples: "
            "anthropic/claude-haiku-4.5, openai/gpt-4o-mini, "
            "google/gemini-2.5-pro. Browse https://openrouter.ai/models. "
            "The model must be present in shared.pricing.PRICES for cost "
            "tracking — extend that table when adopting a new model."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()

    # Friendly fast-fail for the OPENROUTER_API_KEY=None case (Pitfall 10).
    settings = get_settings()
    if settings.openrouter_api_key is None:
        console.print(
            "[red]OPENROUTER_API_KEY required for Tier 1 (chat + embeddings). "
            "Copy .env.example to .env and set your key from "
            "https://openrouter.ai/keys[/red]"
        )
        return 2

    tracker = CostTracker("tier-1")

    # Default behavior (no flags except --model): auto-ingest if empty, then
    # run DEFAULT_QUERY. The presence of --model alone should NOT suppress
    # the canned demo, so it is excluded from the no-flags check.
    no_flags = not (args.ingest or args.query or args.reset)
    if no_flags:
        args.ingest = True  # auto-ingest only if empty (cmd_ingest is idempotent)
        args.query = DEFAULT_QUERY

    if args.ingest:
        rc = cmd_ingest(reset=args.reset, tracker=tracker, console=console)
        if rc != 0:
            return rc

    if args.query:
        return cmd_query(
            query=args.query,
            top_k=args.top_k,
            model=args.model,
            tracker=tracker,
            console=console,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
