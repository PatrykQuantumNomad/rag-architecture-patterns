"""Tier 2: Managed RAG (Gemini File Search) CLI entrypoint.

Usage
-----
Run an end-to-end demo (auto-creates store + uploads KB + runs canned query):

    python tier-2-managed/main.py

Run ingest only (idempotent — skips PDFs already in the store):

    python tier-2-managed/main.py --ingest

Run a query against an already-ingested store:

    python tier-2-managed/main.py --query "Your question?"

Wipe the store and re-ingest from scratch:

    python tier-2-managed/main.py --ingest --reset

Cost (synthetic indexing line + LLM line) and latency are printed via
``shared.display``. The synthetic indexing cost is an *estimate* (see
:func:`_count_pdf_tokens` and ``INDEXING_PRICE_PER_M`` below); the LLM
line is exact (taken from ``response.usage_metadata``).

Decisions referenced (see 129-RESEARCH.md):

- Code Example A: this main.py shape — argparse + cmd_ingest + cmd_query +
  no-flags default = ingest + DEFAULT_QUERY (mirrors Tier 1).
- Pattern 1: store-id caching delegated to ``store.get_or_create_store``.
- Pattern 2 / Pitfall 2: long-running operations + 30s/60s/120s backoff
  delegated to ``store.upload_with_retry``.
- Pattern 3: FileSearch tool wiring delegated to ``query.query``.
- Pitfall 1: ``hasattr(client, "file_search_stores")`` assertion at
  startup with a friendly upgrade hint (google-genai >= 1.49 required).
- Pitfall 2 (resilience): idempotent ingest — ``list_existing_documents``
  is consulted before each upload so 503-storm restarts skip uploads
  already in the store.
- Pitfall 6: ``to_display_chunks`` already defends None grounding_metadata.
- Pitfall 7: Gemini File Search has no per-token indexing line item, so
  we synthesize one — cl100k_base token estimate × $0.15/1M, recorded
  ONCE per ingest via ``tracker.record_embedding("gemini-embedding-001", N)``.
- Pitfall 10: GEMINI_API_KEY guard at startup — friendly red error +
  exit code 2 BEFORE CostTracker instantiation, mirroring Tier 1's
  OPENROUTER_API_KEY guard pattern.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo-root sys.path bootstrap so the script works when invoked as
# ``python tier-2-managed/main.py`` (mirrors Tier 1 lines 30-34).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tiktoken  # noqa: E402
from rich.console import Console  # noqa: E402
from google import genai  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.display import render_query_result  # noqa: E402
from shared.loader import DatasetLoader  # noqa: E402

from tier_2_managed.store import (  # noqa: E402
    STORE_DISPLAY_NAME,
    get_or_create_store,
    list_existing_documents,
    upload_with_retry,
)
from tier_2_managed.query import (  # noqa: E402
    DEFAULT_MODEL,
    query as tier2_query,
    to_display_chunks,
)

# ---------------------------------------------------------------------------
# Locked constants — see 129-RESEARCH.md.
# ---------------------------------------------------------------------------

DEFAULT_QUERY: str = (
    "What is the core mechanism Lewis et al. 2020 introduce in the RAG paper "
    "for combining parametric and non-parametric memory?"
)  # = golden_qa.json::single-hop-001 — same as Tier 1 for apples-to-apples comparison.

# USD per 1M input tokens for Gemini File Search managed indexing.
# Mirrors the gemini-embedding-001 rate in shared/pricing.py — Pitfall 7.
INDEXING_PRICE_PER_M: float = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_pdf_tokens(pdf_path: Path, enc) -> int:
    """Approximate Gemini's tokenizer using cl100k_base (Pitfall 7).

    Returns a conservative *estimated* token count for the synthetic
    indexing-cost line item. Two paths:

    1. If ``import fitz`` succeeds (because [tier-1] is also installed in
       the same env), extract page text via PyMuPDF and tokenize a
       capped sample with ``cl100k_base``. This is the more accurate path.
    2. If ``fitz`` is unavailable (Tier 2 is installed without [tier-1]
       extras), fall back to a file-size proxy: ``file_size_bytes // 4``.
       That keeps Tier 2 [tier-1]-independent — Pitfall 11 says each tier
       must own its own dependencies.

    The cap (``[:4096]``) on the encoded text avoids paying the cost of
    encoding a multi-MB document just to feed the synthetic estimate.
    """
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        return max(1, pdf_path.stat().st_size // 4)
    text_len = 0
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                text_len += len(page.get_text("text") or "")
    except Exception:
        return max(1, pdf_path.stat().st_size // 4)
    if text_len == 0:
        return max(1, pdf_path.stat().st_size // 4)
    # Cap the encoder workload — accuracy of ~4096-char tail is plenty for
    # a synthetic estimate; better than encoding the full doc on every run.
    return len(enc.encode(("a" * text_len)[:4096]))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_ingest(
    client: genai.Client,
    store,
    tracker: CostTracker,
    console: Console,
) -> int:
    """Idempotent ingest: enumerate manifest, skip already-uploaded display_names, upload rest sequentially.

    Pitfall 2: uploads are *sequential* (never parallel) and use
    ``upload_with_retry`` for 30s/60s/120s backoff. Pitfall 7: a single
    synthetic indexing-cost line is recorded at the end via
    ``tracker.record_embedding("gemini-embedding-001", N)``.
    """
    loader = DatasetLoader()
    papers = loader.papers()
    if not papers:
        console.print(
            "[red]No papers in dataset/manifests/papers.json. "
            "Run Phase 127 ingest first.[/red]"
        )
        return 2

    existing = list_existing_documents(client, store.name)
    console.print(
        f"[cyan]Store has {len(existing)} existing documents; "
        f"checking against manifest of {len(papers)}...[/cyan]"
    )

    dataset_root = Path(get_settings().dataset_root)
    enc = tiktoken.get_encoding("cl100k_base")
    total_indexed_tokens = 0
    uploaded = 0
    skipped = 0
    for p in papers:
        pdf_path = dataset_root / "papers" / p["filename"]
        display_name = p["filename"]
        if not pdf_path.exists():
            console.print(f"[yellow]Skipping missing PDF: {pdf_path}[/yellow]")
            continue
        if display_name in existing:
            skipped += 1
            continue
        console.print(
            f"  uploading {display_name} ({uploaded + 1}/{len(papers)})..."
        )
        upload_with_retry(client, store.name, str(pdf_path), display_name)
        total_indexed_tokens += _count_pdf_tokens(pdf_path, enc)
        uploaded += 1

    if total_indexed_tokens > 0:
        # Pitfall 7 — synthetic indexing cost line, recorded once per ingest.
        tracker.record_embedding("gemini-embedding-001", total_indexed_tokens)
        synth_cost = (total_indexed_tokens / 1_000_000) * INDEXING_PRICE_PER_M
        console.print(
            f"[green]Uploaded {uploaded} docs ({skipped} already present). "
            f"Synthetic indexing cost (estimate): ~${synth_cost:.4f} "
            f"for {total_indexed_tokens:,} tokens.[/green]"
        )
    else:
        console.print(
            f"[green]All {skipped} documents already present in the store; "
            "nothing to do.[/green]"
        )
    return 0


def cmd_query(
    client: genai.Client,
    store,
    query_text: str,
    model: str,
    tracker: CostTracker,
    console: Console,
) -> int:
    """Run a single managed-RAG query and render the result.

    The FileSearch tool is attached to ``generate_content``; the response
    carries ``grounding_metadata.grounding_chunks`` for the citations
    (Tier 2's managed-RAG win). Latency is timed end-to-end. Cost JSON is
    persisted at the end via ``tracker.persist()`` (D-13 schema).
    """
    t0 = time.monotonic()
    resp = tier2_query(client, store.name, query_text, model=model)
    latency = time.monotonic() - t0

    usage = getattr(resp, "usage_metadata", None)
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    if in_tok or out_tok:
        tracker.record_llm(model, in_tok, out_tok)

    chunks = to_display_chunks(resp)
    render_query_result(
        query=query_text,
        chunks=chunks,
        answer=getattr(resp, "text", None) or "",
        cost_usd=tracker.total_usd(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        console_override=console,
    )
    console.print(
        f"[bold]Latency:[/bold] {latency:.2f}s  [dim]model={model}[/dim]"
    )
    persisted = tracker.persist()
    console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier-2-managed",
        description=(
            "Tier 2 (Managed RAG) — Gemini File Search managed indexing "
            "+ retrieval. The store persists in Google's cloud; built-in "
            "citations land via grounding_metadata."
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Upload PDFs to the FileSearchStore (idempotent — skips already-uploaded).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a query against the persisted store.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the FileSearchStore + clear .store_id sidecar before --ingest.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            f"Gemini chat model for the answer step (default {DEFAULT_MODEL}). "
            "Must be present in shared.pricing.PRICES for cost tracking."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = Console()

    # Friendly fast-fail for the GEMINI_API_KEY=None case (Pitfall 10).
    # Pydantic-settings makes gemini_api_key REQUIRED, so settings construction
    # will normally raise before reaching the explicit None check below; the
    # explicit check is belt-and-braces in case the schema evolves to optional.
    settings = get_settings()
    if settings.gemini_api_key is None:  # pragma: no cover — pydantic raises first
        console.print(
            "[red]GEMINI_API_KEY required for Tier 2 (Gemini File Search). "
            "Copy .env.example to .env and set your key from "
            "https://aistudio.google.com/apikey[/red]"
        )
        return 2

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    # Pitfall 1: file_search_stores arrived in google-genai 1.49. If the
    # installed SDK is older, fail fast with a friendly upgrade hint.
    if not hasattr(client, "file_search_stores"):
        console.print(
            "[red]google-genai >= 1.49 required for File Search. "
            "Run `uv pip install -e \".[tier-2]\"` to upgrade.[/red]"
        )
        return 2

    tracker = CostTracker("tier-2")

    # Default behavior (no flags): auto-ingest (idempotent) + run DEFAULT_QUERY.
    no_flags = not (args.ingest or args.query or args.reset)
    if no_flags:
        args.ingest = True
        args.query = DEFAULT_QUERY

    store = get_or_create_store(client, reset=args.reset)
    console.print(
        f"[cyan]FileSearchStore: {store.name} "
        f"(display_name={STORE_DISPLAY_NAME})[/cyan]"
    )

    if args.ingest:
        rc = cmd_ingest(client, store, tracker, console)
        if rc != 0:
            return rc

    if args.query:
        return cmd_query(
            client, store, args.query, args.model, tracker, console
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
