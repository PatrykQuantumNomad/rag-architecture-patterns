"""Tier 4 — Multimodal RAG CLI (RAG-Anything).

Usage
-----
End-to-end demo (CONFIRMS first-ingest cost; pass --yes to skip prompt):

    python tier-4-multimodal/main.py --yes

Run ingest only (PDFs + standalone images):

    python tier-4-multimodal/main.py --ingest --yes

Query an existing multimodal store (no ingest):

    python tier-4-multimodal/main.py --query "What does Figure 1 depict?"

Skip standalone images (PDFs only — faster local dev iteration):

    python tier-4-multimodal/main.py --ingest --no-images --yes

Wipe the store and rebuild:

    python tier-4-multimodal/main.py --reset --ingest --yes

Architecture
------------
Async-first because RAG-Anything's public API is async-only (``ainsert`` /
``aquery``). The single ``asyncio.run(amain(args, console))`` boundary
lives in ``main()`` (Pitfall 8 — async/sync integration). Mirrors Phase 129
Plan 05's ``tier-3-graph/main.py`` structure with two Tier 4-specific
extensions:

* ``--no-images`` / ``--no-pdfs`` — control the hybrid ingest path. Tier 4
  pulls from BOTH ``dataset/papers/`` (PDFs via MineRU) AND
  ``dataset/images/`` (standalone figures via ``insert_content_list``).
* ``--device`` with autodetect (Open Q3) — MineRU's compute device. Picks
  ``cuda:0`` if torch sees a GPU, else ``mps`` on Apple Silicon, else
  ``cpu``. Users can override.

Cost-surprise mitigation (Pitfall 3) is implemented via the ``--yes``
flag interacting with two confirmation gates:

1. Before ``--reset`` wipes ``rag_anything_storage/tier-4-multimodal/``
   (because the subsequent ingest will run MineRU + entity extraction).
2. Before the first ingest (detected by absence of
   ``graph_chunk_entity_relation.graphml``) — prevents accidental cost
   burn on a fresh checkout.

Cost tracking is a single ``CostTracker("tier-4")`` threaded through
``CostAdapter(tracker, llm_model, embed_model)`` into
``build_rag(llm_token_tracker=adapter)``. Outcome A from Phase 129 Plan 03
(probe-validated) extends to Tier 4 verbatim because RAG-Anything composes
a LightRAG instance internally — both LLM and embedding paths in
lightrag-hku==1.4.15 accept ``token_tracker``.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as `python tier-4-multimodal/main.py`
# (so `from shared...` and `from tier_4_multimodal...` both resolve). Mirrors the
# bootstrap pattern from tier-1-naive / tier-3-graph main.py files.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.display import render_query_result  # noqa: E402
from shared.loader import DatasetLoader  # noqa: E402

from tier_4_multimodal.cost_adapter import CostAdapter  # noqa: E402
from tier_4_multimodal.ingest_images import ingest_standalone_images  # noqa: E402
from tier_4_multimodal.ingest_pdfs import ingest_pdfs  # noqa: E402
from tier_4_multimodal.query import run_query, to_display_chunks  # noqa: E402
from tier_4_multimodal.rag import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    WORKING_DIR,
    build_rag,
)


# ---------------------------------------------------------------------------
# Locked constants (do NOT promote to CLI flags without --reset semantics)
# ---------------------------------------------------------------------------

DEFAULT_QUERY: str = (
    "What does Figure 1 of the Attention Is All You Need paper depict, "
    "and how does the encoder-decoder architecture relate to "
    "retrieval-augmented generation?"
)
"""The Tier 4 default probe query.

Multimodal-flavored on purpose: asks about a figure (image) AND its
relationship to RAG narrative (text). Tier 4's hybrid KG should surface
both the figure caption AND the cross-paper conceptual link.
"""

DEFAULT_MODE: str = "hybrid"
VALID_MODES: list[str] = ["naive", "local", "global", "hybrid", "mix"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Autodetect MineRU compute device (Open Q3).

    Falls back to ``cpu`` on any exception (e.g. torch absent if user only
    installed [tier-5] extras and somehow imported main.py from this venv).
    Returns:
      * ``"cuda:0"`` if torch sees a CUDA-capable GPU
      * ``"mps"`` on Apple Silicon with MPS backend
      * ``"cpu"`` otherwise (sandbox-friendly default)
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _confirm_or_abort(prompt: str, yes: bool, console: Console) -> bool:
    """Cost-surprise gate. CI-safe: returns False on EOFError / blank input.

    Mirrors ``tier-3-graph/main.py::_confirm_or_abort`` (Phase 129 Plan 05).
    Non-interactive shells without ``--yes`` MUST ABORT, not silently proceed.
    """
    if yes:
        console.print(f"[dim]{prompt}  [confirmed via --yes][/dim]")
        return True
    console.print(f"[yellow]{prompt}[/yellow]")
    try:
        answer = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer == "y"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


async def cmd_ingest(rag, console: Console, args, tracker: CostTracker) -> int:
    """Run the hybrid Tier 4 ingest pass.

    PDFs flow through MineRU (Pattern 2 from 130-RESEARCH.md); standalone
    images flow through ``insert_content_list`` (Pattern 3 — Pitfall 4
    absolute paths). Both honored by the respective ``--no-pdfs`` /
    ``--no-images`` flags.
    """
    settings = get_settings()
    dataset_root = Path(settings.dataset_root)
    loader = DatasetLoader(dataset_root)
    papers = loader.papers()

    n_pdfs = 0
    if args.include_pdfs:
        if not papers:
            console.print(
                "[yellow]No papers in dataset/manifests/papers.json — "
                "skipping PDF ingest.[/yellow]"
            )
        else:
            n_pdfs = await ingest_pdfs(
                rag,
                papers,
                dataset_root,
                console,
                device=args.device,
                parser="mineru",
            )
    else:
        console.print("[dim]Skipping PDF ingest (--no-pdfs).[/dim]")

    n_images = 0
    if args.include_images:
        n_images = await ingest_standalone_images(rag, dataset_root)
        console.print(
            f"[cyan]Tier 4 standalone-image ingest: {n_images} figures.[/cyan]"
        )
    else:
        console.print("[dim]Skipping standalone-image ingest (--no-images).[/dim]")

    console.print(
        f"[green]Tier 4 ingest complete: {n_pdfs} PDFs + {n_images} images.[/green]"
    )
    return 0


async def cmd_query(rag, console: Console, args, tracker: CostTracker) -> int:
    """Run a single Tier 4 query and render the result panel.

    Threads ``console_override=console`` through ``render_query_result`` so
    the Cost / Latency / Cost-JSON-written lines all land on the same Rich
    console (Plan 128-06 retro-fix).
    """
    question = args.query or DEFAULT_QUERY

    # Snapshot tracker pre/post for per-query token counts (mirrors Tier 3
    # query.py — CostTracker accumulates ingest+query so we delta here).
    pre_in = sum(
        q["input_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )
    pre_out = sum(
        q["output_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )

    t0 = time.monotonic()
    answer = await run_query(rag, question, mode=args.mode)
    latency = time.monotonic() - t0

    post_in = sum(
        q["input_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )
    post_out = sum(
        q["output_tokens"] for q in tracker.queries if q.get("kind") == "llm"
    )

    # Pitfall 7 — RAG-Anything 1.2.10 returns a string, no chunks list.
    chunks = to_display_chunks(answer if not isinstance(answer, str) else None)

    render_query_result(
        query=question,
        chunks=chunks,
        answer=str(answer),
        cost_usd=tracker.total_usd(),
        input_tokens=max(0, post_in - pre_in),
        output_tokens=max(0, post_out - pre_out),
        console_override=console,
    )
    console.print(
        f"[bold]Latency:[/bold] {latency:.2f}s  "
        f"[dim]mode={args.mode}  model={args.model or DEFAULT_LLM_MODEL}[/dim]"
    )
    return 0


# ---------------------------------------------------------------------------
# Async main — all ingest/query/cost work happens here.
# ---------------------------------------------------------------------------


async def amain(args, console: Console) -> int:
    """Async entry point. Returns the process exit code."""
    settings = get_settings()
    # Pitfall 10 — fail fast on missing-OR-empty key; do NOT instantiate
    # CostTracker / build_rag yet (they would otherwise try to read the
    # missing key on first call and produce a worse error). Both paths
    # are real-world: env-unset (fresh checkout, no .env) and empty .env
    # value (user copied .env.example but never filled in the key).
    raw_key = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key is not None
        else ""
    )
    if not raw_key:
        console.print(
            "[red]OPENROUTER_API_KEY required for Tier 4 (RAG-Anything "
            "entity-extraction + answers + embeddings + vision). Copy "
            ".env.example to .env and set your key from "
            "https://openrouter.ai/keys[/red]"
        )
        console.print("[red]See tier-4-multimodal/README.md → Quickstart.[/red]")
        return 2

    # Forward the SecretStr value to process env so the LightRAG closures
    # in rag.py see it on first call. pydantic-settings populates
    # ``settings.openrouter_api_key`` from .env but does NOT mutate process
    # env automatically.
    import os

    os.environ["OPENROUTER_API_KEY"] = raw_key

    # Device autodetect / echo for transparency (Open Q3).
    detected_device = _detect_device()
    if args.device == "auto":
        args.device = detected_device
    console.print(
        f"[dim]MineRU device: {args.device} (autodetected: {detected_device})[/dim]"
    )

    # --reset cost-surprise gate. Wipe BEFORE building the RAG-Anything
    # instance because its constructor will recreate the storage directory.
    if args.reset and Path(WORKING_DIR).exists():
        if not _confirm_or_abort(
            f"--reset will wipe {WORKING_DIR}/ — next --ingest will re-extract "
            f"entities + re-run MineRU on every PDF (~$0.50-1.50 full corpus).",
            args.yes,
            console,
        ):
            console.print("[yellow]Aborted by user.[/yellow]")
            return 0
        shutil.rmtree(WORKING_DIR)

    # Build a SINGLE CostTracker + adapter pair for the whole run. Threaded
    # into both LightRAG closures via build_rag(llm_token_tracker=adapter).
    tracker = CostTracker("tier-4")
    adapter = CostAdapter(
        tracker,
        llm_model=args.model or DEFAULT_LLM_MODEL,
        embed_model=DEFAULT_EMBED_MODEL,
    )
    rag = build_rag(
        working_dir=WORKING_DIR,
        llm_token_tracker=adapter,
        model=args.model,
    )

    if args.ingest:
        # Pitfall 3 — first-ingest cost-surprise gate. Mirror Tier 3's
        # graphml-size > 1KB heuristic; the file is created at constructor
        # init time as an empty placeholder, so bare existence is not
        # enough to suppress the gate.
        graphml_path = Path(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        already_built = (
            graphml_path.exists() and graphml_path.stat().st_size > 1024
        )
        if not already_built and not _confirm_or_abort(
            "First-time Tier 4 ingest runs MineRU + multimodal entity "
            "extraction (~$0.50-1.50 full corpus, ~$0.05 for a 3-paper "
            "subset). Re-runs are near-free.",
            args.yes,
            console,
        ):
            console.print("[yellow]Aborted by user.[/yellow]")
            return 0
        rc = await cmd_ingest(rag, console, args, tracker)
        if rc != 0:
            return rc

    if args.query is not None or not args.ingest:
        # Run query if explicit --query was passed OR if we're in
        # default-flag-less demo mode (which sets args.ingest + args.query).
        rc = await cmd_query(rag, console, args, tracker)
        persisted = tracker.persist()
        console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
        return rc

    # Ingest-only path: persist what we have and exit clean.
    persisted = tracker.persist()
    console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
    return 0


# ---------------------------------------------------------------------------
# Argparse + sync entry point — the ONLY async/sync bridge (Pitfall 8).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier-4-multimodal",
        description=(
            "Tier 4 (Multimodal RAG) — RAG-Anything + MineRU + LightRAG over "
            "PDFs and standalone figures."
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingest pass (PDFs via MineRU + standalone images).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to answer. Defaults to a canned multimodal probe.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=VALID_MODES,
        help=(
            f"LightRAG query mode (default {DEFAULT_MODE}). "
            "naive=vector-only, local=entity-neighborhood, "
            "global=community-summary, hybrid=local+global, "
            "mix=hybrid+rerank (requires reranker)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            f"OpenRouter LLM slug for entity extraction + answers + vision "
            f"(default {DEFAULT_LLM_MODEL}). Must be present in "
            "shared.pricing.PRICES."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=f"Wipe {WORKING_DIR} before --ingest.",
    )
    parser.add_argument(
        "--include-images",
        dest="include_images",
        action="store_true",
        default=True,
        help="Include standalone images in ingest (default: on).",
    )
    parser.add_argument(
        "--no-images",
        dest="include_images",
        action="store_false",
        help="Skip standalone images during ingest.",
    )
    parser.add_argument(
        "--include-pdfs",
        dest="include_pdfs",
        action="store_true",
        default=True,
        help="Include PDFs in ingest (default: on).",
    )
    parser.add_argument(
        "--no-pdfs",
        dest="include_pdfs",
        action="store_false",
        help="Skip PDF ingest (e.g. images-only run).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="MineRU compute device: auto (default), cpu, cuda:0, or mps.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip cost-surprise confirmation prompts (for non-interactive use).",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Public alias — mirrors the plan body's import contract."""
    return _build_parser()


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — the single asyncio.run boundary (Pitfall 8)."""
    args = _build_parser().parse_args(argv)
    console = Console()

    # Default flag-less invocation: auto-run the canned ingest + query demo
    # (Phase 128 Plan 04 convention). Presence of --model / --mode / --device
    # / --yes alone should NOT suppress the demo because they only configure
    # HOW ingest+query run, not whether to run them.
    no_action = not (args.ingest or args.query or args.reset)
    if no_action:
        args.ingest = True

    return asyncio.run(amain(args, console))


__all__ = [
    "main",
    "amain",
    "build_parser",
    "DEFAULT_QUERY",
    "DEFAULT_MODE",
    "VALID_MODES",
]


if __name__ == "__main__":
    raise SystemExit(main())
