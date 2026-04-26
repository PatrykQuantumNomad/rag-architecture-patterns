"""Tier 3: Graph RAG (LightRAG) CLI entrypoint.

Usage
-----
End-to-end demo (CONFIRMS first ingest costs ~$1; pass --yes to skip prompt):

    python tier-3-graph/main.py --yes

Run ingest only (idempotent — re-runs near-free via kv_store_doc_status):

    python tier-3-graph/main.py --ingest --yes

Query an existing graph:

    python tier-3-graph/main.py --query "Your question?" --mode hybrid

Wipe the graph and rebuild (~$1, ~10 min):

    python tier-3-graph/main.py --reset --ingest --yes

Cost & latency are printed via shared.display.

Architecture
------------
This CLI is intentionally async-first because LightRAG's public API is
async-only (``ainsert`` / ``aquery``). Bridging happens at exactly one
place — ``asyncio.run(amain(args, console))`` at the bottom of ``main()``
(Pitfall 8 — async/sync integration). All other ingest/query/cost work runs
inside ``amain``.

Cost-surprise mitigation (Pitfall 3 + Pitfall 10) is implemented via the
``--yes`` flag interacting with two confirmation gates:

1. Before ``--reset`` wipes ``lightrag_storage/tier-3-graph/`` (because the
   subsequent ingest will cost ~$1, ~10 min).
2. Before the first ingest (detected by absence of
   ``graph_chunk_entity_relation.graphml``) — prevents accidental cost burn
   on a fresh checkout.

Re-running ``--ingest`` after the graph exists is near-free (LightRAG's
``kv_store_doc_status.json`` deduplicates by ``ids=[paper_id]``), so the
gate fires only on TRULY first runs.

Cost tracking is a single ``CostTracker("tier-3")`` instance threaded
through ``CostAdapter(tracker, llm_model, embed_model)`` into
``build_rag(llm_token_tracker=adapter)``. Outcome A (probe-validated in
Plan 03): both LLM and embedding paths in lightrag-hku==1.4.15 accept
``token_tracker`` and call ``add_usage(...)`` after each response. The
adapter dispatches LLM payloads (have ``completion_tokens``) to
``record_llm`` and embed payloads (no ``completion_tokens``) to
``record_embedding``. End result: persisted cost JSON includes BOTH ingest
and query spend automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as `python tier-3-graph/main.py`
# (so `from shared...` and `from tier_3_graph...` both resolve). Mirrors the
# bootstrap pattern from tier-1-naive/main.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.display import render_query_result  # noqa: E402
from shared.loader import DatasetLoader  # noqa: E402

from tier_3_graph.cost_adapter import CostAdapter  # noqa: E402
from tier_3_graph.ingest import ingest_corpus  # noqa: E402
from tier_3_graph.query import run_query  # noqa: E402
from tier_3_graph.rag import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    WORKING_DIR,
    build_rag,
)

# ---------------------------------------------------------------------------
# Locked constants — see 129-RESEARCH.md Code Example B + Open Q4.
# ---------------------------------------------------------------------------

DEFAULT_QUERY: str = (
    "Comparing Lewis et al. 2020's RAG and Karpukhin et al. 2020's DPR, "
    "how does dense passage retrieval relate to RAG's non-parametric memory?"
)
"""The Tier 3 multi-hop probe query.

Designed to surface graph-traversal advantage: requires synthesizing two
papers via cross-document entity edges (DPR → RAG via shared
"non-parametric memory" entity). Tier 1's vector-only top-k often misses
one of the two papers; Tier 3's hybrid/mix mode traverses entities in the
graph and surfaces both. Phase 131 will quantify; Phase 129 just needs the
demo to produce a coherent answer.
"""

DEFAULT_MODE: str = "hybrid"
"""Default LightRAG query mode (Open Q4 resolution).

``hybrid`` works without a separately-configured reranker; ``mix`` may
require one (and will silently degrade if absent). Users can opt into
``mix`` via ``--mode mix`` once they have a reranker configured. See
``VALID_MODES`` for the complete set.
"""

VALID_MODES: list[str] = ["naive", "local", "global", "hybrid", "mix"]


# ---------------------------------------------------------------------------
# Confirmation helper — used by both --reset and first-ingest gates.
# ---------------------------------------------------------------------------


def _confirm_or_abort(prompt: str, yes: bool, console: Console) -> bool:
    """Confirm a cost-surprise action, or auto-confirm under ``--yes``.

    Parameters
    ----------
    prompt
        The cost-warning message to show the user. Should mention the
        estimated $ cost and elapsed time so the user can make an informed
        decision (Pitfall 3 — cost-surprise).
    yes
        If ``True``, skip the interactive prompt and return ``True``
        immediately. The prompt is still printed (in dim) so the user can
        see what was auto-confirmed in their scrollback.
    console
        Where to print the warning / confirmation echo.

    Returns
    -------
    bool
        ``True`` to proceed; ``False`` to abort.
    """
    if yes:
        console.print(f"[dim]{prompt}  [confirmed via --yes][/dim]")
        return True
    console.print(f"[yellow]{prompt}[/yellow]")
    try:
        answer = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        # Non-interactive shell with no --yes ⇒ safe default = abort.
        return False
    return answer == "y"


# ---------------------------------------------------------------------------
# Async main — all ingest/query/cost work happens here.
# ---------------------------------------------------------------------------


async def amain(args, console: Console) -> int:
    """Async entry point. Returns the process exit code."""
    settings = get_settings()
    if settings.openrouter_api_key is None:
        # Pitfall 10 — fail fast with a friendly message; do NOT instantiate
        # CostTracker / build_rag yet (they would otherwise try to read the
        # missing key on first call and produce a worse error).
        console.print(
            "[red]OPENROUTER_API_KEY required for Tier 3 (LightRAG "
            "entity-extraction + answers + embeddings). Copy .env.example to "
            ".env and set your key from https://openrouter.ai/keys[/red]"
        )
        return 2

    # Make the key visible to LightRAG's openai-compat closures, which read
    # ``os.environ["OPENROUTER_API_KEY"]`` lazily inside ``rag.py`` (Plan 03).
    # pydantic-settings populates ``settings.openrouter_api_key`` from the .env
    # file but does NOT mutate process env, so we forward the SecretStr value
    # explicitly here.
    import os

    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key.get_secret_value()

    # --reset cost-surprise gate. Wipe BEFORE building the LightRAG instance
    # because LightRAG's constructor will recreate the storage directory
    # (we don't want our wipe to race the constructor's mkdir).
    if args.reset and Path(WORKING_DIR).exists():
        if not _confirm_or_abort(
            f"--reset will wipe {WORKING_DIR}/ — next --ingest will re-extract "
            f"entities (~$1, ~10 min).",
            args.yes,
            console,
        ):
            console.print("[yellow]Aborted by user.[/yellow]")
            return 0
        shutil.rmtree(WORKING_DIR)

    # Build a SINGLE CostTracker + adapter pair for the whole run. Threaded
    # into both LightRAG closures via build_rag(llm_token_tracker=...).
    tracker = CostTracker("tier-3")
    adapter = CostAdapter(
        tracker,
        llm_model=args.model,
        embed_model=DEFAULT_EMBED_MODEL,
    )
    rag = build_rag(llm_token_tracker=adapter, model=args.model)
    await rag.initialize_storages()

    try:
        if args.ingest:
            # Pitfall 3 — first-ingest cost-surprise gate. Detect "first run"
            # via the graphml file (LightRAG creates it eagerly on init, so
            # we check for entities IN the graph rather than file existence
            # alone — but in practice file presence is a strong proxy because
            # the file is empty at init and grows during ingest; we use the
            # idempotency contract: if entities exist, the cost gate is moot).
            graphml_path = Path(WORKING_DIR, "graph_chunk_entity_relation.graphml")
            already_built = (
                graphml_path.exists() and graphml_path.stat().st_size > 1024
            )
            if not already_built and not _confirm_or_abort(
                f"First-time ingest extracts a knowledge graph from ~100 papers. "
                f"Estimated cost: ~$1.00 via {args.model} (one-time). Re-runs "
                f"are near-free.",
                args.yes,
                console,
            ):
                console.print("[yellow]Aborted by user.[/yellow]")
                return 0
            loader = DatasetLoader()
            papers = loader.papers()
            if not papers:
                console.print(
                    "[red]No papers in dataset/manifests/papers.json. "
                    "Run `python scripts/curate_corpus.py` (Phase 127 Plan 04) "
                    "first.[/red]"
                )
                return 2
            await ingest_corpus(
                rag,
                papers,
                console=console,
                tracker=tracker,
                model=args.model,
            )

        if args.query:
            t0 = time.monotonic()
            answer_text, in_tok, out_tok = await run_query(
                rag,
                args.query,
                mode=args.mode,
                model=args.model,
                tracker=tracker,
            )
            latency = time.monotonic() - t0
            # LightRAG does not expose per-chunk citations the same way Tier 1
            # does (the graph traversal blends multiple sources into the
            # answer). Passing chunks=[] is the honest representation; the
            # renderer prints "No chunks retrieved." in that case.
            render_query_result(
                query=args.query,
                chunks=[],
                answer=answer_text,
                cost_usd=tracker.total_usd(),
                input_tokens=in_tok,
                output_tokens=out_tok,
                console_override=console,
            )
            console.print(
                f"[bold]Latency:[/bold] {latency:.2f}s  "
                f"[dim]mode={args.mode}  model={args.model}[/dim]"
            )
            persisted = tracker.persist()
            console.print(f"[dim]Cost JSON written to {persisted}[/dim]")
        return 0
    finally:
        # Always close LightRAG storages cleanly so background tasks (event
        # loops over the vdb store) get cancelled before the process exits.
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Argparse + sync entry point — the ONLY async/sync bridge (Pitfall 8).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier-3-graph",
        description=(
            "Tier 3 (Graph RAG) — LightRAG knowledge-graph extraction + "
            "multi-mode retrieval over the rag-architecture-patterns corpus."
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Build the knowledge graph (one-time ~$1, idempotent on re-run).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a query against the persisted graph.",
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
        "--reset",
        action="store_true",
        help=f"Wipe {WORKING_DIR} before --ingest (warning: re-ingest costs ~$1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=(
            "OpenRouter LLM slug for entity extraction + answers "
            "(default %(default)s). Must be present in shared.pricing.PRICES."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip cost-surprise confirmation prompts (for non-interactive use).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — the single asyncio.run boundary (Pitfall 8)."""
    args = _build_parser().parse_args(argv)
    console = Console()

    # Default behavior (no flags except --model / --mode / --yes): auto-run
    # the canned ingest+query demo. The presence of --model / --mode / --yes
    # alone should NOT suppress the demo because they only configure HOW
    # ingest+query run, not whether to run them.
    no_flags = not (args.ingest or args.query or args.reset)
    if no_flags:
        args.ingest = True
        args.query = DEFAULT_QUERY

    return asyncio.run(amain(args, console))


__all__ = [
    "main",
    "amain",
    "DEFAULT_QUERY",
    "DEFAULT_MODE",
    "VALID_MODES",
]


if __name__ == "__main__":
    raise SystemExit(main())
