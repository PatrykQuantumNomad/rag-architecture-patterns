"""Tier 5 — Agentic RAG CLI.

Single-shot agent loop: user asks a question, the OpenAI Agents SDK
:class:`~agents.Runner` drives ``ResearchAssistant`` (built in
:mod:`tier_5_agentic.agent`) up to ``--max-turns`` iterations, the agent
autonomously decides which of its two tools (``search_text_chunks``,
``lookup_paper_metadata``) to call, and the final answer + cost is rendered
via :func:`shared.display.render_query_result`.

Run::

    python tier-5-agentic/main.py                     # canned multi-tool query
    python tier-5-agentic/main.py --query "..."       # custom query
    python tier-5-agentic/main.py --max-turns 5       # tighter cap
    python tier-5-agentic/main.py --model openrouter/anthropic/claude-haiku-4.5

Pitfall 6 (130-RESEARCH.md): ``MaxTurnsExceeded.usage`` may be ``None`` on
some 0.x SDK patch releases — guarded with ``getattr(exc, "usage", None)``.
On truncation the CLI exits with code 3 (distinct from the fast-fail 2)
so callers can distinguish "missing key/index" from "agent ran out of turns".

Pitfall 12: :data:`shared.pricing.PRICES` keys are *provider-only* slugs
(e.g. ``google/gemini-2.5-flash``) without the ``openrouter/`` prefix.
:func:`_strip_openrouter_prefix` normalizes the model name before lookup so
the cost table works for any LiteLLM-routable OpenRouter model that has a
matching entry. Models without an entry log a yellow warning and report
USD=0 (cost tracking is best-effort, not load-bearing for correctness).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Repo-root sys.path bootstrap — same pattern as Tier 1's main.py. Required
# when this file is invoked directly (``python tier-5-agentic/main.py``)
# because then ``shared.*`` and ``tier_5_agentic.*`` would not otherwise
# resolve from the script's CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402

from agents import Runner  # noqa: E402
from agents.exceptions import MaxTurnsExceeded  # noqa: E402

from shared.config import get_settings  # noqa: E402
from shared.cost_tracker import CostTracker  # noqa: E402
from shared.display import render_query_result  # noqa: E402
from shared.pricing import PRICES  # noqa: E402

from tier_5_agentic.agent import build_agent, DEFAULT_MODEL  # noqa: E402


# ROADMAP-locked: TIER-05 success criterion 2 specifies max_iterations=10.
# Do NOT raise this for "nicer" outputs; the cap is the autonomy guarantee.
MAX_TURNS = 10

# Tier 1 owns this on-disk path. Tier 5 reads from it (Pitfall 9 — no
# writes ever). If the directory is absent the user has not run Tier 1's
# ingest yet — fast-fail with the explicit remediation command.
CHROMA_DIR = "chroma_db/tier-1-naive"

DEFAULT_QUERY = (
    "Compare the dense retrieval approach in DPR (Karpukhin 2020) with the "
    "retrieval-augmented generation approach in RAG (Lewis 2020). What is the "
    "key architectural difference? Cite paper_ids."
)


def _strip_openrouter_prefix(model: str) -> str:
    """Normalize an ``openrouter/<provider>/<model>`` slug to provider/model.

    Pitfall 12: :data:`shared.pricing.PRICES` is keyed by the provider-only
    slug. ``openrouter/google/gemini-2.5-flash`` → ``google/gemini-2.5-flash``.
    Models without an ``openrouter/`` prefix are returned unchanged.
    """
    return model.split("/", 1)[1] if model.startswith("openrouter/") else model


async def amain(args: argparse.Namespace) -> int:
    """Async entry point — runs the agent and renders the answer + cost.

    Returns
    -------
    int
        ``0`` on a successful run; ``2`` on a fast-fail (missing key or
        missing Tier 1 index); ``3`` on ``MaxTurnsExceeded`` so callers can
        distinguish truncation from outright failure.
    """
    settings = get_settings()
    console = Console(record=True)

    # Pitfall 10 inheritance from Phase 128: fast-fail BEFORE building the
    # agent so the user sees the friendly red error instead of a deep
    # LiteLLM authentication stack trace.
    if not settings.openrouter_api_key:
        console.print(
            "[red]OPENROUTER_API_KEY is not set — Tier 5 cannot run.[/red]"
        )
        console.print(
            "[red]Copy .env.example to .env and set your key from "
            "https://openrouter.ai/keys[/red]"
        )
        console.print(
            "[red]See tier-5-agentic/README.md → Quickstart for details.[/red]"
        )
        return 2

    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        console.print(
            f"[red]Tier 1 ChromaDB index not found at {chroma_path!r}.[/red]"
        )
        console.print(
            "[red]Tier 5 reuses Tier 1's index. Run Tier 1's ingest first:[/red]"
        )
        console.print("[red]  python tier-1-naive/main.py --ingest[/red]")
        return 2

    agent = build_agent(model=args.model)
    tracker = CostTracker("tier-5")

    truncated = False
    answer = ""
    usage = None
    t0 = time.monotonic()
    try:
        result = await Runner.run(
            agent,
            args.query or DEFAULT_QUERY,
            max_turns=args.max_turns,
        )
        answer = result.final_output or ""
        usage = result.context_wrapper.usage
    except MaxTurnsExceeded as exc:
        # Pitfall 6: usage may be None on some 0.x patch releases.
        truncated = True
        usage = getattr(exc, "usage", None)
        answer = (
            f"[truncated — agent exceeded max_turns={args.max_turns}] {exc}"
        )
    latency = time.monotonic() - t0

    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)

    pricing_key = _strip_openrouter_prefix(args.model or DEFAULT_MODEL)
    if pricing_key in PRICES:
        try:
            tracker.record_llm(pricing_key, in_tok, out_tok)
        except KeyError:  # Pitfall 12 defense — should not trigger after the membership check
            console.print(
                f"[yellow]Warning: model {pricing_key!r} not in PRICES; "
                "cost USD=0.[/yellow]"
            )
    else:
        console.print(
            f"[yellow]Warning: model {pricing_key!r} not in shared/pricing.py "
            "PRICES; cost USD=0.[/yellow]"
        )

    # Pitfall 7 / Plan 128-06 retro-fix: thread console_override so the
    # Cost: line lands on the SAME Rich console that the rest of the run
    # output uses. Tier 5 self-cites in answer text (no separate chunks
    # list to render).
    render_query_result(
        query=args.query or DEFAULT_QUERY,
        chunks=[],
        answer=answer,
        cost_usd=tracker.total_usd(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        console_override=console,
    )
    console.print(
        f"[dim]Latency: {latency:.2f}s, max_turns={args.max_turns}, "
        f"model={args.model or DEFAULT_MODEL}[/dim]"
    )
    if truncated:
        console.print(
            "[yellow]Agent hit max_turns cap — answer is partial.[/yellow]"
        )

    tracker.persist()
    console.print(
        f"[dim]Cost JSON: evaluation/results/costs/tier-5-{tracker.timestamp}.json[/dim]"
    )
    return 0 if not truncated else 3


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser."""
    p = argparse.ArgumentParser(
        description="Tier 5 — Agentic RAG via OpenAI Agents SDK + LiteLLM."
    )
    p.add_argument(
        "--query",
        default=None,
        help="Question for the agent. Defaults to a canned multi-tool probe.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"Hard cap on agent iterations (default: {MAX_TURNS}, ROADMAP-locked).",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"LiteLLM model slug (default: {DEFAULT_MODEL}). "
            "Must start with 'openrouter/'."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — wraps :func:`amain` in :func:`asyncio.run`."""
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
