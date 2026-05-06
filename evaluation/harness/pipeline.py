"""Phase 5 — pipeline driver: capture -> score -> compare -> freeze in one command.

Run from repo root:
    cd rag-architecture-patterns
    python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes
    python -m evaluation.harness.pipeline --tiers 1 --limit 5 --yes
    python -m evaluation.harness.pipeline --tiers 4 \\
        --tier-4-from-cache evaluation/results/queries/tier-4-2026-04-28T12_00_00Z.json --yes

In-process composition (one Python process; no shell-out): one sweep-level git SHA + ISO timestamp
captured at the top via run._git_sha() / run._ts(), threaded into run.amain via
the new git_sha_override / ts_override kwargs (HARN-01). HARN-02 single-tier
rerun semantics work for free — score._latest_query_log + compare._latest are
already tier-isolated mtime-DESC. Compare stage always rolls up all 5 tiers.

Decisions referenced (05-01-PLAN.md frontmatter):
- D-Q1: targeted ~6-LOC kwarg plumb-through in run.py (no env-var coupling).
- D-Q2: --no-capture / --no-score / --no-compare deferred to v1.1.
- D-Q3: --force NOT exposed; user invokes freeze.py directly when overwriting.
- D-Q4: compare stage always operates on all 5 tiers (single-tier reruns
  produce a complete comparison.md).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from evaluation.harness import compare, freeze, run, score


def _pipeline_cost_estimate(tiers: list[int], n_q: int) -> tuple[float, float, float]:
    """Combined capture + judge cost ballpark (USD).

    Capture uses run.COST_PER_Q (per-tier). Judge factor is conservative
    ~$0.003/q × n_metrics(3) × n_internal_calls(~3) ≈ $0.003/q (per RESEARCH.md).
    """
    capture = sum(run.COST_PER_Q.get(t, 0.0) * n_q for t in tiers)
    judge = 0.003 * n_q * len(tiers)
    return capture, judge, capture + judge


def _build_run_args(args, sweep_sha: str, sweep_ts: str) -> Namespace:
    """Synthesize the Namespace run.amain expects (matches run.build_parser schema)
    plus git_sha_override + ts_override (HARN-01) and yes=True (suppress inner prompt)."""
    return Namespace(
        tiers=args.tiers,
        limit=args.limit,
        smoke_question_ids=args.smoke_question_ids,
        tier_4_from_cache=args.tier_4_from_cache,
        output_dir=str(args.results_dir),
        yes=True,
        mode=args.mode,
        tier1_k=args.tier1_k,
        git_sha_override=sweep_sha,
        ts_override=sweep_ts,
    )


def _build_score_args(args) -> Namespace:
    """Synthesize the Namespace score.amain expects (matches score.build_parser schema).
    yes=True suppresses score's inline cost prompt (Success Criterion 4)."""
    return Namespace(
        queries_dir=str(Path(args.results_dir) / "queries"),
        output_dir=str(args.results_dir),
        tiers=args.tiers,
        judge_model=args.judge_model,
        judge_emb=args.judge_emb,
        batch_size=args.batch_size,
        limit=args.limit,
        yes=True,
    )


def _build_compare_args(args) -> Namespace:
    """Synthesize the Namespace compare._run expects. D-Q4: always all 5 tiers
    so single-tier reruns produce a complete comparison.md."""
    return Namespace(
        results_dir=str(args.results_dir),
        out=str(Path(args.results_dir) / "comparison.md"),
        tiers="1,2,3,4,5",
    )


async def amain(args, console: Console) -> int:
    """Async entry — single asyncio.run boundary at top of main() (Pitfall 5)."""
    sweep_sha = run._git_sha()
    sweep_ts = run._ts()
    console.rule(f"Pipeline start git={sweep_sha} ts={sweep_ts}")

    tiers = sorted(int(t) for t in args.tiers.split(",") if t.strip())
    qa = run._load_golden_qa()
    n_q = min(args.limit, len(qa)) if args.limit else len(qa)

    if not args.yes:
        cap, jud, total = _pipeline_cost_estimate(tiers, n_q)
        console.print(
            f"[yellow]Capture ~${cap:.4f} + judge ~${jud:.4f} ≈ ${total:.4f} "
            f"(ballpark; actual logged in cost JSONs).[/yellow]"
        )
        try:
            ans = input("Continue? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in {"y", "yes"}:
            console.print("[red]Aborted.[/red]")
            return 1

    code = await run.amain(_build_run_args(args, sweep_sha, sweep_ts), console)
    if code != 0:
        console.print(f"[red]Pipeline halted at capture stage (exit {code}).[/red]")
        return code

    code = await score.amain(_build_score_args(args), console)
    if code != 0:
        console.print(f"[red]Pipeline halted at score stage (exit {code}).[/red]")
        return code

    code = await asyncio.to_thread(compare._run, _build_compare_args(args))
    if code != 0:
        console.print(f"[red]Pipeline halted at compare stage (exit {code}).[/red]")
        return code

    if args.freeze:
        try:
            path = await asyncio.to_thread(
                freeze.freeze,
                version=args.freeze,
                force=False,
                results_dir=Path(args.results_dir),
                source=None,
            )
        except (FileExistsError, FileNotFoundError, RuntimeError) as e:
            console.print(f"[red]Freeze refused: {e}[/red]")
            return 2
        console.print(f"[green]Frozen: {path}[/green]")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser; exposed so tests can introspect flags."""
    p = argparse.ArgumentParser(
        prog="python -m evaluation.harness.pipeline",
        description="Phase 5 pipeline driver — capture -> score -> compare -> freeze in one command.",
    )
    p.add_argument("--tiers", default="1,2,3,4,5",
                   help="Comma-separated tier numbers. Default: all 5.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only first N questions (default: all 30).")
    p.add_argument("--smoke-question-ids", default=None,
                   help="Comma-separated question ids to filter golden_qa.json down to a smoke subset.")
    p.add_argument("--tier-4-from-cache", default=None,
                   help="Path to a user-pre-captured Tier 4 QueryLog JSON.")
    p.add_argument("--results-dir", default="evaluation/results",
                   help="Parent of queries/, costs/, metrics/, frozen/. Default: evaluation/results.")
    p.add_argument("--judge-model", default=score.JUDGE_LLM_SLUG_DEFAULT,
                   help=f"LiteLLM judge slug. Default: {score.JUDGE_LLM_SLUG_DEFAULT}.")
    p.add_argument("--judge-emb", default=score.JUDGE_EMB_SLUG_DEFAULT,
                   help=f"LiteLLM embedder slug. Default: {score.JUDGE_EMB_SLUG_DEFAULT}.")
    p.add_argument("--batch-size", type=int, default=10,
                   help="evaluate() batch_size (Pitfall 3 — bounds concurrency). Default: 10.")
    p.add_argument("--mode", default="hybrid",
                   choices=["naive", "local", "global", "hybrid", "mix"],
                   help="LightRAG mode for Tier 3 (default: hybrid).")
    p.add_argument("--tier1-k", type=int, default=5,
                   help="Top-K chunks for Tier 1 retrieval (default: 5).")
    p.add_argument("--freeze", default=None,
                   help="Version slug (e.g. v1.0); when set, freezes comparison.md after compare.")
    p.add_argument("--yes", action="store_true",
                   help="Skip cost-surprise prompt; threaded into run+score args.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, Console()))


if __name__ == "__main__":
    raise SystemExit(main())
