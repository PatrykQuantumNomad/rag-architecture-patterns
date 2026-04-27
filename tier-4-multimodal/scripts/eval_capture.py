"""Tier 4 — local-run helper that produces evaluation/results/queries/tier-4-{ts}.json.

Why this exists: Phase 130 SC-1 deferred Tier 4's live test to the user (sandbox
OMP shmem block on MineRU). The harness's Stage 1 cannot drive Tier 4 in-sandbox.
This helper drives Tier 4 LOCALLY (where MineRU works) and writes the canonical
QueryLog JSON the harness Stage 2 reads.

Run from repo root:
    cd rag-architecture-patterns
    python tier-4-multimodal/scripts/eval_capture.py            # all 30 questions, hybrid mode
    python tier-4-multimodal/scripts/eval_capture.py --limit 5  # smoke
    python tier-4-multimodal/scripts/eval_capture.py --mode local
    python tier-4-multimodal/scripts/eval_capture.py --yes      # skip cost-surprise gate

Cost ballpark: 30 questions × ~$0.0015/query (per-query cost from 130-RESEARCH
@ 2026-04 vintage) = ~$0.05 for query phase. Ingest is one-time and assumed
already done before this helper runs.

Output paths:
    evaluation/results/queries/tier-4-{ISO timestamp}.json   — QueryLog
    evaluation/results/costs/tier-4-eval-{ISO timestamp}.json — CostTracker D-13 schema
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from shared.config import get_settings
from shared.cost_tracker import CostTracker

from tier_4_multimodal.rag import build_rag, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
from tier_4_multimodal.cost_adapter import CostAdapter
from tier_4_multimodal.query import run_query

from evaluation.harness.records import EvalRecord, QueryLog, write_query_log


GOLDEN_QA = _REPO_ROOT / "evaluation" / "golden_qa.json"
RESULTS_QUERIES = _REPO_ROOT / "evaluation" / "results" / "queries"
RAG_STORAGE = _REPO_ROOT / "rag_anything_storage" / "tier-4-multimodal"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def _capture(args, console: Console) -> int:
    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY not set — Tier 4 capture cannot run.[/red]")
        return 2
    if not RAG_STORAGE.exists():
        console.print(f"[red]Tier 4 storage missing at {RAG_STORAGE}.[/red]")
        console.print("[red]Run `python tier-4-multimodal/main.py --ingest --yes` first.[/red]")
        return 2

    qa = json.loads(GOLDEN_QA.read_text())
    if args.limit is not None:
        qa = qa[: args.limit]

    if not args.yes:
        console.print(
            f"[yellow]About to drive Tier 4 over {len(qa)} questions × ~$0.0015/q "
            f"≈ ${0.0015 * len(qa):.4f} (query-only; ingest cost not included).[/yellow]"
        )
        console.print("[yellow]Pass --yes to skip this prompt next time.[/yellow]")
        try:
            ans = input("Continue? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in {"y", "yes"}:
            console.print("[red]Aborted.[/red]")
            return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tracker = CostTracker("tier-4-eval")
    adapter = CostAdapter(tracker, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL)
    rag = build_rag(llm_token_tracker=adapter)
    await rag.initialize_storages()

    records: list[EvalRecord] = []
    for i, q in enumerate(qa):
        question_id = q["id"]
        question = q["question"]
        console.print(f"[dim][{i + 1}/{len(qa)}] {question_id}: {question[:80]}...[/dim]")

        contexts: list[str] = []
        try:
            from lightrag import QueryParam
            ctx_str = await rag.aquery(
                question, param=QueryParam(mode=args.mode, only_need_context=True)
            )
            if ctx_str:
                contexts = [c.strip() for c in str(ctx_str).split("-----") if c.strip()]
        except Exception as exc:  # noqa: BLE001
            console.print(f"[dim]  context probe unavailable ({exc.__class__.__name__})[/dim]")

        t0 = time.monotonic()
        try:
            answer = await run_query(rag, question, mode=args.mode)
            latency = time.monotonic() - t0
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 — record + continue
            answer = ""
            latency = time.monotonic() - t0
            error = f"{exc.__class__.__name__}: {exc}"

        records.append(EvalRecord(
            question_id=question_id,
            question=question,
            answer=str(answer) if answer else "",
            retrieved_contexts=contexts,
            latency_s=latency,
            cost_usd_at_capture=tracker.total_usd(),
            error=error,
        ))

    log = QueryLog(
        tier="tier-4",
        timestamp=timestamp,
        git_sha=_git_sha(),
        model=DEFAULT_LLM_MODEL,
        records=records,
    )

    out = RESULTS_QUERIES / f"tier-4-{timestamp.replace(':', '_')}.json"
    write_query_log(out, log)

    tracker.persist()
    console.print(f"[green]Wrote {out}[/green]")
    console.print(f"[green]Total cost: ${tracker.total_usd():.6f}[/green]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tier 4 eval capture — drives the 30 golden Q&A locally.")
    p.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"])
    p.add_argument("--model", default=None, help="(reserved; LLM is baked into build_rag at construction)")
    p.add_argument("--limit", type=int, default=None, help="Run on first N questions (default: all 30).")
    p.add_argument("--yes", action="store_true", help="Skip cost-surprise prompt.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_capture(args, Console()))


if __name__ == "__main__":
    raise SystemExit(main())
