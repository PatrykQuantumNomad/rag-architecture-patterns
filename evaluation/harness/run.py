"""Phase 131 Stage 1 — capture per-tier query logs over the 30-question golden Q&A.

Run from repo root:
    cd rag-architecture-patterns
    python -m evaluation.harness.run --tiers 1,2,3,5 --yes              # full sweep
    python -m evaluation.harness.run --tiers 1 --limit 5 --yes          # smoke
    python -m evaluation.harness.run --tiers 4 \\
        --tier-4-from-cache evaluation/results/queries/tier-4-2026-04-28T12_00_00Z.json --yes

Outputs:
    evaluation/results/queries/{tier}-{timestamp}.json   — per-question records
    evaluation/results/costs/{tier}-{timestamp}.json     — D-13 schema (existing)

Architecture: ONE asyncio.run; per-tier serial loops; ONE CostTracker per tier
per invocation (Pitfall 11 collision avoidance). Tier 3 + Tier 5 reuse a single
LightRAG / Agent instance across the loop (storage init / agent build are heavy).
Tier 4 is dual-mode — cached path requires --tier-4-from-cache.

Decisions referenced (131-RESEARCH.md):
- Pattern 7: PRICES table already correct; no edits.
- Pitfall 5: single asyncio.run boundary; sync work via asyncio.to_thread inside adapters.
- Pitfall 11: one CostTracker per tier per invocation; distinct tier strings.
- Phase 130 SC-1 deferral: Tier 4 in run.py is cached-mode-only by default.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console

from shared.config import get_settings
from shared.cost_tracker import CostTracker

from evaluation.harness.records import EvalRecord, QueryLog, write_query_log

# Adapter dispatch (lazy-imported per tier requested to avoid pulling unused deps)
SUPPORTED_TIERS = {1, 2, 3, 4, 5}

# Cost ballpark per tier per question (USD), used for the cost-surprise gate.
# Sourced from Phase 128/129/130 live test SUMMARYs:
#   Tier 1 — ~$0.0001/q (Phase 128-06 live: $0.001 / 8 calls ≈ $0.000125; conservative)
#   Tier 2 — ~$0.000012/q (Phase 129-06 live: 20s, $0.000239 / 20 calls ≈ $0.000012)
#   Tier 3 — ~$0.01/q (Phase 129-07 live: $0.26 / ~25 calls ≈ $0.01; mostly graph traversal cost)
#   Tier 4 — ~$0.0015/q (Phase 130-03 cost surprise gate ballpark)
#   Tier 5 — ~$0.001/q (Phase 130-06 live: $0.000795 for one multi-tool query)
COST_PER_Q = {1: 0.0002, 2: 0.0001, 3: 0.01, 4: 0.0015, 5: 0.001}


def _git_sha() -> str:
    """Short HEAD SHA; 'unknown' if git unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _ts() -> str:
    """ISO 8601 UTC with second precision; D-13-compatible."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_for_filename(ts: str) -> str:
    """Replace ':' so macOS Finder is happy (D-13 convention)."""
    return ts.replace(":", "_")


def _load_golden_qa() -> list[dict]:
    """Load the 30-question golden Q&A from evaluation/golden_qa.json."""
    return json.loads((_REPO_ROOT / "evaluation" / "golden_qa.json").read_text())


def _check_prereqs(tiers: list[int], console: Console) -> int:
    """Return exit code (0 = ok, 2 = fail) per tier-specific prereqs.

    Mirrors the fast-fail discipline of tier-{1,3,5}-*/main.py (OPENROUTER_API_KEY)
    and tier-2-managed/main.py (GEMINI_API_KEY + .store_id).
    """
    settings = get_settings()
    fail = False

    needs_openrouter = bool({1, 3, 5} & set(tiers))
    needs_gemini = 2 in tiers

    if needs_openrouter and not settings.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY not set — tiers {1,3,5} cannot run.[/red]")
        fail = True
    if needs_gemini and not settings.gemini_api_key:
        console.print("[red]GEMINI_API_KEY not set — tier 2 cannot run.[/red]")
        fail = True
    if (1 in tiers or 5 in tiers) and not (_REPO_ROOT / "chroma_db" / "tier-1-naive").exists():
        console.print("[red]chroma_db/tier-1-naive/ missing.[/red]")
        console.print("[red]Run `python tier-1-naive/main.py --ingest` first.[/red]")
        fail = True
    if 2 in tiers and not (_REPO_ROOT / "tier-2-managed" / ".store_id").exists():
        console.print("[red]tier-2-managed/.store_id missing.[/red]")
        console.print("[red]Run `python tier-2-managed/main.py --ingest` first.[/red]")
        fail = True
    if 3 in tiers and not (_REPO_ROOT / "lightrag_storage" / "tier-3-graph").exists():
        console.print("[red]lightrag_storage/tier-3-graph/ missing.[/red]")
        console.print("[red]Run `python tier-3-graph/main.py --ingest --yes` first.[/red]")
        fail = True

    return 2 if fail else 0


def _cost_surprise(tiers: list[int], n_questions: int, console: Console) -> bool:
    """Cost-surprise gate; returns True to proceed, False to abort."""
    total = sum(COST_PER_Q[t] * n_questions for t in tiers if t in COST_PER_Q)
    console.print(
        f"[yellow]About to run {len(tiers)} tier(s) × {n_questions} question(s) "
        f"≈ ${total:.4f} (ballpark; actual logged in cost JSONs).[/yellow]"
    )
    console.print("[yellow]Pass --yes to skip this prompt next time.[/yellow]")
    try:
        ans = input("Continue? [y/N]: ").strip().lower()
    except EOFError:
        ans = "n"
    return ans in {"y", "yes"}


async def _capture_tier(
    tier: int,
    qa: list[dict],
    args,
    console: Console,
) -> Optional[QueryLog]:
    """Drive ONE tier over `qa`; return its QueryLog or None on prereq-skip.

    ONE CostTracker per tier per invocation (Pitfall 11 collision avoidance).
    Tier 3 + Tier 5 build their reusable instance (rag/agent) BEFORE the loop
    so storage init / agent construction is amortized across all questions.
    """
    timestamp = _ts()
    git_sha = _git_sha()
    tracker = CostTracker(f"tier-{tier}-eval")
    records: list[EvalRecord] = []

    if tier == 1:
        from evaluation.harness.adapters.tier_1 import (
            DEFAULT_MODEL as T1_MODEL,
            run_tier1,
        )
        for i, q in enumerate(qa):
            console.print(
                f"[dim][T1 {i+1}/{len(qa)}] {q['id']}: {q['question'][:80]}...[/dim]"
            )
            rec = await run_tier1(q["id"], q["question"], k=args.tier1_k, tracker=tracker)
            records.append(rec)
        model = T1_MODEL

    elif tier == 2:
        from evaluation.harness.adapters.tier_2 import run_tier2
        from tier_2_managed.query import DEFAULT_MODEL as T2_MODEL
        store_id_path = _REPO_ROOT / "tier-2-managed" / ".store_id"
        store_name = store_id_path.read_text().strip()
        for i, q in enumerate(qa):
            console.print(
                f"[dim][T2 {i+1}/{len(qa)}] {q['id']}: {q['question'][:80]}...[/dim]"
            )
            rec = await run_tier2(
                q["id"], q["question"], store_name=store_name, tracker=tracker
            )
            records.append(rec)
        model = T2_MODEL

    elif tier == 3:
        from evaluation.harness.adapters.tier_3 import run_tier3
        from tier_3_graph.cost_adapter import CostAdapter
        from tier_3_graph.rag import (
            DEFAULT_EMBED_MODEL,
            DEFAULT_LLM_MODEL as T3_MODEL,
            build_rag,
        )
        # ONE LightRAG instance reused across the loop (storage init is ~30s)
        adapter = CostAdapter(tracker, T3_MODEL, DEFAULT_EMBED_MODEL)
        rag = build_rag(llm_token_tracker=adapter)
        await rag.initialize_storages()
        for i, q in enumerate(qa):
            console.print(
                f"[dim][T3 {i+1}/{len(qa)}] {q['id']}: {q['question'][:80]}...[/dim]"
            )
            rec = await run_tier3(
                q["id"], q["question"], mode=args.mode, rag=rag, tracker=tracker
            )
            records.append(rec)
        model = T3_MODEL

    elif tier == 4:
        from evaluation.harness.adapters.tier_4 import CachedTier4Miss, run_tier4
        if not args.tier_4_from_cache:
            console.print(
                "[yellow]Tier 4 SKIPPED — no --tier-4-from-cache supplied.[/yellow]"
            )
            console.print(
                "[yellow]Run `python tier-4-multimodal/scripts/eval_capture.py` locally "
                "and pass the resulting JSON to --tier-4-from-cache.[/yellow]"
            )
            return None
        cache_path = Path(args.tier_4_from_cache)
        try:
            for i, q in enumerate(qa):
                console.print(f"[dim][T4 cached {i+1}/{len(qa)}] {q['id']}[/dim]")
                rec = await run_tier4(
                    q["id"], q["question"], from_cache=cache_path, tracker=tracker
                )
                records.append(rec)
        except CachedTier4Miss as exc:
            console.print(f"[red]Tier 4 cached log missing: {exc}[/red]")
            return None
        # Nominal — Tier 4 cached records carry their own model in the source log.
        model = "google/gemini-2.5-flash"

    elif tier == 5:
        from evaluation.harness.adapters.tier_5 import run_tier5
        from tier_5_agentic.agent import DEFAULT_MODEL as T5_MODEL, build_agent
        agent = build_agent(model=T5_MODEL)
        for i, q in enumerate(qa):
            console.print(
                f"[dim][T5 {i+1}/{len(qa)}] {q['id']}: {q['question'][:80]}...[/dim]"
            )
            rec = await run_tier5(
                q["id"], q["question"], agent=agent, tracker=tracker
            )
            records.append(rec)
        model = T5_MODEL

    else:
        console.print(f"[red]Unsupported tier: {tier}[/red]")
        return None

    log = QueryLog(
        tier=f"tier-{tier}",
        timestamp=timestamp,
        git_sha=git_sha,
        model=model,
        records=records,
    )

    out_dir = Path(args.output_dir) / "queries"
    out = out_dir / f"tier-{tier}-{_ts_for_filename(timestamp)}.json"
    write_query_log(out, log)

    tracker.persist()
    console.print(f"[green]Tier {tier} → {out}[/green]")
    console.print(f"[green]Tier {tier} cost: ${tracker.total_usd():.6f}[/green]")
    return log


async def amain(args, console: Console) -> int:
    """Async entry point — single asyncio.run boundary at top level (Pitfall 5)."""
    tiers = sorted(int(t) for t in args.tiers.split(",") if t.strip())
    invalid = [t for t in tiers if t not in SUPPORTED_TIERS]
    if invalid:
        console.print(
            f"[red]Unsupported tier(s): {invalid}. "
            f"Supported: {sorted(SUPPORTED_TIERS)}.[/red]"
        )
        return 2

    # Tier 4's "prereq" is the cache flag (validated inside _capture_tier);
    # the storage check would block users who run Tier 4 cached-only.
    code = _check_prereqs([t for t in tiers if t != 4], console)
    if code != 0:
        return code

    qa = _load_golden_qa()
    if args.limit is not None:
        qa = qa[: args.limit]

    if not args.yes:
        if not _cost_surprise(tiers, len(qa), console):
            console.print("[red]Aborted.[/red]")
            return 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "queries").mkdir(parents=True, exist_ok=True)

    for tier in tiers:
        log = await _capture_tier(tier, qa, args, console)
        if log is None:
            console.print(f"[yellow]Tier {tier} produced no log — moving on.[/yellow]")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser; exposed so tests can introspect flags."""
    p = argparse.ArgumentParser(
        description="Phase 131 Stage 1 — capture per-tier query logs."
    )
    p.add_argument(
        "--tiers",
        required=True,
        help="Comma-separated tier numbers, e.g. '1,2,3,5'.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N questions (default: all 30).",
    )
    p.add_argument(
        "--tier-4-from-cache",
        default=None,
        help="Path to a user-pre-captured Tier 4 QueryLog JSON (Phase 130 SC-1 deferral).",
    )
    p.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to write queries/ and (via tracker.persist) costs/ under.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip cost-surprise prompt.",
    )
    p.add_argument(
        "--mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="LightRAG mode for Tier 3 (default: hybrid).",
    )
    p.add_argument(
        "--tier1-k",
        type=int,
        default=5,
        help="Top-K chunks for Tier 1 retrieval (default: 5).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(amain(args, Console()))


if __name__ == "__main__":
    raise SystemExit(main())
