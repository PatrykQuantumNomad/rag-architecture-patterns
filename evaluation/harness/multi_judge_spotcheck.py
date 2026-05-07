"""Phase 8 -- multi-judge spot-check (CAP-02).

Re-scores 5 questions x 3 tiers from the Phase 7 sweep with a non-Gemini
judge and writes a structured delta JSON for the Phase 9 frozen doc.

Output JSON shape (RESEARCH "Output JSON Schema"). Key invariants:
- ``source_capture_git_sha`` = ``QueryLog.git_sha`` (Phase 7: ``75f6f1b``);
  read by :func:`_read_source_sha` -- NEVER by :func:`_git_sha`
  (Pitfall 3 / BLOCKER #3 fix).
- ``spotcheck_run_git_sha`` = current HEAD via :func:`_git_sha`.
- ``secondary_judge.max_tokens`` = 8192 (Plan 02-04 lesson).
- ``cells[].delta`` = secondary - primary, ``None`` propagates (Pitfall 5).
- Cost ledger written to ``{results_dir}/costs/`` (Pitfall 2 / D-7).
- When ragas usage parser returns 0 tokens but n_scored > 0, fallback
  estimator sets ``{"estimated": true}`` (RESEARCH A6 / D-8).
CRITICAL (BLOCKER #3): :func:`_git_sha` is imported solely to populate
``spotcheck_run_git_sha`` -- NEVER for ``source_capture_git_sha``.
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console
from evaluation.harness.records import QueryLog, ScoreRecord, read_query_log
from evaluation.harness.run import _git_sha, _ts, _ts_for_filename, _load_golden_qa
from evaluation.harness.score import (
    JUDGE_EMB_SLUG_DEFAULT, _build_judge, _strip_openrouter_prefix, score_query_log,
)
from shared.cost_tracker import CostTracker
from shared.pricing import PRICES

# D-1, D-2, D-3 locks.
WANTED_IDS = ("single-hop-001", "single-hop-002",
              "multi-hop-001", "multi-hop-002", "multimodal-001")
DEFAULT_TIERS = (1, 4, 5)
SECONDARY_JUDGE_DEFAULT = "openrouter/anthropic/claude-haiku-4.5"
# D-8 fallback heuristic (RESEARCH Cost Budget projection).
_FALLBACK_INPUT_TOKENS_PER_CELL = 2400
_FALLBACK_OUTPUT_TOKENS_PER_CELL = 1150
PRIMARY_JUDGE_MODEL = "google/gemini-2.5-flash"
PRIMARY_EMBEDDER = "openai/text-embedding-3-small"
JUDGE_MAX_TOKENS = 8192
_METRICS = ("faithfulness", "answer_relevancy", "context_precision")

def _signed_delta(secondary: Optional[float], primary: Optional[float]) -> Optional[float]:
    """secondary - primary; None propagates (Pitfall 5 / Pattern 4)."""
    if secondary is None or primary is None:
        return None
    return secondary - primary

def _filter_records(log: QueryLog, ids: tuple[str, ...]) -> QueryLog:
    """Filter and reorder records to match `ids` order (Pitfall 7)."""
    by_id = {r.question_id: r for r in log.records}
    return log.model_copy(update={"records": [by_id[i] for i in ids if i in by_id]})

def _read_primary_metrics(metrics_dir: Path, tier: int, ts: str) -> dict[str, dict]:
    """Pin to exact ts -- NOT _latest() (Pitfall 3 / Pattern 3)."""
    path = metrics_dir / f"tier-{tier}-{_ts_for_filename(ts)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Primary metrics missing for tier-{tier}@{ts}: {path}")
    return {m["question_id"]: m for m in json.loads(path.read_text())}

def _read_source_sha(queries_dir: Path, tiers: list[int],
                     source_ts: Optional[str]) -> str:
    """Read ``git_sha`` from FIRST matching source QueryLog (BLOCKER #3 / D-6).

    Pulls Phase 7 capture SHA into ``source_capture_git_sha``; MUST NOT call
    :func:`_git_sha` (current HEAD may have advanced past the capture).
    """
    if source_ts is None:
        raise ValueError("_read_source_sha requires source_ts (Pitfall 3); pin to capture ts")
    ts_filename = _ts_for_filename(source_ts)
    for tier in tiers:
        path = queries_dir / f"tier-{tier}-{ts_filename}.json"
        if path.exists():
            return read_query_log(path).git_sha
    raise FileNotFoundError(
        f"No source QueryLog found in {queries_dir} for tiers={tiers} ts={source_ts}"
    )

def _estimate_cost_fallback(usage: dict, slug: str, n_scored: int) -> dict:
    """RESEARCH A6 / D-8: estimate cost when LiteLLM usage parser returns 0."""
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    price = PRICES.get(_strip_openrouter_prefix(slug), {})
    estimated = (in_tok == 0 and out_tok == 0)
    if estimated:
        in_tok = _FALLBACK_INPUT_TOKENS_PER_CELL * n_scored
        out_tok = _FALLBACK_OUTPUT_TOKENS_PER_CELL * n_scored
    usd = (in_tok / 1_000_000) * price.get("input", 0.0) \
        + (out_tok / 1_000_000) * price.get("output", 0.0)
    return {"estimated": estimated, "input_tokens": in_tok,
            "output_tokens": out_tok, "usd": usd}

def _summarise(subset: list[dict]) -> dict:
    out: dict = {"n_cells": len(subset), "n_skipped_due_to_nan": 0}
    sums: dict[str, list[float]] = {m: [] for m in _METRICS}
    for c in subset:
        vals = [c["delta"].get(m) for m in _METRICS]
        if any(v is None for v in vals): out["n_skipped_due_to_nan"] += 1
        for m, v in zip(_METRICS, vals):
            if v is not None: sums[m].append(v)
    for m in _METRICS:
        out[f"mean_delta_{m}"] = (sum(sums[m]) / len(sums[m])) if sums[m] else None
    return out

def _aggregate(cells: list[dict]) -> dict:
    """Per-tier + overall mean delta with NaN-skip counts."""
    keys = sorted({c["tier"] for c in cells})
    return {"by_tier": {k: _summarise([c for c in cells if c["tier"] == k]) for k in keys},
            "overall": _summarise(cells)}

def _build_cell(qid: str, tier: int, sec: ScoreRecord, pri: dict) -> dict:
    return {
        "question_id": qid, "tier": f"tier-{tier}",
        "primary": {m: pri.get(m) for m in _METRICS} | {"nan_reason": pri.get("nan_reason")},
        "secondary": {m: getattr(sec, m) for m in _METRICS} | {"nan_reason": sec.nan_reason},
        "delta": {m: _signed_delta(getattr(sec, m), pri.get(m)) for m in _METRICS},
    }

async def amain(args, console: Console) -> int:
    """Main async entrypoint -- orchestrates 3-tier spot-check re-score."""
    spotcheck_run_git_sha = _git_sha()  # NEVER for source_capture_git_sha
    spotcheck_run_ts = _ts()
    queries_dir = Path(args.results_dir) / "queries"
    metrics_dir = Path(args.results_dir) / "metrics"
    tiers = sorted(int(t) for t in str(args.tiers).split(",") if str(t).strip())
    qids = (tuple(q.strip() for q in args.question_ids.split(",") if q.strip())
            if getattr(args, "question_ids", "") else WANTED_IDS)
    try:
        source_capture_git_sha = _read_source_sha(queries_dir, tiers, args.source_ts)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Cannot determine source_capture_git_sha: {e}[/red]")
        return 2
    if source_capture_git_sha != spotcheck_run_git_sha:
        console.print(
            f"[yellow]Warning: spotcheck_run_git_sha={spotcheck_run_git_sha} differs "
            f"from source_capture_git_sha={source_capture_git_sha} (HEAD advanced "
            "post-capture; frozen doc cites source SHA per Pitfall 3)[/yellow]"
        )
    secondary_llm, secondary_emb = _build_judge(args.judge, args.judge_emb)
    qa_index = {q["id"]: q for q in _load_golden_qa()}
    cost_tracker = CostTracker("multi-judge-spotcheck")
    pricing_key = _strip_openrouter_prefix(args.judge)
    cells: list[dict] = []
    fallback_triggered = False
    agg_usage = {"input_tokens": 0, "output_tokens": 0, "n_scored": 0}
    for tier in tiers:
        src_path = queries_dir / f"tier-{tier}-{_ts_for_filename(args.source_ts)}.json"
        if not src_path.exists():
            console.print(f"[red]Tier {tier}: source capture missing: {src_path}[/red]")
            return 2
        src_log = read_query_log(src_path)
        filtered = _filter_records(src_log, qids)
        if len(filtered.records) != len(qids):
            console.print(f"[red]Tier {tier}: missing IDs ({len(filtered.records)}/{len(qids)} found); abort.[/red]")
            return 2
        secondary_scores, usage = await score_query_log(
            filtered, qa_index, secondary_llm, secondary_emb, batch_size=args.batch_size,
        )
        primary_by_qid = _read_primary_metrics(metrics_dir, tier, src_log.timestamp)
        for s in secondary_scores:
            cells.append(_build_cell(s.question_id, tier, s, primary_by_qid.get(s.question_id, {})))
        n_scored = int(usage.get("n_scored", 0) or 0)
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        agg_usage["input_tokens"] += in_tok; agg_usage["output_tokens"] += out_tok
        agg_usage["n_scored"] += n_scored
        if in_tok == 0 and n_scored > 0: fallback_triggered = True
        if pricing_key in PRICES:
            try: cost_tracker.record_llm(pricing_key, in_tok, out_tok)
            except KeyError: pass
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"multi-judge-spotcheck-{_ts_for_filename(spotcheck_run_ts)}.json"
    payload = {
        "$schema_version": "1.0",
        "spotcheck_run_timestamp": spotcheck_run_ts,
        "spotcheck_run_git_sha": spotcheck_run_git_sha,
        "source_capture_timestamp": args.source_ts,
        "source_capture_git_sha": source_capture_git_sha,
        "primary_judge": {"model": PRIMARY_JUDGE_MODEL, "embedder": PRIMARY_EMBEDDER},
        "secondary_judge": {
            "model": _strip_openrouter_prefix(args.judge), "model_slug": args.judge,
            "embedder": _strip_openrouter_prefix(args.judge_emb),
            "max_tokens": JUDGE_MAX_TOKENS,
        },
        "cells": cells, "aggregate": _aggregate(cells),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cost_path = cost_tracker.persist(dest_dir=Path(args.results_dir) / "costs")
    if fallback_triggered:
        cost_payload = json.loads(cost_path.read_text())
        est = _estimate_cost_fallback(agg_usage, args.judge, agg_usage["n_scored"])
        cost_payload["estimator"] = {
            "estimated": True, "method": "fixed_per_cell",
            "tokens_per_cell": {"input": _FALLBACK_INPUT_TOKENS_PER_CELL,
                                "output": _FALLBACK_OUTPUT_TOKENS_PER_CELL},
            "estimated_input_tokens": est["input_tokens"],
            "estimated_output_tokens": est["output_tokens"], "estimated_usd": est["usd"],
        }
        cost_path.write_text(json.dumps(cost_payload, indent=2), encoding="utf-8")
    console.print(f"[green]Spotcheck -> {out_path}[/green]")
    console.print(f"[green]Cost     -> {cost_path} (${cost_tracker.total_usd():.6f})[/green]")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 8 -- multi-judge spot-check (CAP-02).")
    p.add_argument("--source-ts", required=True, help="ISO 8601 UTC ts of source capture.")
    p.add_argument("--tiers", default="1,4,5", help="Comma-separated tier numbers.")
    p.add_argument("--judge", default=SECONDARY_JUDGE_DEFAULT, help="LiteLLM secondary judge slug.")
    p.add_argument("--judge-emb", default=JUDGE_EMB_SLUG_DEFAULT, help="LiteLLM embedder slug.")
    p.add_argument("--question-ids", default="", help="Comma-separated IDs (default: WANTED_IDS).")
    p.add_argument("--results-dir", default="evaluation/results", help="Parent of queries/metrics/costs.")
    p.add_argument("--batch-size", type=int, default=10, help="evaluate() batch_size.")
    p.add_argument("--yes", action="store_true", help="Suppress cost-surprise prompts.")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(amain(build_parser().parse_args(argv), Console()))

if __name__ == "__main__":
    sys.exit(main())
