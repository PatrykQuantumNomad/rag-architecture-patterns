"""Phase 131 Stage 3 — aggregate queries/costs/metrics into comparison.md.

Run from repo root:
    cd rag-architecture-patterns
    python -m evaluation.harness.compare                                          # all tiers, default paths
    python -m evaluation.harness.compare --tiers 1,2,3,5                         # skip Tier 4
    python -m evaluation.harness.compare --out evaluation/results/comparison.md  # explicit out path

Reads (most-recent by mtime per tier):
    evaluation/results/queries/tier-{N}-*.json    — Plan 04 outputs
    evaluation/results/costs/tier-{N}-eval-*.json — Plan 04 outputs (D-13)
    evaluation/results/metrics/tier-{N}-*.json    — Plan 05 outputs

Writes:
    evaluation/results/comparison.md  — TWO tables + footer (committed to git;
                                        Phase 133 BLOG-04 import target)

Architecture (per 131-RESEARCH "Two-stage execution model" + Pattern 5):
- Aggregation = pure file I/O + numpy.nanmean (no LLM calls, no async, no judge cost).
- Re-runnable: changing rounding or adding a column is a 1-line edit + re-run; doesn't
  invalidate the upstream JSONs.

Decisions referenced (131-RESEARCH.md):
- Pattern 5: 2-table layout (tier rollup + per-class rollup).
- Pattern 9: 'Cross-document verification' note in footer (multi-hop ≡ cross-document).
- Pitfall 2: numpy.nanmean ignores None values (per-record None when metric is NaN).
- Pitfall 8: n_truncated reported separately from n_NaN.
- Pitfall 9: 'Tiers 1-3 are text-only; multimodal scores reflect their limitation' footer.
- Open Q5 in research: comparison.md IS committed; queries/ + metrics/ are NOT.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SUPPORTED_TIERS = (1, 2, 3, 4, 5)


def _latest(dir_: Path, pattern: str) -> Optional[Path]:
    if not dir_.exists():
        return None
    files = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_golden_qa() -> list[dict]:
    return json.loads((_REPO_ROOT / "evaluation" / "golden_qa.json").read_text())


def _classify(question_id: str, qa_index: dict[str, dict]) -> str:
    """Return 'single-hop' | 'multi-hop' | 'multimodal' | 'unknown'."""
    q = qa_index.get(question_id)
    if not q:
        return "unknown"
    if q.get("modality_tag") == "multimodal":
        return "multimodal"
    return q.get("hop_count_tag", "unknown")


def aggregate_tier(tier: int, results_dir: Path) -> Optional[dict]:
    """Aggregate one tier's queries + costs + metrics into a row dict.

    Returns None if the queries log is missing — emit_markdown handles None by
    emitting an em-dashed placeholder row for the tier.
    """
    queries_path = _latest(results_dir / "queries", f"tier-{tier}-*.json")
    metrics_path = _latest(results_dir / "metrics", f"tier-{tier}-*.json")
    # Costs may have either tier-N-eval-*.json (Plan 04) or tier-N-*.json (Phase 128 / 130 main.py)
    cost_path = _latest(results_dir / "costs", f"tier-{tier}-eval-*.json") or \
                _latest(results_dir / "costs", f"tier-{tier}-*.json")

    if not queries_path:
        return None

    queries = json.loads(queries_path.read_text())
    records = queries.get("records", [])
    n = len(records)
    latencies = [r.get("latency_s", 0.0) for r in records if r.get("latency_s") is not None]

    # Costs (D-13 schema)
    if cost_path:
        costs = json.loads(cost_path.read_text())
        total_cost = float(costs.get("totals", {}).get("usd", 0.0))
    else:
        total_cost = 0.0

    # Metrics
    if metrics_path:
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = []

    # Build per-question dict for class rollup downstream
    metrics_by_qid = {m["question_id"]: m for m in metrics if "question_id" in m}

    # Aggregate metrics (filter Nones; np.nanmean is safe on the filtered list)
    f_scores = [m.get("faithfulness") for m in metrics if m.get("faithfulness") is not None]
    ar_scores = [m.get("answer_relevancy") for m in metrics if m.get("answer_relevancy") is not None]
    cp_scores = [m.get("context_precision") for m in metrics if m.get("context_precision") is not None]
    n_nan = sum(1 for m in metrics if m.get("nan_reason") is not None)
    nan_breakdown: dict[str, int] = {}
    for m in metrics:
        reason = m.get("nan_reason")
        if reason:
            nan_breakdown[reason] = nan_breakdown.get(reason, 0) + 1

    return {
        "tier": tier,
        "tier_label": f"tier-{tier}",
        "faithfulness": float(np.nanmean(f_scores)) if f_scores else float("nan"),
        "answer_relevancy": float(np.nanmean(ar_scores)) if ar_scores else float("nan"),
        "context_precision": float(np.nanmean(cp_scores)) if cp_scores else float("nan"),
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_query_usd": total_cost / max(1, n),
        "n": n,
        "n_nan": n_nan,
        "nan_breakdown": nan_breakdown,
        "records": records,
        "metrics_by_qid": metrics_by_qid,
        "timestamp": queries.get("timestamp"),
        "git_sha": queries.get("git_sha"),
        "model": queries.get("model"),
        "queries_path": str(queries_path),
        "cost_path": str(cost_path) if cost_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }


def aggregate_class_rollup(tier_data: list[dict], qa_index: dict[str, dict]) -> list[dict]:
    """For each (class, tier) pair, mean(faithfulness/answer_relevancy/context_precision).

    Iterates classes in [single-hop, multi-hop, multimodal] order; tiers in input order.
    Skips empty (class, tier) buckets entirely.
    """
    classes = ["single-hop", "multi-hop", "multimodal"]
    rows = []
    for cls in classes:
        for td in tier_data:
            if td is None:
                continue
            f, ar, cp = [], [], []
            n = 0
            n_nan = 0
            for rec in td["records"]:
                if _classify(rec["question_id"], qa_index) != cls:
                    continue
                n += 1
                m = td["metrics_by_qid"].get(rec["question_id"], {})
                if m.get("nan_reason") is not None:
                    n_nan += 1
                if m.get("faithfulness") is not None:
                    f.append(m["faithfulness"])
                if m.get("answer_relevancy") is not None:
                    ar.append(m["answer_relevancy"])
                if m.get("context_precision") is not None:
                    cp.append(m["context_precision"])
            if n == 0:
                continue
            rows.append({
                "class": cls,
                "tier": td["tier"],
                "tier_label": td["tier_label"],
                "faithfulness": float(np.nanmean(f)) if f else float("nan"),
                "answer_relevancy": float(np.nanmean(ar)) if ar else float("nan"),
                "context_precision": float(np.nanmean(cp)) if cp else float("nan"),
                "n": n,
                "n_nan": n_nan,
            })
    return rows


def _fmt_float(v: float, places: int) -> str:
    if v != v:  # NaN
        return "—"
    return f"{v:.{places}f}"


def _missing_row(tier: int) -> str:
    return f"| tier-{tier} | — | — | — | — | — | — | 0 | 0 |"


def emit_markdown(
    tier_rows: list[Optional[dict]],
    class_rows: list[dict],
    judge_model: str,
    judge_emb: str,
    capture_provenance: list[dict],
    tier_4_present: bool,
) -> str:
    """Emit the comparison.md content as a single string.

    Layout (per 131-RESEARCH Pattern 5):
      1. Title + generator note
      2. Tier rollup table (9 columns, sorted tier 1→5)
      3. Per-question-class rollup (7 columns, single-hop → multi-hop → multimodal)
      4. Provenance footer (judge model/emb, capture timestamps, NaN breakdown,
         Tier 4 deferral note when missing, honest disclaimers)
    """
    lines = [
        "# RAG Tier Comparison — Phase 131",
        "",
        "Generated by `python -m evaluation.harness.compare`. Source artifact for Phase 133 BLOG-04.",
        "",
        "## Tier Rollup",
        "",
        "| Tier | Faithfulness | Answer Relevancy | Context Precision | Mean Latency (s) | Total Cost (USD) | Cost / Query (USD) | n | n NaN |",
        "|------|--------------|------------------|-------------------|------------------|------------------|--------------------|---|-------|",
    ]
    for row, tier_int in zip(tier_rows, SUPPORTED_TIERS):
        if row is None:
            lines.append(_missing_row(tier_int))
            continue
        lines.append(
            f"| {row['tier_label']} | {_fmt_float(row['faithfulness'], 3)} "
            f"| {_fmt_float(row['answer_relevancy'], 3)} "
            f"| {_fmt_float(row['context_precision'], 3)} "
            f"| {_fmt_float(row['mean_latency_s'], 2)} "
            f"| {_fmt_float(row['total_cost_usd'], 6)} "
            f"| {_fmt_float(row['cost_per_query_usd'], 6)} "
            f"| {row['n']} | {row['n_nan']} |"
        )

    lines.extend([
        "",
        "## Per-Question-Class Rollup",
        "",
        "| Class | Tier | Faithfulness | Answer Relevancy | Context Precision | n | n NaN |",
        "|-------|------|--------------|------------------|-------------------|---|-------|",
    ])
    for r in class_rows:
        lines.append(
            f"| {r['class']} | {r['tier_label']} | {_fmt_float(r['faithfulness'], 3)} "
            f"| {_fmt_float(r['answer_relevancy'], 3)} "
            f"| {_fmt_float(r['context_precision'], 3)} "
            f"| {r['n']} | {r['n_nan']} |"
        )

    # Footer
    lines.extend([
        "",
        "## Provenance & Honest Disclosures",
        "",
        f"**Judge LLM:** {judge_model}",
        f"**Judge embedder:** {judge_emb}",
        "",
        "**Capture provenance per tier:**",
        "",
    ])
    if capture_provenance:
        for prov in capture_provenance:
            lines.append(
                f"- `{prov['tier_label']}`: captured {prov['timestamp']} "
                f"(model `{prov['model']}`, git `{prov['git_sha']}`)"
            )
    else:
        lines.append("- (no tier query logs found)")

    # NaN breakdown
    lines.extend(["", "**NaN breakdown per tier:**", ""])
    nan_any = False
    for row in tier_rows:
        if row is None:
            continue
        if row["nan_breakdown"]:
            nan_any = True
            parts = ", ".join(f"{n} {reason}" for reason, n in sorted(row["nan_breakdown"].items()))
            lines.append(f"- `{row['tier_label']}`: {parts}")
    if not nan_any:
        lines.append("- No NaN scores across the run (all questions × tiers scored cleanly).")

    # Tier 4 deferral footer
    if not tier_4_present:
        lines.extend([
            "",
            "**Tier 4: deferred to user.** Phase 130 SC-1 deferred Tier 4's live test "
            "to the user (sandbox kernel-level OMP shmem block on MineRU). Run "
            "`python tier-4-multimodal/scripts/eval_capture.py` locally to produce a "
            "Tier 4 query log, then re-run `python -m evaluation.harness.score --tiers 4` "
            "and `python -m evaluation.harness.compare` to fill the Tier 4 row.",
        ])

    # Honest disclaimers
    lines.extend([
        "",
        "**Honest disclaimers:**",
        "",
        "- 30 questions × ≤5 tiers is too small for statistical-significance testing. "
        "Numbers above are raw means; the magnitude (not p-values) is what the blog post discusses.",
        "- Multi-hop ≡ cross-document for this corpus (verified in Phase 127-06: every "
        "multi-hop entry in `golden_qa.json` cites ≥2 source papers).",
        "- Tiers 1-3 are text-only — their multimodal scores reflect that limitation; "
        "Tier 4 is the multimodal-RAG win (Phase 130 SC-1 deferred to user).",
        "- The judge LLM was the same Gemini Flash family Tiers 2-4 use for generation — "
        "self-grading bias is acknowledged. See 131-RESEARCH Open Q4.",
        "",
    ])
    return "\n".join(lines) + "\n"


def _detect_judge_provenance(results_dir: Path) -> tuple[str, str]:
    """Read the latest results/costs/ragas-judge-*.json to recover the canonical judge model.

    Falls back to the 131-RESEARCH defaults if no judge cost JSON is found.
    """
    judge_path = _latest(results_dir / "costs", "ragas-judge-*-*.json")
    if judge_path:
        try:
            data = json.loads(judge_path.read_text())
            # D-13 schema: queries[*].model holds the slug; pick the first llm-kind entry
            for q in data.get("queries", []):
                if q.get("kind") == "llm" and q.get("model"):
                    return q["model"], "openai/text-embedding-3-small"
        except (json.JSONDecodeError, OSError):
            pass
    return "google/gemini-2.5-flash", "openai/text-embedding-3-small"


def _run(args) -> int:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"results_dir not found: {results_dir}", file=sys.stderr)
        return 2

    try:
        tiers = sorted(int(t) for t in args.tiers.split(",") if t.strip())
    except ValueError:
        print(f"Invalid --tiers: {args.tiers!r} (expected comma-separated ints)", file=sys.stderr)
        return 2
    invalid = [t for t in tiers if t not in SUPPORTED_TIERS]
    if invalid:
        print(f"Unsupported tier(s): {invalid}. Supported: {list(SUPPORTED_TIERS)}.", file=sys.stderr)
        return 2

    qa = _load_golden_qa()
    qa_index = {q["id"]: q for q in qa}

    tier_rows = [aggregate_tier(t, results_dir) for t in tiers]
    # Pad to all SUPPORTED_TIERS so emit_markdown's 5-row layout stays consistent
    rows_by_tier = {t: tr for t, tr in zip(tiers, tier_rows)}
    full_tier_rows: list[Optional[dict]] = [rows_by_tier.get(t) for t in SUPPORTED_TIERS]

    class_rows = aggregate_class_rollup([r for r in tier_rows if r is not None], qa_index)

    judge_model, judge_emb = _detect_judge_provenance(results_dir)

    capture_provenance = []
    for r in tier_rows:
        if r is None:
            continue
        capture_provenance.append({
            "tier_label": r["tier_label"],
            "timestamp": r.get("timestamp", "—"),
            "model": r.get("model", "—"),
            "git_sha": r.get("git_sha", "—"),
        })

    tier_4_present = any(
        r is not None for t, r in zip(tiers, tier_rows) if t == 4
    )

    md = emit_markdown(
        full_tier_rows,
        class_rows,
        judge_model,
        judge_emb,
        capture_provenance,
        tier_4_present,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 131 Stage 3 — emit comparison.md.")
    p.add_argument("--results-dir", default="evaluation/results",
                   help="Parent of queries/, costs/, metrics/. Default: evaluation/results.")
    p.add_argument("--out", default="evaluation/results/comparison.md",
                   help="Output Markdown path. Default: evaluation/results/comparison.md.")
    p.add_argument("--tiers", default="1,2,3,4,5",
                   help="Comma-separated tier numbers to include. Default: all 5.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
