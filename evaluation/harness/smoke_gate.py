"""Smoke gate evaluator for Phase 1/2 (Tier 5 / Tier 4 fixes).

Reads QueryLog + list[ScoreRecord], classifies each row, computes a measurable
denominator (excluding agent_truncated and empty_no_tool_calls), and applies
the >=0.8 populated/measurable ratio gate per
.planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md D-04.

Returns one of: PASS, FAIL, INCONCLUSIVE.

- PASS:         measurable >= min_measurable AND populated/measurable >= ratio_threshold
                AND faithfulness/context_precision are non-NaN on every populated row.
- FAIL:         measurable >= min_measurable but ratio < threshold or NaN metrics on populated.
- INCONCLUSIVE: fewer than `min_measurable` cells are measurable. Legitimate Pitfall-9
                self-cite and/or Pitfall-8 truncation ate the smoke; re-run with
                different IDs or investigate.

Pitfall references:
- Pitfall 5 of 132-RESEARCH: classify BEFORE computing denominator. The canonical
  case (3 populated + 1 truncated + 1 empty) MUST PASS — naive 3/5=60% would FAIL,
  exclusion-aware ratio is 3/3=100%.
- Pitfall 8 of 130-RESEARCH: agent_truncated rows excluded from denominator.
- Pitfall 9 of 130-RESEARCH: zero-tool-call rows (empty_contexts) excluded from denominator.

Run as a CLI:
    python -m evaluation.harness.smoke_gate --tier 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.records import EvalRecord, QueryLog, ScoreRecord, read_query_log

Classification = Literal["populated", "empty_no_tool_calls", "agent_truncated"]
Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


class SmokeGateResult(BaseModel):
    """Structured smoke-gate verdict — read by humans + downstream summaries."""

    verdict: Verdict
    n_total: int
    n_populated: int
    n_empty_no_tool_calls: int
    n_agent_truncated: int
    n_measurable: int
    ratio: Optional[float]  # populated / measurable; None when measurable=0
    non_nan_faithfulness_count: int
    non_nan_context_precision_count: int
    message: str


def classify_row(rec: EvalRecord, score: Optional[ScoreRecord]) -> Classification:
    """Classify one (record, score) pair into the smoke-gate taxonomy.

    Pitfall 5 of 132-RESEARCH: priority ordering matters.
        agent_truncated > empty_no_tool_calls > populated.

    Truncation takes priority on the EvalRecord side (Pitfall 8 of 130-RESEARCH:
    `error == "max_turns_exceeded"`). If the score row records `nan_reason ==
    "agent_truncated"` that's also truncation. After that, a missing score row
    OR empty contexts both classify as `empty_no_tool_calls` — the judge could
    not produce metrics, so the row is excluded from the measurable denominator
    per D-04.
    """
    if rec.error == "max_turns_exceeded" or (
        score is not None and score.nan_reason == "agent_truncated"
    ):
        return "agent_truncated"
    if score is None or not rec.retrieved_contexts:
        return "empty_no_tool_calls"
    return "populated"


def evaluate_smoke(
    query_log: QueryLog,
    scores: list[ScoreRecord],
    min_measurable: int = 3,
    ratio_threshold: float = 0.8,
) -> SmokeGateResult:
    """Apply the gate per D-04. Pure function over Pydantic-typed inputs.

    Args:
        query_log: capture-stage QueryLog (from `evaluation.harness.run`).
        scores:    score-stage list of ScoreRecord (from `evaluation.harness.score`).
        min_measurable: minimum measurable cells before a non-INCONCLUSIVE verdict
                        is possible. Default 3 per D-04.
        ratio_threshold: minimum populated/measurable ratio for PASS. Default 0.8
                         (4/5-equivalent on a 5-question smoke).

    Returns:
        SmokeGateResult with verdict, counts, ratio, and a human-readable message.
    """
    scores_by_id = {s.question_id: s for s in scores}
    classified: list[tuple[EvalRecord, Optional[ScoreRecord], Classification]] = []
    for rec in query_log.records:
        score = scores_by_id.get(rec.question_id)
        classified.append((rec, score, classify_row(rec, score)))

    n_total = len(classified)
    n_populated = sum(1 for _, _, c in classified if c == "populated")
    n_empty = sum(1 for _, _, c in classified if c == "empty_no_tool_calls")
    n_truncated = sum(1 for _, _, c in classified if c == "agent_truncated")
    n_measurable = n_populated  # truncated and empty_no_tool_calls excluded per D-04

    # Non-NaN counts on the populated subset (D-04 score gate).
    non_nan_faith = sum(
        1
        for _, s, c in classified
        if c == "populated" and s is not None and s.faithfulness is not None
    )
    non_nan_cp = sum(
        1
        for _, s, c in classified
        if c == "populated" and s is not None and s.context_precision is not None
    )

    if n_measurable < min_measurable:
        return SmokeGateResult(
            verdict="INCONCLUSIVE",
            n_total=n_total,
            n_populated=n_populated,
            n_empty_no_tool_calls=n_empty,
            n_agent_truncated=n_truncated,
            n_measurable=n_measurable,
            ratio=None,
            non_nan_faithfulness_count=non_nan_faith,
            non_nan_context_precision_count=non_nan_cp,
            message=(
                f"INCONCLUSIVE: only {n_measurable}/{n_total} cells measurable "
                f"(need >={min_measurable}); re-run with different IDs or "
                f"investigate Pitfall-9 self-cite rate "
                f"(empty={n_empty}, truncated={n_truncated})."
            ),
        )

    ratio = n_populated / max(n_measurable, 1)
    all_metrics_ok = (
        non_nan_faith == n_populated and non_nan_cp == n_populated
    )
    if ratio >= ratio_threshold and all_metrics_ok:
        return SmokeGateResult(
            verdict="PASS",
            n_total=n_total,
            n_populated=n_populated,
            n_empty_no_tool_calls=n_empty,
            n_agent_truncated=n_truncated,
            n_measurable=n_measurable,
            ratio=ratio,
            non_nan_faithfulness_count=non_nan_faith,
            non_nan_context_precision_count=non_nan_cp,
            message=(
                f"PASS: {n_populated}/{n_measurable} measurable populated "
                f"(ratio={ratio:.2f} >= {ratio_threshold}); all RAGAS metrics non-NaN."
            ),
        )

    reasons: list[str] = []
    if ratio < ratio_threshold:
        reasons.append(f"ratio {ratio:.2f} < {ratio_threshold}")
    if not all_metrics_ok:
        reasons.append(
            f"NaN metrics on populated rows "
            f"(faith={non_nan_faith}/{n_populated}, cp={non_nan_cp}/{n_populated})"
        )
    return SmokeGateResult(
        verdict="FAIL",
        n_total=n_total,
        n_populated=n_populated,
        n_empty_no_tool_calls=n_empty,
        n_agent_truncated=n_truncated,
        n_measurable=n_measurable,
        ratio=ratio,
        non_nan_faithfulness_count=non_nan_faith,
        non_nan_context_precision_count=non_nan_cp,
        message=f"FAIL: {' AND '.join(reasons)}.",
    )


def evaluate_smoke_from_paths(
    query_log_path: Path,
    score_path: Path,
    min_measurable: int = 3,
    ratio_threshold: float = 0.8,
) -> SmokeGateResult:
    """CLI / live-test convenience wrapper. Reads JSON from disk.

    `score.py:_persist_metrics` writes a BARE LIST of ScoreRecord dicts. We also
    accept a `{"records": [...]}` dict shape as a defensive forward-compat guard;
    the current writer does not produce that shape today.
    """
    ql = read_query_log(query_log_path)
    score_data = json.loads(Path(score_path).read_text())
    if isinstance(score_data, dict) and "records" in score_data:
        score_data = score_data["records"]
    scores = [ScoreRecord(**s) for s in score_data]
    return evaluate_smoke(
        ql, scores, min_measurable=min_measurable, ratio_threshold=ratio_threshold
    )


def _build_parser() -> argparse.ArgumentParser:
    """argparse parser; module-level so tests can introspect."""
    p = argparse.ArgumentParser(
        description=(
            "Plan 01-02 smoke gate — classify Tier 5/4 capture+score outputs and "
            "report PASS / FAIL / INCONCLUSIVE per D-04."
        )
    )
    p.add_argument(
        "--tier",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Tier number whose latest capture+score outputs to gate.",
    )
    p.add_argument(
        "--results-dir",
        default="evaluation/results",
        help="Parent of queries/ + metrics/. Default: evaluation/results.",
    )
    p.add_argument(
        "--min-measurable",
        type=int,
        default=3,
        help="Minimum measurable cells before a non-INCONCLUSIVE verdict (D-04). Default: 3.",
    )
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.8,
        help="Minimum populated/measurable ratio for PASS (D-04). Default: 0.8.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry. Resolves the latest query+score files for `--tier` then evaluates."""
    # Late import to keep --help responsive.
    from evaluation.harness.compare import _latest

    args = _build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = _REPO_ROOT / results_dir

    query_log_path = _latest(results_dir / "queries", f"tier-{args.tier}-*.json")
    score_path = _latest(results_dir / "metrics", f"tier-{args.tier}-*.json")
    if query_log_path is None:
        print(
            f"ERROR: no query log found in {results_dir / 'queries'} "
            f"for tier-{args.tier}.",
            file=sys.stderr,
        )
        return 2
    if score_path is None:
        print(
            f"ERROR: no score file found in {results_dir / 'metrics'} "
            f"for tier-{args.tier}.",
            file=sys.stderr,
        )
        return 2

    result = evaluate_smoke_from_paths(
        query_log_path,
        score_path,
        min_measurable=args.min_measurable,
        ratio_threshold=args.ratio_threshold,
    )
    print(result.model_dump_json(indent=2))
    if result.verdict == "PASS":
        return 0
    if result.verdict == "INCONCLUSIVE":
        return 3
    return 1  # FAIL


__all__ = [
    "Classification",
    "Verdict",
    "SmokeGateResult",
    "classify_row",
    "evaluate_smoke",
    "evaluate_smoke_from_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
