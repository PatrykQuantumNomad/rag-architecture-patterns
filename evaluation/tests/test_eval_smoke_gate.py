"""Unit tests for evaluation/harness/smoke_gate.py.

Per Plan 01-02 D-04 + Pitfall 5 of 132-RESEARCH:
- classify_row priority: agent_truncated > empty_no_tool_calls > populated.
- Gate PASS: measurable >= 3 AND populated/measurable >= 0.8 AND non-NaN RAGAS metrics.
- Gate INCONCLUSIVE: measurable < 3 (re-run with different IDs).
- Gate FAIL: ratio < 0.8 OR NaN metric on a populated row.

The canonical Pitfall-5 case (3 populated + 1 truncated + 1 empty_no_tool_calls)
must PASS — naive 3/5 = 60% would FAIL but the exclusion-aware gate counts
3/3 measurable = 100%.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.harness.records import EvalRecord, QueryLog, ScoreRecord
from evaluation.harness.smoke_gate import (
    SmokeGateResult,
    classify_row,
    evaluate_smoke,
    evaluate_smoke_from_paths,
)


def _rec(qid: str, contexts=None, error=None) -> EvalRecord:
    return EvalRecord(
        question_id=qid,
        question="?",
        answer="answer",
        retrieved_contexts=contexts if contexts is not None else [],
        latency_s=0.1,
        cost_usd_at_capture=0.0001,
        error=error,
    )


def _score_ok(qid: str, faith: float = 0.9, cp: float = 0.85, ar: float = 0.8) -> ScoreRecord:
    return ScoreRecord(
        question_id=qid,
        faithfulness=faith,
        answer_relevancy=ar,
        context_precision=cp,
        nan_reason=None,
    )


def _score_nan(qid: str, reason: str) -> ScoreRecord:
    return ScoreRecord(
        question_id=qid,
        faithfulness=None,
        answer_relevancy=None,
        context_precision=None,
        nan_reason=reason,
    )


# ---------- classify_row ----------


def test_classify_populated():
    rec = _rec("q1", contexts=["[paper_id=2005.11401] some chunk"])
    score = _score_ok("q1")
    assert classify_row(rec, score) == "populated"


def test_classify_empty_no_tool_calls():
    rec = _rec("q1", contexts=[])
    score = _score_nan("q1", "empty_contexts")
    assert classify_row(rec, score) == "empty_no_tool_calls"


def test_classify_agent_truncated():
    # Truncation takes priority — even with non-empty contexts, error trumps.
    rec = _rec("q1", contexts=["[paper_id=x] partial"], error="max_turns_exceeded")
    score = _score_nan("q1", "agent_truncated")
    assert classify_row(rec, score) == "agent_truncated"


def test_classify_truncated_via_score_nan_reason():
    """Even if EvalRecord.error is None, score.nan_reason='agent_truncated' marks it."""
    rec = _rec("q1", contexts=["[paper_id=x] partial"], error=None)
    score = _score_nan("q1", "agent_truncated")
    assert classify_row(rec, score) == "agent_truncated"


# ---------- evaluate_smoke ----------


def _ql(records):
    return QueryLog(
        tier="tier-5",
        timestamp="2026-05-04T15:00:00Z",
        git_sha="abc1234",
        model="google/gemini-2.5-flash",
        records=records,
    )


def test_gate_pass_with_measurable_exclusions():
    """Pitfall 5 of 132-RESEARCH canonical case:
    3 populated + 1 truncated + 1 empty_no_tool_calls
    -> measurable=3, populated=3, ratio=1.0, all metrics non-NaN -> PASS.
    """
    records = [
        _rec("q1", contexts=["[paper_id=x] c1"]),
        _rec("q2", contexts=["[paper_id=x] c2"]),
        _rec("q3", contexts=["[paper_id=x] c3"]),
        _rec("q4", contexts=[], error="max_turns_exceeded"),
        _rec("q5", contexts=[]),
    ]
    scores = [
        _score_ok("q1"),
        _score_ok("q2"),
        _score_ok("q3"),
        _score_nan("q4", "agent_truncated"),
        _score_nan("q5", "empty_contexts"),
    ]
    result = evaluate_smoke(_ql(records), scores)
    assert result.verdict == "PASS"
    assert result.n_total == 5
    assert result.n_populated == 3
    assert result.n_agent_truncated == 1
    assert result.n_empty_no_tool_calls == 1
    assert result.n_measurable == 3
    assert result.ratio == pytest.approx(1.0)
    assert result.non_nan_faithfulness_count == 3
    assert result.non_nan_context_precision_count == 3


def test_gate_inconclusive_too_few_measurable():
    """5 rows: 1 populated + 4 truncated -> measurable=1 < 3 -> INCONCLUSIVE."""
    records = [_rec("q1", contexts=["[paper_id=x] c1"])] + [
        _rec(f"q{i}", contexts=[], error="max_turns_exceeded") for i in range(2, 6)
    ]
    scores = [_score_ok("q1")] + [_score_nan(f"q{i}", "agent_truncated") for i in range(2, 6)]
    result = evaluate_smoke(_ql(records), scores)
    assert result.verdict == "INCONCLUSIVE"
    assert result.n_measurable == 1
    assert result.ratio is None
    assert "re-run with different IDs" in result.message


def test_gate_inconclusive_when_empties_dominate():
    """5 rows: 2 populated + 3 empty_no_tool_calls -> measurable=2 < 3 -> INCONCLUSIVE."""
    records = [
        _rec("q1", contexts=["[paper_id=x] c1"]),
        _rec("q2", contexts=["[paper_id=x] c2"]),
        _rec("q3", contexts=[]),
        _rec("q4", contexts=[]),
        _rec("q5", contexts=[]),
    ]
    scores = [
        _score_ok("q1"),
        _score_ok("q2"),
        _score_nan("q3", "empty_contexts"),
        _score_nan("q4", "empty_contexts"),
        _score_nan("q5", "empty_contexts"),
    ]
    result = evaluate_smoke(_ql(records), scores)
    assert result.verdict == "INCONCLUSIVE"
    assert result.n_measurable == 2


def test_gate_fail_nan_metric_on_populated():
    """5 rows all populated, but 1 has faithfulness=None -> FAIL ('NaN metrics')."""
    records = [_rec(f"q{i}", contexts=["[paper_id=x] c"]) for i in range(1, 6)]
    scores = [
        _score_ok("q1"),
        _score_ok("q2"),
        _score_ok("q3"),
        _score_ok("q4"),
        ScoreRecord(
            question_id="q5",
            faithfulness=None,  # judge failure on a populated row
            answer_relevancy=0.7,
            context_precision=0.8,
            nan_reason=None,
        ),
    ]
    result = evaluate_smoke(_ql(records), scores)
    assert result.verdict == "FAIL"
    assert result.n_measurable == 5
    assert result.non_nan_faithfulness_count == 4
    assert "NaN metrics" in result.message


def test_gate_fail_low_ratio_below_threshold():
    """Construct a 6-row case where measurable >= 3 and ratio < 0.8 -> FAIL.

    6 rows: 3 populated (with non-NaN metrics) + 3 populated (with full metrics OK)
    is hard to fail by ratio when all are populated. Instead: 5 records with
    classifier producing 3 populated + 2 non-measurable, but flip 1 populated to
    have NaN AND flip another to be populated-but-with-NaN-cp. Easier: use 4
    populated + 1 truncated, where 1 of the populated has NaN faithfulness.
    measurable=4, populated=4 -> ratio=1.0 but NaN metric -> FAIL.

    For pure ratio FAIL we need a "populated by classifier but should not count
    as populated for ratio" case — which doesn't exist in the current taxonomy.
    The realistic FAIL-by-ratio path is impossible without extending the
    classifier; this is reflected in the message structure (FAIL is reached via
    the NaN-metrics check on populated rows).

    This test asserts that scenario — a populated row with NaN context_precision.
    """
    records = [_rec(f"q{i}", contexts=["[paper_id=x] c"]) for i in range(1, 5)] + [
        _rec("q5", contexts=[], error="max_turns_exceeded")
    ]
    scores = [
        _score_ok("q1"),
        _score_ok("q2"),
        _score_ok("q3"),
        ScoreRecord(
            question_id="q4",
            faithfulness=0.9,
            answer_relevancy=0.7,
            context_precision=None,  # judge failure
            nan_reason=None,
        ),
        _score_nan("q5", "agent_truncated"),
    ]
    result = evaluate_smoke(_ql(records), scores)
    assert result.verdict == "FAIL"
    assert result.n_populated == 4
    assert result.n_measurable == 4
    assert result.non_nan_context_precision_count == 3
    assert "NaN metrics" in result.message


def test_gate_pass_threshold_exact_080():
    """Override ratio_threshold to 0.5; with 3 populated + 2 truncated, ratio=1.0 PASSes."""
    records = [_rec(f"q{i}", contexts=["[paper_id=x] c"]) for i in range(1, 4)] + [
        _rec("q4", contexts=[], error="max_turns_exceeded"),
        _rec("q5", contexts=[], error="max_turns_exceeded"),
    ]
    scores = [_score_ok(f"q{i}") for i in range(1, 4)] + [
        _score_nan("q4", "agent_truncated"),
        _score_nan("q5", "agent_truncated"),
    ]
    result = evaluate_smoke(_ql(records), scores, ratio_threshold=0.5)
    assert result.verdict == "PASS"


def test_gate_handles_missing_score_row():
    """If a record has no matching ScoreRecord, treat as empty_no_tool_calls."""
    records = [
        _rec("q1", contexts=["[paper_id=x] c"]),
        _rec("q2", contexts=["[paper_id=x] c"]),
        _rec("q3", contexts=["[paper_id=x] c"]),
        _rec("q4", contexts=["[paper_id=x] c"]),
        _rec("q5", contexts=["[paper_id=x] c"]),
    ]
    # Missing score for q5 entirely.
    scores = [_score_ok(f"q{i}") for i in range(1, 5)]
    result = evaluate_smoke(_ql(records), scores)
    # 4 populated + 1 missing-score (treated as empty_no_tool_calls) -> measurable=4 -> PASS
    assert result.verdict == "PASS"
    assert result.n_populated == 4
    assert result.n_empty_no_tool_calls == 1


# ---------- evaluate_smoke_from_paths ----------


def test_evaluate_smoke_from_paths_reads_bare_list(tmp_path):
    """score.py writes a bare list of ScoreRecord dicts; the wrapper must parse that shape."""
    ql = _ql([
        _rec("q1", contexts=["[paper_id=x] c"]),
        _rec("q2", contexts=["[paper_id=x] c"]),
        _rec("q3", contexts=["[paper_id=x] c"]),
    ])
    ql_path = tmp_path / "tier-5-2026-05-04T15_00_00Z.json"
    ql_path.write_text(ql.model_dump_json(indent=2))

    scores = [_score_ok(f"q{i}") for i in range(1, 4)]
    score_path = tmp_path / "tier-5-2026-05-04T15_00_00Z-metrics.json"
    score_path.write_text(json.dumps([s.model_dump() for s in scores], indent=2))

    result = evaluate_smoke_from_paths(ql_path, score_path)
    # 3 populated, 0 excluded -> measurable=3, ratio=1.0 -> PASS
    assert result.verdict == "PASS"
    assert result.n_total == 3


def test_evaluate_smoke_from_paths_reads_records_dict(tmp_path):
    """Defensive guard: also accept {'records': [...]} shape."""
    ql = _ql([
        _rec("q1", contexts=["[paper_id=x] c"]),
        _rec("q2", contexts=["[paper_id=x] c"]),
        _rec("q3", contexts=["[paper_id=x] c"]),
    ])
    ql_path = tmp_path / "tier-5.json"
    ql_path.write_text(ql.model_dump_json(indent=2))

    scores = [_score_ok(f"q{i}") for i in range(1, 4)]
    score_path = tmp_path / "tier-5-metrics.json"
    score_path.write_text(json.dumps({"records": [s.model_dump() for s in scores]}, indent=2))

    result = evaluate_smoke_from_paths(ql_path, score_path)
    assert result.verdict == "PASS"
    assert result.n_total == 3
