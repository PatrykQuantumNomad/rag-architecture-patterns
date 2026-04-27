"""Non-live unit tests for evaluation/harness/score.py.

Covers the deterministic surfaces that don't require a real judge LLM:
- _short_circuit_nan branches (Pitfalls 2 + 8 + tier4 + cached_miss)
- _to_float_or_none (NaN / pandas-NA mapping)
- _persist_metrics (JSON shape Plan 06's compare.py will read)
- score_query_log when EVERY record short-circuits → no RAGAS call
- argparse --help exits 0 (CLI smoke)

All tests run in <2s; no API keys consumed.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from evaluation.harness.records import EvalRecord, QueryLog, ScoreRecord


def test_short_circuit_empty_contexts():
    """Pitfall 2: empty retrieved_contexts → nan_reason='empty_contexts'."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="x",
        retrieved_contexts=[],
        latency_s=0.1,
        cost_usd_at_capture=0.0,
    )
    sr = _short_circuit_nan(rec)
    assert sr is not None
    assert sr.nan_reason == "empty_contexts"
    assert sr.faithfulness is None
    assert sr.answer_relevancy is None
    assert sr.context_precision is None


def test_short_circuit_agent_truncated():
    """Pitfall 8: error='max_turns_exceeded' → nan_reason='agent_truncated'."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="[truncated]",
        retrieved_contexts=["c"],
        latency_s=0.1,
        cost_usd_at_capture=0.0,
        error="max_turns_exceeded",
    )
    sr = _short_circuit_nan(rec)
    assert sr is not None
    assert sr.nan_reason == "agent_truncated"


def test_short_circuit_tier4_unavailable():
    """Tier 4 import error: empty contexts + error='tier4_import_error' →
    either nan_reason is acceptable (empty_contexts checked first in current order)."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="",
        retrieved_contexts=[],
        latency_s=0.0,
        cost_usd_at_capture=0.0,
        error="tier4_import_error: no module",
    )
    sr = _short_circuit_nan(rec)
    assert sr is not None
    assert sr.nan_reason in {"tier4_unavailable", "empty_contexts"}


def test_short_circuit_tier4_unavailable_with_contexts():
    """Tier 4 import error + non-empty contexts → nan_reason='tier4_unavailable' deterministically."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="",
        retrieved_contexts=["something"],
        latency_s=0.0,
        cost_usd_at_capture=0.0,
        error="tier4_import_error: ImportError(raganything)",
    )
    sr = _short_circuit_nan(rec)
    assert sr is not None
    assert sr.nan_reason == "tier4_unavailable"


def test_short_circuit_cached_miss():
    """Cached miss: error startswith 'cached_miss' + non-empty contexts → nan_reason='cached_miss'."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="",
        retrieved_contexts=["something"],
        latency_s=0.0,
        cost_usd_at_capture=0.0,
        error="cached_miss: question id not in cache file",
    )
    sr = _short_circuit_nan(rec)
    assert sr is not None
    assert sr.nan_reason == "cached_miss"


def test_short_circuit_passthrough():
    """Populated contexts + no error → returns None (let RAGAS score it)."""
    from evaluation.harness.score import _short_circuit_nan

    rec = EvalRecord(
        question_id="q1",
        question="?",
        answer="real answer",
        retrieved_contexts=["ctx"],
        latency_s=0.1,
        cost_usd_at_capture=0.0,
    )
    assert _short_circuit_nan(rec) is None


def test_to_float_or_none():
    """Maps None / float / numeric str / NaN / non-numeric to Optional[float] correctly."""
    from evaluation.harness.score import _to_float_or_none

    assert _to_float_or_none(None) is None
    assert _to_float_or_none(0.5) == 0.5
    assert _to_float_or_none("0.85") == 0.85
    assert _to_float_or_none(float("nan")) is None
    assert _to_float_or_none("not a number") is None


def test_persist_metrics(tmp_path):
    """Persistence schema: list of ScoreRecord-shaped dicts; nan rows have None metrics + nan_reason."""
    from evaluation.harness.score import _persist_metrics

    scores = [
        ScoreRecord(question_id="q1", faithfulness=0.85, answer_relevancy=0.9, context_precision=0.8),
        ScoreRecord(question_id="q2", nan_reason="empty_contexts"),
    ]
    out = tmp_path / "metrics" / "tier-1-2026-04-27T12_00_00Z.json"
    _persist_metrics(scores, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert data[0]["question_id"] == "q1"
    assert data[0]["faithfulness"] == 0.85
    assert data[0]["nan_reason"] is None
    assert data[1]["question_id"] == "q2"
    assert data[1]["nan_reason"] == "empty_contexts"
    assert data[1]["faithfulness"] is None


def test_strip_openrouter_prefix():
    """Pattern 7 helper: strips openrouter/ prefix; passes plain slugs through."""
    from evaluation.harness.score import _strip_openrouter_prefix

    assert _strip_openrouter_prefix("openrouter/google/gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert _strip_openrouter_prefix("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert _strip_openrouter_prefix("openrouter/openai/text-embedding-3-small") == "openai/text-embedding-3-small"


def test_score_query_log_all_short_circuit():
    """When every record short-circuits, score_query_log returns without calling RAGAS.

    Critical regression check: usage['n_scored']=0 and usage['input_tokens']=0 PROVE
    no judge call was made (so no judge cost was paid for known-bad records).
    """
    from evaluation.harness import score as score_mod

    log = QueryLog(
        tier="tier-1",
        timestamp="2026-04-27T12:00:00Z",
        git_sha="abc1234",
        model="m",
        records=[
            EvalRecord(
                question_id="q1",
                question="?",
                answer="",
                retrieved_contexts=[],
                latency_s=0.1,
                cost_usd_at_capture=0.0,
            ),
            EvalRecord(
                question_id="q2",
                question="?",
                answer="[truncated]",
                retrieved_contexts=["c"],
                latency_s=0.1,
                cost_usd_at_capture=0.0,
                error="max_turns_exceeded",
            ),
        ],
    )
    qa_index = {"q1": {"expected_answer": "x"}, "q2": {"expected_answer": "y"}}

    scores, usage = asyncio.run(score_mod.score_query_log(log, qa_index))
    assert len(scores) == 2
    assert scores[0].nan_reason == "empty_contexts"
    assert scores[1].nan_reason == "agent_truncated"
    assert usage["n_scored"] == 0
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_cli_help_exits_zero():
    """argparse --help raises SystemExit(0) — required smoke for CI."""
    from evaluation.harness import score as score_mod

    parser = score_mod.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
