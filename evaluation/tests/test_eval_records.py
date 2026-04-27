"""Non-live unit tests for evaluation/harness/records.py."""
from pathlib import Path

import pytest

from evaluation.harness.records import (
    EvalRecord,
    QueryLog,
    ScoreRecord,
    read_query_log,
    write_query_log,
)


def test_eval_record_defaults():
    rec = EvalRecord(
        question_id="single-hop-001",
        question="What is RAG?",
        answer="RAG combines parametric and non-parametric memory.",
        latency_s=1.5,
        cost_usd_at_capture=0.0001,
    )
    assert rec.retrieved_contexts == []
    assert rec.error is None


def test_eval_record_full():
    rec = EvalRecord(
        question_id="multi-hop-002",
        question="How does DPR differ from RAG?",
        answer="DPR is dense retrieval; RAG combines retrieval with generation.",
        retrieved_contexts=["chunk 1", "chunk 2"],
        latency_s=4.32,
        cost_usd_at_capture=0.000795,
        error=None,
    )
    assert len(rec.retrieved_contexts) == 2
    assert rec.latency_s == 4.32


def test_query_log_roundtrip(tmp_path):
    log = QueryLog(
        tier="tier-1",
        timestamp="2026-04-27T12:00:00Z",
        git_sha="abc1234",
        model="google/gemini-2.5-flash",
        records=[
            EvalRecord(
                question_id="q1",
                question="What is RAG?",
                answer="RAG combines memory.",
                retrieved_contexts=["c1"],
                latency_s=1.0,
                cost_usd_at_capture=0.0001,
            ),
        ],
    )
    out = tmp_path / "tier-1-2026-04-27T12_00_00Z.json"
    write_query_log(out, log)
    assert out.exists()

    loaded = read_query_log(out)
    assert loaded.tier == "tier-1"
    assert loaded.git_sha == "abc1234"
    assert len(loaded.records) == 1
    assert loaded.records[0].question_id == "q1"


def test_score_record_defaults_all_nan():
    sr = ScoreRecord(question_id="q1", nan_reason="empty_contexts")
    assert sr.faithfulness is None
    assert sr.answer_relevancy is None
    assert sr.context_precision is None
    assert sr.nan_reason == "empty_contexts"


def test_score_record_truncated():
    sr = ScoreRecord(
        question_id="multi-hop-005",
        nan_reason="agent_truncated",
    )
    assert sr.nan_reason == "agent_truncated"
    assert all(v is None for v in (sr.faithfulness, sr.answer_relevancy, sr.context_precision))


def test_score_record_full_pass():
    sr = ScoreRecord(
        question_id="q1",
        faithfulness=0.85,
        answer_relevancy=0.92,
        context_precision=0.78,
    )
    assert sr.nan_reason is None
    assert 0.0 <= sr.faithfulness <= 1.0
