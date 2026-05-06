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


# ---------------------------------------------------------------------------
# Phase 6 / Plan 06-01 — embedder + embedder_source provenance fields.
# ---------------------------------------------------------------------------


def test_query_log_carries_embedder_field():
    """QueryLog accepts and round-trips embedder + embedder_source.

    CAP-03 / 06-01 Task 1. Construct a QueryLog with both new fields and
    confirm they survive Pydantic v2 model_dump_json -> model_validate_json
    round-trip with their original values. This locks the schema contract
    consumed by run.py / eval_capture.py / compare.py / freeze.py.
    """
    log = QueryLog(
        tier="tier-1",
        timestamp="2026-05-06T12:00:00Z",
        git_sha="abc1234",
        model="google/gemini-2.5-flash",
        embedder="openai/text-embedding-3-small",
        embedder_source="openrouter",
        records=[],
    )
    assert log.embedder == "openai/text-embedding-3-small"
    assert log.embedder_source == "openrouter"

    roundtrip = QueryLog.model_validate_json(log.model_dump_json())
    assert roundtrip.embedder == "openai/text-embedding-3-small"
    assert roundtrip.embedder_source == "openrouter"
    # Pre-existing fields must still round-trip cleanly (regression guard).
    assert roundtrip.tier == "tier-1"
    assert roundtrip.git_sha == "abc1234"
    assert roundtrip.model == "google/gemini-2.5-flash"


def test_query_log_legacy_json_loads_with_none_embedder():
    """Legacy QueryLog JSONs (no embedder fields) load with None defaults.

    CAP-03 / 06-01 Task 1 / D-BACKCOMPAT. evaluation/results/queries/ is
    gitignored — older locally-cached JSONs lack the new embedder fields.
    Pydantic v2's Optional[str] = None default must absorb their absence
    without raising; the loaded instance reads None for both new fields.
    """
    legacy_json = (
        '{"tier":"tier-1","timestamp":"2026-05-02T17:26:59Z",'
        '"git_sha":"ce5c2ad","model":"google/gemini-2.5-flash","records":[]}'
    )
    log = QueryLog.model_validate_json(legacy_json)
    assert log.embedder is None
    assert log.embedder_source is None
    # Pre-existing fields still load (regression guard).
    assert log.tier == "tier-1"
    assert log.git_sha == "ce5c2ad"


def test_per_tier_embedder_constants_importable():
    """Each tier module exports the (EMBED_MODEL, EMBEDDER_SOURCE) tuple
    used by run.py / eval_capture.py to populate QueryLog fields.

    Plan 06-01 Task 2 / D-ROADMAP-OVERRIDE / D-Q2.

    Tier 5 reuses Tier 1's embedder (verified in 06-RESEARCH.md against
    tier-5-agentic/tools.py:47-50,90-101) — overriding ROADMAP's incorrect
    "OpenAI hosted vector-store" claim. Tier 5 imports EMBED_MODEL
    transitively via `from tier_1_naive.embed_openai import EMBED_MODEL`,
    so the import below is resolvable from tier_5_agentic.tools.
    """
    from tier_1_naive.embed_openai import EMBED_MODEL as T1_EMB, EMBEDDER_SOURCE as T1_SRC
    from tier_2_managed.main import EMBED_MODEL as T2_EMB, EMBEDDER_SOURCE as T2_SRC
    from tier_3_graph.rag import DEFAULT_EMBED_MODEL as T3_EMB, EMBEDDER_SOURCE as T3_SRC
    from tier_4_multimodal.rag import DEFAULT_EMBED_MODEL as T4_EMB, EMBEDDER_SOURCE as T4_SRC
    from tier_5_agentic.tools import EMBED_MODEL as T5_EMB, EMBEDDER_SOURCE as T5_SRC

    assert T1_EMB == "openai/text-embedding-3-small" and T1_SRC == "openrouter"
    assert T2_EMB == "gemini-embedding-001" and T2_SRC == "google-managed"
    assert T3_EMB == "openai/text-embedding-3-small" and T3_SRC == "openrouter"
    assert T4_EMB == "openai/text-embedding-3-small" and T4_SRC == "openrouter"
    # D-ROADMAP-OVERRIDE: Tier 5 IS Tier 1's embedder, NOT a hosted vector store.
    assert T5_EMB == "openai/text-embedding-3-small" and T5_SRC == "openrouter"
