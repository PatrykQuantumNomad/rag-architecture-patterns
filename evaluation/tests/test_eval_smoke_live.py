"""Live smoke for Phase 131 — Tier 1 x 1 question through all 3 stages.

@pytest.mark.live; gated on live_eval_keys_ok + tier1_index_present (both ship in
Plan 01's evaluation/tests/conftest.py).

Cost: ~$0.005-0.01 per run. Wall time: <60s. Question count: 1.

Empirically resolves:
- Open Q1 (token_usage_parser parses OpenRouter responses) -- judge cost > 0 confirms.
- Open Q2 (batch_size=10 sufficient at concurrency 1) -- trivially yes for n=1.
- A2 (token_usage_parser correctness) -- same as Open Q1.
- A6 (EvaluationResult.to_pandas()) -- exception path tells us if it works in 0.4.x.
- A8 (OPENROUTER_API_KEY sole-key sufficiency) -- judge LLM + embedder both LiteLLM/OR.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

import pytest

from evaluation.harness.records import EvalRecord, QueryLog, write_query_log


SMOKE_QUESTION = {
    "id": "single-hop-001",
    "question": (
        "What is the core mechanism Lewis et al. 2020 introduce in the RAG paper "
        "for combining parametric and non-parametric memory?"
    ),
    "expected_answer": (
        "RAG models combine a pre-trained seq2seq parametric memory (BART) with a "
        "non-parametric memory (a dense vector index of Wikipedia accessed via a "
        "pre-trained neural retriever, DPR). Two formulations are proposed: "
        "RAG-Sequence, which conditions on the same retrieved passages across the "
        "whole generated sequence, and RAG-Token, which can use different retrieved "
        "passages per generated token."
    ),
}


@pytest.mark.live
def test_eval_smoke_tier1_full_pipeline(live_eval_keys_ok, tier1_index_present, tmp_path):
    """End-to-end smoke: run.py -> score.py -> compare.py on Tier 1 x 1 question.

    Asserts:
      Stage 1: EvalRecord with non-empty answer, non-empty contexts, cost > 0, latency > 0.
      Stage 2: ScoreRecord with all 3 metrics in [0, 1] (NOT NaN).
      Stage 3: aggregate_tier produces a populated row with n=1, n_nan=0.
    Empirical resolution:
      A2 / Open Q1: usage["input_tokens"] > 0 confirms token_usage_parser worked.
    """
    from evaluation.harness.adapters.tier_1 import run_tier1
    from evaluation.harness.score import score_query_log
    from evaluation.harness.compare import aggregate_tier, _classify

    # --- Stage 1: capture ---
    rec = asyncio.run(run_tier1(
        question_id=SMOKE_QUESTION["id"],
        question=SMOKE_QUESTION["question"],
        k=5,
    ))
    assert isinstance(rec, EvalRecord)
    assert rec.error is None, f"Tier 1 captured an error: {rec.error}"
    assert len(rec.answer.strip()) > 0, "Tier 1 returned empty answer"
    assert len(rec.retrieved_contexts) > 0, "Tier 1 returned no retrieved contexts"
    assert rec.cost_usd_at_capture > 0.0, "Tier 1 cost was not tracked"
    assert rec.latency_s > 0.0, "Tier 1 latency was not tracked"

    # Persist query log to tmp_path (isolated from real evaluation/results/)
    log = QueryLog(
        tier="tier-1",
        timestamp="2026-04-27T00:00:00Z",  # synthetic timestamp for tmp_path isolation
        git_sha="smoke-test",
        model="google/gemini-2.5-flash",
        records=[rec],
    )
    queries_dir = tmp_path / "queries"
    costs_dir = tmp_path / "costs"
    metrics_dir = tmp_path / "metrics"
    queries_dir.mkdir()
    costs_dir.mkdir()
    metrics_dir.mkdir()

    # Synthesize the cost JSON in D-13 shape (the real one was persisted by the adapter
    # to evaluation/results/costs/ -- for tmp_path isolation, mirror its shape here).
    (costs_dir / "tier-1-eval-2026-04-27T00_00_00Z.json").write_text(json.dumps({
        "tier": "tier-1-eval",
        "timestamp": "2026-04-27T00:00:00Z",
        "queries": [],
        "totals": {
            "usd": rec.cost_usd_at_capture,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "embedding_tokens": 0,
        },
    }))
    write_query_log(queries_dir / "tier-1-2026-04-27T00_00_00Z.json", log)

    # --- Stage 2: score (real RAGAS + real OpenRouter judge) ---
    qa_index = {SMOKE_QUESTION["id"]: SMOKE_QUESTION}

    import litellm
    from ragas.llms import llm_factory
    try:
        from ragas.embeddings.base import embedding_factory
    except ImportError:
        from ragas.embeddings import embedding_factory

    judge_llm = llm_factory(
        "openrouter/google/gemini-2.5-flash",
        provider="litellm",
        client=litellm.completion,
    )
    judge_emb = embedding_factory(
        "litellm",
        model="openrouter/openai/text-embedding-3-small",
    )

    scores, usage = asyncio.run(score_query_log(
        log,
        qa_index,
        judge_llm=judge_llm,
        judge_emb=judge_emb,
        batch_size=10,
        raise_exceptions=False,
    ))

    assert len(scores) == 1, "Expected exactly 1 ScoreRecord"
    sr = scores[0]
    assert sr.nan_reason is None, f"Smoke score was NaN with reason: {sr.nan_reason}"
    assert sr.faithfulness is not None and 0.0 <= sr.faithfulness <= 1.0, \
        f"Faithfulness out of range: {sr.faithfulness}"
    assert sr.answer_relevancy is not None and 0.0 <= sr.answer_relevancy <= 1.0, \
        f"Answer relevancy out of range: {sr.answer_relevancy}"
    assert sr.context_precision is not None and 0.0 <= sr.context_precision <= 1.0, \
        f"Context precision out of range: {sr.context_precision}"

    # A2 / Open Q1 empirical resolution: judge cost > 0 means token_usage_parser worked.
    assert usage["input_tokens"] > 0, "Judge token_usage_parser returned zero input_tokens"
    assert usage["output_tokens"] >= 0, "Negative output_tokens (impossible)"

    # Persist metrics in the shape compare.py expects.
    metrics_payload = [s.model_dump() for s in scores]
    (metrics_dir / "tier-1-2026-04-27T00_00_00Z.json").write_text(
        json.dumps(metrics_payload, indent=2)
    )

    # --- Stage 3: compare ---
    row = aggregate_tier(1, tmp_path)
    assert row is not None, "aggregate_tier returned None on populated tmp_path"
    assert row["n"] == 1, f"Expected n=1, got {row['n']}"
    assert row["n_nan"] == 0, f"Expected n_nan=0, got {row['n_nan']}"
    assert 0.0 <= row["faithfulness"] <= 1.0, \
        f"Aggregated faithfulness out of range: {row['faithfulness']}"
    assert row["total_cost_usd"] > 0.0, "Aggregated total_cost_usd was zero"

    # _classify smoke check: SMOKE_QUESTION is single-hop.
    qa_index_full = {SMOKE_QUESTION["id"]: {
        "hop_count_tag": "single-hop",
        "modality_tag": "text",
    }}
    assert _classify(SMOKE_QUESTION["id"], qa_index_full) == "single-hop"
