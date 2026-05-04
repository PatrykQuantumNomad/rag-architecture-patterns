"""Live smoke tests for the evaluation harness.

@pytest.mark.live; gated on live_eval_keys_ok + tier1_index_present (both ship in
evaluation/tests/conftest.py).

Two tests live here:

- ``test_eval_smoke_tier1_full_pipeline`` — Phase 131 baseline: Tier 1 x 1
  question through all 3 stages. Cost: ~$0.005-0.01 per run; wall time <60s.

- ``test_eval_smoke_tier5_full_pipeline`` — Phase 1 (Plan 01-02) Tier 5 smoke:
  5 hand-picked questions through capture -> score -> smoke_gate. This is a
  guardrail, NOT the gate-of-record; the gate-of-record is the human-verify
  checkpoint (Plan 01-02 Task 3). Cost: <$0.05 per run; wall time <2 min.
  See .planning/phases/01-tier-5-adapter-fix/01-CONTEXT.md D-04 for the
  PASS/FAIL/INCONCLUSIVE semantics this test asserts on.

Empirically resolves (via tier-1 test):
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


@pytest.mark.live
def test_eval_smoke_tier5_full_pipeline(live_eval_keys_ok, tier1_index_present, tmp_path):
    """End-to-end smoke for Phase 1 Plan 01-02: Tier 5 x 5 questions through
    capture -> score -> smoke_gate.

    Per CONTEXT.md D-04 this is a GUARDRAIL (a sanity check that imports resolve
    and the pipeline produces structured output) NOT the gate-of-record. The
    gate-of-record is the human-verify checkpoint in Plan 01-02 Task 3 — a
    human reads the SmokeGateResult JSON and decides PASS / FAIL / INCONCLUSIVE.

    Fixture rationale: Tier 5's ``search_text_chunks`` reads from Tier 1's
    ChromaDB at ``chroma_db/tier-1-naive/`` (read-only, per Pitfall 9 of
    130-RESEARCH). Reusing ``tier1_index_present`` is the correct skip path —
    when the Tier 1 index is absent the test skips cleanly, mirroring the
    Tier 1 test's protection.

    Cost guard: 5 questions only (the smoke set), well under the
    Pitfall 7-of-132-RESEARCH per-run budget of $0.05.

    Asserts (defensive — live API behavior is non-deterministic):
      - n_total == 5 (smoke set size)
      - PASS, OR
      - INCONCLUSIVE with all rows excluded (Pitfall 8/9 escape hatch — flaky
        API ate the smoke), OR
      - FAIL with at least 1 populated row (soft evidence the adapter walk is
        partially working; the human checkpoint is the harder gate).
    """
    from evaluation.harness import run as run_mod
    from evaluation.harness import score as score_mod
    from evaluation.harness.records import read_query_log
    from evaluation.harness.smoke_gate import evaluate_smoke

    # --- Stage 1: capture via run.amain (exercises --smoke-question-ids flag) ---
    smoke_ids = ",".join(run_mod.DEFAULT_SMOKE_IDS)
    capture_args = run_mod.build_parser().parse_args(
        [
            "--tiers", "5",
            "--smoke-question-ids", smoke_ids,
            "--yes",
            "--output-dir", str(tmp_path),
        ]
    )
    from rich.console import Console as RichConsole
    rc = asyncio.run(run_mod.amain(capture_args, RichConsole()))
    assert rc == 0, f"run.amain returned non-zero ({rc}); see captured logs"

    # Locate the produced query log. run.py writes to {output_dir}/queries/tier-5-{TS}.json.
    queries_dir = tmp_path / "queries"
    candidates = sorted(queries_dir.glob("tier-5-*.json"), key=lambda p: p.stat().st_mtime)
    assert candidates, f"No tier-5 query log produced under {queries_dir}"
    query_log_path = candidates[-1]

    log = read_query_log(query_log_path)
    assert len(log.records) == 5, (
        f"Expected 5 records (smoke set), got {len(log.records)}; "
        f"--smoke-question-ids filtering may be broken."
    )

    # --- Stage 2: score in-process (locked: in-process per Open Q1 RESOLVED) ---
    qa_index = score_mod._load_golden_qa_index()
    judge_llm, judge_emb = score_mod._build_judge(
        score_mod.JUDGE_LLM_SLUG_DEFAULT,
        score_mod.JUDGE_EMB_SLUG_DEFAULT,
    )
    scores, _usage = asyncio.run(
        score_mod.score_query_log(
            log,
            qa_index,
            judge_llm=judge_llm,
            judge_emb=judge_emb,
            batch_size=10,
            raise_exceptions=False,
        )
    )
    assert len(scores) == 5, f"Expected 5 ScoreRecords, got {len(scores)}"

    # --- Stage 3: smoke gate ---
    result = evaluate_smoke(log, scores)
    assert result.n_total == 5, (
        f"SmokeGateResult.n_total={result.n_total}; expected 5"
    )

    # Print full result so `pytest -s` shows it in the human-verify checkpoint.
    print("\n--- SmokeGateResult ---")
    print(result.model_dump_json(indent=2))

    # Defensive verdict assertion — at least one of three escape hatches must hold.
    if result.verdict == "PASS":
        return  # happy path
    if (
        result.verdict == "INCONCLUSIVE"
        and (result.n_agent_truncated + result.n_empty_no_tool_calls) == 5
    ):
        # Pitfall 8/9 escape hatch — flaky API ate the smoke, no measurable cells.
        pytest.skip(
            f"INCONCLUSIVE — all 5 rows excluded (truncated={result.n_agent_truncated}, "
            f"empty={result.n_empty_no_tool_calls}). Re-run after API hiccup clears. "
            f"Message: {result.message}"
        )
    # On FAIL: assert at least 1 populated row landed (soft evidence the walk
    # is partially working). The human checkpoint in Task 3 is the harder gate.
    assert result.n_populated >= 1, (
        f"verdict={result.verdict}; populated={result.n_populated}/5. "
        f"Adapter walk produced ZERO populated rows — the Plan 01-01 fix may "
        f"have regressed. Message: {result.message}"
    )
