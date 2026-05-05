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


def test_build_judge_passes_max_tokens(monkeypatch):
    """Gap closure (Plan 02-04): _build_judge must pass max_tokens=8192 to
    llm_factory so RAGAS faithfulness can extract atomic statements from
    long Tier 4 hybrid-mode answers without truncation. Default RAGAS
    max_tokens=1024 caused 4/5 NaN on Plan 02-03 smoke."""
    from evaluation.harness import score

    captured = {}
    def fake_llm_factory(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        class _Stub:
            pass
        return _Stub()
    def fake_emb_factory(*args, **kwargs):
        class _Emb:
            def embed_text(self, *a, **k): return [0.0]
            def embed_texts(self, *a, **k): return [[0.0]]
            async def aembed_text(self, *a, **k): return [0.0]
            async def aembed_texts(self, *a, **k): return [[0.0]]
        return _Emb()

    monkeypatch.setattr("ragas.llms.llm_factory", fake_llm_factory)
    # Patch BOTH possible embedding_factory import paths used by score._build_judge:
    try:
        monkeypatch.setattr("ragas.embeddings.base.embedding_factory", fake_emb_factory)
    except (AttributeError, ImportError):
        pass
    try:
        monkeypatch.setattr("ragas.embeddings.embedding_factory", fake_emb_factory)
    except (AttributeError, ImportError):
        pass

    score._build_judge("openrouter/google/gemini-2.5-flash",
                      "openrouter/openai/text-embedding-3-small")

    assert "max_tokens" in captured["kwargs"], (
        f"_build_judge did not pass max_tokens to llm_factory; "
        f"got kwargs={list(captured['kwargs'].keys())}. "
        f"Plan 02-03 smoke FAILed because RAGAS default 1024 truncates long answers."
    )
    assert captured["kwargs"]["max_tokens"] >= 8192, (
        f"max_tokens={captured['kwargs']['max_tokens']} is below the 8192 floor "
        f"required to fit RAGAS faithfulness statement-list output for Tier 4 hybrid answers."
    )


def test_build_judge_max_tokens_is_named_constant():
    """The max_tokens value should live in a module-level constant so it's
    greppable (`grep JUDGE_MAX_TOKENS evaluation/harness/score.py`) and
    tweakable without editing the function body."""
    from evaluation.harness import score
    assert hasattr(score, "JUDGE_MAX_TOKENS"), (
        "score.py must expose JUDGE_MAX_TOKENS as a module-level constant"
    )
    assert score.JUDGE_MAX_TOKENS >= 8192


# ----------------------------------------------------------------------
# Phase 3 Plan 03-01 — NaNReasonTracer + _classify_post_evaluate_nan
# See .planning/phases/03-nan-reason-instrumentation/03-01-PLAN.md
# ----------------------------------------------------------------------


def test_nan_reason_smoke_import():
    """Smoke import: NaNReasonTracer + _classify_post_evaluate_nan must be
    importable from evaluation.harness.score (Plan 03-01 surface)."""
    from evaluation.harness.score import (  # noqa: F401
        NaNReasonTracer,
        _classify_post_evaluate_nan,
    )


# --- NaNReasonTracer tests (A-F) ---


def test_nan_reason_tracer_row_only_no_metric_no_capture():
    """Test A: ROW chain start + ROW-level on_chain_error → errors empty
    (no metric_name attached at ROW level, EVALUATION-level errors are silently ignored)."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 0},
    )
    tracer.on_chain_error(RuntimeError("row-level boom"), run_id=row_rid)
    assert tracer.errors == {}


def test_nan_reason_tracer_captures_metric_exception_parser():
    """Test B: ROW + METRIC chain, METRIC fires on_chain_error(RagasOutputParserException) →
    errors[(0, "faithfulness")] == "RagasOutputParserException"."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    met_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 0},
    )
    tracer.on_chain_start(
        {"name": "faithfulness-0"}, {}, run_id=met_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    # Synthetic exception class — type-name is what matters.
    class RagasOutputParserException(Exception):
        pass
    tracer.on_chain_error(RagasOutputParserException("bad json"), run_id=met_rid)
    assert tracer.errors == {(0, "faithfulness"): "RagasOutputParserException"}


def test_nan_reason_tracer_captures_metric_exception_llm_did_not_finish():
    """Test C: METRIC fires on_chain_error(LLMDidNotFinishException) →
    errors[(0, "faithfulness")] == "LLMDidNotFinishException"."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    met_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 0},
    )
    tracer.on_chain_start(
        {"name": "faithfulness-0"}, {}, run_id=met_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    class LLMDidNotFinishException(Exception):
        pass
    tracer.on_chain_error(LLMDidNotFinishException("max tokens"), run_id=met_rid)
    assert tracer.errors == {(0, "faithfulness"): "LLMDidNotFinishException"}


def test_nan_reason_tracer_ragas_prompt_inherits_via_parent_walk():
    """Test D: ROW(row_index=2) → METRIC(name="answer_relevancy-2") → RAGAS_PROMPT;
    on_chain_error on PROMPT chain propagates row_idx=2, metric="answer_relevancy"
    via parent walk (RAGAS_PROMPT inherits from METRIC parent)."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    met_rid = uuid.uuid4()
    prompt_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 2},
    )
    tracer.on_chain_start(
        {"name": "answer_relevancy-2"}, {}, run_id=met_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    tracer.on_chain_start(
        {}, {}, run_id=prompt_rid,
        parent_run_id=met_rid,
        metadata={"type": ChainType.RAGAS_PROMPT},
    )
    class RagasOutputParserException(Exception):
        pass
    tracer.on_chain_error(RagasOutputParserException("bad json"), run_id=prompt_rid)
    assert tracer.errors == {(2, "answer_relevancy"): "RagasOutputParserException"}


def test_nan_reason_tracer_two_metrics_same_row():
    """Test E: Two metrics on same row both error → errors holds both
    ("faithfulness", "answer_relevancy") keys for row_idx=0."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    faith_rid = uuid.uuid4()
    ar_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 0},
    )
    tracer.on_chain_start(
        {"name": "faithfulness-0"}, {}, run_id=faith_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    tracer.on_chain_start(
        {"name": "answer_relevancy-0"}, {}, run_id=ar_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    class RagasOutputParserException(Exception):
        pass
    class LLMDidNotFinishException(Exception):
        pass
    tracer.on_chain_error(RagasOutputParserException("bad"), run_id=faith_rid)
    tracer.on_chain_error(LLMDidNotFinishException("max"), run_id=ar_rid)
    assert tracer.errors == {
        (0, "faithfulness"): "RagasOutputParserException",
        (0, "answer_relevancy"): "LLMDidNotFinishException",
    }


def test_nan_reason_tracer_idempotence_first_wins():
    """Test F: Idempotence — fire on_chain_error twice on same (row, metric);
    errors holds the FIRST exception type (parent re-raises don't overwrite leaf)."""
    import uuid
    from ragas.callbacks import ChainType
    from evaluation.harness.score import NaNReasonTracer

    tracer = NaNReasonTracer()
    row_rid = uuid.uuid4()
    met_rid = uuid.uuid4()
    tracer.on_chain_start(
        {}, {}, run_id=row_rid,
        metadata={"type": ChainType.ROW, "row_index": 0},
    )
    tracer.on_chain_start(
        {"name": "faithfulness-0"}, {}, run_id=met_rid,
        parent_run_id=row_rid,
        metadata={"type": ChainType.METRIC},
    )
    class RagasOutputParserException(Exception):
        pass
    class LLMDidNotFinishException(Exception):
        pass
    tracer.on_chain_error(RagasOutputParserException("first"), run_id=met_rid)
    tracer.on_chain_error(LLMDidNotFinishException("second"), run_id=met_rid)
    # First wins — leaf-most exception preserved, parent re-raise does NOT overwrite.
    assert tracer.errors == {(0, "faithfulness"): "RagasOutputParserException"}


# --- _classify_post_evaluate_nan tests (G-N) ---


def test_classify_non_nan_returns_none():
    """Test G: metric_value is not None → returns None (defensive: non-NaN never gets a reason)."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    assert _classify_post_evaluate_nan(0, "faithfulness", 0.85, NaNReasonTracer()) is None


def test_classify_captured_ragas_output_parser_exception():
    """Test H: tracer captured RagasOutputParserException for (0, "faithfulness"),
    value=None → "json_parse_failure"."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    tracer = NaNReasonTracer()
    tracer.errors[(0, "faithfulness")] = "RagasOutputParserException"
    assert _classify_post_evaluate_nan(0, "faithfulness", None, tracer) == "json_parse_failure"


def test_classify_captured_llm_did_not_finish_exception():
    """Test I: tracer captured LLMDidNotFinishException, value=None → "llm_did_not_finish"."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    tracer = NaNReasonTracer()
    tracer.errors[(0, "faithfulness")] = "LLMDidNotFinishException"
    assert _classify_post_evaluate_nan(0, "faithfulness", None, tracer) == "llm_did_not_finish"


def test_classify_empty_tracer_faithfulness_empty_statements():
    """Test J: empty tracer, faithfulness value=None → "empty_statements"
    (ragas/metrics/_faithfulness.py:210-211 returns np.nan when statements=[])."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    assert _classify_post_evaluate_nan(0, "faithfulness", None, NaNReasonTracer()) == "empty_statements"


def test_classify_empty_tracer_answer_relevancy_empty_questions():
    """Test K: empty tracer, answer_relevancy value=None → "empty_questions"
    (ragas/metrics/_answer_relevance.py:120-124 returns np.nan when all gen_questions='')."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    assert _classify_post_evaluate_nan(0, "answer_relevancy", None, NaNReasonTracer()) == "empty_questions"


def test_classify_empty_tracer_context_precision_invalid_verdicts():
    """Test L: empty tracer, context_precision value=None → "invalid_verdicts"
    (ragas/metrics/_context_precision.py:116 returns np.nan from _calculate_average_precision)."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    assert _classify_post_evaluate_nan(0, "context_precision", None, NaNReasonTracer()) == "invalid_verdicts"


def test_classify_unknown_nan_emits_warning(caplog):
    """Test M: empty tracer, unknown metric "mystery_metric" value=None → "unknown_nan"
    AND a logging.warning fires (Pitfall 7 of RESEARCH.md — never silently drop NaN)."""
    import logging
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    with caplog.at_level(logging.WARNING):
        result = _classify_post_evaluate_nan(0, "mystery_metric", None, NaNReasonTracer())
    assert result == "unknown_nan"
    assert any("unknown_nan" in r.message for r in caplog.records)


def test_classify_captured_unknown_exception_falls_to_semantic():
    """Test N: tracer captured an exception type the classifier doesn't recognize
    (e.g. "TimeoutError"); for faithfulness this falls through to per-metric semantic
    path → "empty_statements" (NOT "unknown_nan" — captured-but-unknown-type still
    falls to semantic mapping for known metrics)."""
    from evaluation.harness.score import _classify_post_evaluate_nan, NaNReasonTracer
    tracer = NaNReasonTracer()
    tracer.errors[(0, "faithfulness")] = "TimeoutError"
    assert _classify_post_evaluate_nan(0, "faithfulness", None, tracer) == "empty_statements"
