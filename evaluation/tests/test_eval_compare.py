"""Non-live unit tests for evaluation/harness/compare.py.

Covers (per 131-06 success criteria):
  - aggregate_tier happy path (mean correctness; n; n_nan=0)
  - aggregate_tier with NaN (Pitfall 8 — agent_truncated; nan_breakdown)
  - aggregate_tier returns None when queries log is missing
  - _classify single-hop / multi-hop / multimodal / unknown
  - aggregate_class_rollup buckets + n_nan
  - emit_markdown deferral footer fires when no tier-4 data
  - emit_markdown deferral footer suppressed when tier-4 row populated
  - emit_markdown 9-col / 7-col headers + honest disclaimers
  - _fmt_float NaN → '—'
  - CLI --help → SystemExit(0)
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from evaluation.harness.compare import (
    aggregate_class_rollup,
    aggregate_tier,
    emit_markdown,
    _classify,
    _fmt_float,
)


def _seed_results(
    tmp_path: Path,
    tier: int,
    records: list[dict],
    metrics: list[dict],
    cost_total_usd: float,
):
    """Create the queries/costs/metrics shape compare.py expects for one tier."""
    queries_dir = tmp_path / "queries"
    costs_dir = tmp_path / "costs"
    metrics_dir = tmp_path / "metrics"
    queries_dir.mkdir(exist_ok=True)
    costs_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    timestamp = "2026-04-27T12:00:00Z"
    fname_ts = timestamp.replace(":", "_")

    queries_dir.joinpath(f"tier-{tier}-{fname_ts}.json").write_text(json.dumps({
        "tier": f"tier-{tier}",
        "timestamp": timestamp,
        "git_sha": "abc1234",
        "model": "google/gemini-2.5-flash",
        "records": records,
    }))

    # D-13 frozen schema for cost JSON
    costs_dir.joinpath(f"tier-{tier}-eval-{fname_ts}.json").write_text(json.dumps({
        "tier": f"tier-{tier}-eval",
        "timestamp": timestamp,
        "queries": [],
        "totals": {
            "usd": cost_total_usd,
            "llm_input_tokens": 100,
            "llm_output_tokens": 50,
            "embedding_tokens": 0,
        },
    }))

    metrics_dir.joinpath(f"tier-{tier}-{fname_ts}.json").write_text(json.dumps(metrics))


def test_aggregate_tier_happy(tmp_path):
    records = [
        {"question_id": "q1", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 1.5, "cost_usd_at_capture": 0.001, "error": None},
        {"question_id": "q2", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 2.5, "cost_usd_at_capture": 0.002, "error": None},
    ]
    metrics = [
        {"question_id": "q1", "faithfulness": 0.8, "answer_relevancy": 0.9,
         "context_precision": 0.7, "nan_reason": None},
        {"question_id": "q2", "faithfulness": 0.6, "answer_relevancy": 0.8,
         "context_precision": 0.5, "nan_reason": None},
    ]
    _seed_results(tmp_path, 1, records, metrics, cost_total_usd=0.003)

    row = aggregate_tier(1, tmp_path)
    assert row is not None
    assert row["n"] == 2
    assert row["n_nan"] == 0
    assert abs(row["faithfulness"] - 0.7) < 1e-9
    assert abs(row["answer_relevancy"] - 0.85) < 1e-9
    assert abs(row["context_precision"] - 0.6) < 1e-9
    assert abs(row["mean_latency_s"] - 2.0) < 1e-9
    assert row["total_cost_usd"] == 0.003
    assert abs(row["cost_per_query_usd"] - 0.0015) < 1e-9
    assert row["nan_breakdown"] == {}


def test_aggregate_tier_with_nan(tmp_path):
    """Pitfall 8 — Tier 5 agent truncation surfaces as nan_breakdown['agent_truncated']: 1."""
    records = [
        {"question_id": "q1", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 1.0, "cost_usd_at_capture": 0.001, "error": None},
        {"question_id": "q2", "question": "?", "answer": "[truncated]", "retrieved_contexts": [],
         "latency_s": 5.0, "cost_usd_at_capture": 0.0, "error": "max_turns_exceeded"},
    ]
    metrics = [
        {"question_id": "q1", "faithfulness": 0.9, "answer_relevancy": 0.8,
         "context_precision": 0.7, "nan_reason": None},
        {"question_id": "q2", "faithfulness": None, "answer_relevancy": None,
         "context_precision": None, "nan_reason": "agent_truncated"},
    ]
    _seed_results(tmp_path, 5, records, metrics, cost_total_usd=0.001)

    row = aggregate_tier(5, tmp_path)
    assert row is not None
    assert row["n"] == 2
    assert row["n_nan"] == 1
    assert row["nan_breakdown"] == {"agent_truncated": 1}
    # numpy.nanmean over the single non-None faithfulness yields 0.9 (None values filtered)
    assert abs(row["faithfulness"] - 0.9) < 1e-9


def test_aggregate_tier_with_new_reasons(tmp_path):
    """Plan 03-02 regression: nan_breakdown buckets the NEW Phase 3 reason
    strings (json_parse_failure, empty_statements) alongside the OLD
    pre-call reasons (empty_contexts). Proves compare.py needs ZERO
    modification for HARN-05 — the existing dict iteration at compare.py:110-114
    is reason-agnostic.
    """
    records = [
        {"question_id": "q1", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 1.0, "cost_usd_at_capture": 0.001, "error": None},
        {"question_id": "q2", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 1.0, "cost_usd_at_capture": 0.001, "error": None},
        {"question_id": "q3", "question": "?", "answer": "a", "retrieved_contexts": ["c"],
         "latency_s": 1.0, "cost_usd_at_capture": 0.001, "error": None},
        {"question_id": "q4", "question": "?", "answer": "a", "retrieved_contexts": [],
         "latency_s": 1.0, "cost_usd_at_capture": 0.0, "error": None},
    ]
    metrics = [
        {"question_id": "q1", "faithfulness": 0.9, "answer_relevancy": 0.85,
         "context_precision": 0.8, "nan_reason": None},
        {"question_id": "q2", "faithfulness": None, "answer_relevancy": None,
         "context_precision": None, "nan_reason": "json_parse_failure"},
        {"question_id": "q3", "faithfulness": None, "answer_relevancy": None,
         "context_precision": None, "nan_reason": "empty_statements"},
        {"question_id": "q4", "faithfulness": None, "answer_relevancy": None,
         "context_precision": None, "nan_reason": "empty_contexts"},
    ]
    _seed_results(tmp_path, 1, records, metrics, cost_total_usd=0.003)

    row = aggregate_tier(1, tmp_path)
    assert row is not None
    assert row["n"] == 4
    assert row["n_nan"] == 3
    assert row["nan_breakdown"] == {
        "json_parse_failure": 1,
        "empty_statements": 1,
        "empty_contexts": 1,
    }
    # Only q1 contributes to faithfulness mean
    assert abs(row["faithfulness"] - 0.9) < 1e-9

    # WARNING 2 closure: ALSO exercise the emit_markdown rendering path so the
    # Wave 0 gap test_emit_markdown_with_new_reasons is covered. compare.py
    # iterates SUPPORTED_TIERS zip with tier_rows, so the seeded tier-1 row
    # must land at index 0 to be rendered.
    md = emit_markdown(
        [row, None, None, None, None], [], "j", "e", [], tier_4_present=False,
    )
    # New reasons appear in the footer
    assert "json_parse_failure" in md
    assert "empty_statements" in md
    # OLD reason still renders (regression — compare.py:276 is reason-agnostic)
    assert "empty_contexts" in md
    # Per-tier breakdown line sorts reasons alphabetically (compare.py:276
    # `sorted(row["nan_breakdown"].items())`); assert the joined-counts line
    # for the seeded tier-1 row contains all three counts together.
    expected_line = "1 empty_contexts, 1 empty_statements, 1 json_parse_failure"
    assert expected_line in md, (
        f"Expected joined breakdown line {expected_line!r} in markdown footer; "
        f"actual footer:\n{md.split(chr(10) + chr(10) + chr(42) + chr(42) + 'NaN breakdown', 1)[-1][:500]}"
    )


def test_aggregate_tier_missing_returns_none(tmp_path):
    """Missing queries log → aggregate_tier returns None (sentinel for placeholder row)."""
    (tmp_path / "queries").mkdir()
    (tmp_path / "costs").mkdir()
    (tmp_path / "metrics").mkdir()
    assert aggregate_tier(1, tmp_path) is None


def test_classify():
    qa_index = {
        "single-hop-001": {"hop_count_tag": "single-hop", "modality_tag": "text"},
        "multi-hop-001": {"hop_count_tag": "multi-hop", "modality_tag": "text"},
        "multimodal-001": {"hop_count_tag": "single-hop", "modality_tag": "multimodal"},
    }
    assert _classify("single-hop-001", qa_index) == "single-hop"
    assert _classify("multi-hop-001", qa_index) == "multi-hop"
    # modality_tag="multimodal" overrides hop_count_tag (Pattern 9)
    assert _classify("multimodal-001", qa_index) == "multimodal"
    assert _classify("unknown-id", qa_index) == "unknown"


def test_aggregate_class_rollup():
    qa_index = {
        "q1": {"hop_count_tag": "single-hop", "modality_tag": "text"},
        "q2": {"hop_count_tag": "multi-hop", "modality_tag": "text"},
        "q3": {"hop_count_tag": "single-hop", "modality_tag": "multimodal"},
    }
    tier_data = [{
        "tier": 1, "tier_label": "tier-1",
        "records": [
            {"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"},
        ],
        "metrics_by_qid": {
            "q1": {"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7,
                   "nan_reason": None},
            "q2": {"faithfulness": 0.6, "answer_relevancy": 0.7, "context_precision": 0.5,
                   "nan_reason": None},
            "q3": {"faithfulness": None, "answer_relevancy": None, "context_precision": None,
                   "nan_reason": "empty_contexts"},
        },
    }]
    rows = aggregate_class_rollup(tier_data, qa_index)
    by_class = {(r["class"], r["tier"]): r for r in rows}
    assert ("single-hop", 1) in by_class
    assert ("multi-hop", 1) in by_class
    assert ("multimodal", 1) in by_class
    # multimodal bucket has 1 record with nan_reason set
    assert by_class[("multimodal", 1)]["n"] == 1
    assert by_class[("multimodal", 1)]["n_nan"] == 1
    # single-hop bucket: q1 only
    assert abs(by_class[("single-hop", 1)]["faithfulness"] - 0.8) < 1e-9
    assert by_class[("single-hop", 1)]["n_nan"] == 0


def test_emit_markdown_includes_tier4_deferral_footer():
    """When no tier-4 data is present, the footer must point at the user-helper script."""
    tier_rows = [None, None, None, None, None]
    md = emit_markdown(
        tier_rows, [], "google/gemini-2.5-flash",
        "openai/text-embedding-3-small", [], tier_4_present=False,
    )
    assert "Tier 4: deferred to user" in md
    assert "tier-4-multimodal/scripts/eval_capture.py" in md
    assert "MineRU" in md


def test_emit_markdown_no_deferral_when_tier4_present():
    """When tier-4 row IS populated, deferral footer is omitted."""
    fake_tier4 = {
        "tier": 4, "tier_label": "tier-4",
        "faithfulness": 0.7, "answer_relevancy": 0.8, "context_precision": 0.6,
        "mean_latency_s": 12.3, "total_cost_usd": 0.05, "cost_per_query_usd": 0.0017,
        "n": 30, "n_nan": 0, "nan_breakdown": {},
        "records": [], "metrics_by_qid": {},
        "timestamp": "2026-04-28T12:00:00Z", "git_sha": "def5678",
        "model": "google/gemini-2.5-flash",
        "queries_path": "x", "cost_path": "y", "metrics_path": "z",
    }
    tier_rows = [None, None, None, fake_tier4, None]
    md = emit_markdown(
        tier_rows, [], "google/gemini-2.5-flash",
        "openai/text-embedding-3-small", [], tier_4_present=True,
    )
    assert "Tier 4: deferred to user" not in md
    # the populated tier-4 row still appears (the table emits it via tier_label='tier-4')
    assert "tier-4" in md


def test_emit_markdown_format():
    """Header columns + honest-disclaimer phrases + cross-document equivalence note."""
    tier_rows = [None] * 5
    md = emit_markdown(tier_rows, [], "j", "e", [], tier_4_present=False)
    # 9-column tier rollup header (verbatim)
    assert "| Tier | Faithfulness | Answer Relevancy | Context Precision | Mean Latency (s) | Total Cost (USD) | Cost / Query (USD) | n | n NaN |" in md
    # 7-column per-class rollup header (verbatim)
    assert "| Class | Tier | Faithfulness | Answer Relevancy | Context Precision | n | n NaN |" in md
    # Honest-disclaimer phrases
    assert "Honest disclaimers" in md
    assert "30 questions" in md
    # Pattern 9: multi-hop ≡ cross-document for our corpus
    assert "Multi-hop ≡ cross-document" in md
    # Pitfall 9: multimodal-vs-text-only narrative
    assert "Tiers 1-3 are text-only" in md


def test_fmt_float_handles_nan():
    assert _fmt_float(float("nan"), 3) == "—"
    assert _fmt_float(0.123456, 3) == "0.123"
    assert _fmt_float(1.5, 2) == "1.50"
    assert _fmt_float(0.001234, 6) == "0.001234"


def _seed_results_with_embedder(
    tmp_path: Path,
    tier: int,
    embedder: str | None,
    embedder_source: str | None,
):
    """Seed a minimal queries+costs+metrics fixture for one tier with the
    new (embedder, embedder_source) provenance fields populated.

    Plan 06-01 Task 5 helper. Mirrors _seed_results above but takes the new
    fields explicitly. When embedder/embedder_source are None, the queries
    JSON OMITS the keys entirely (legacy-shape) so the em-dash fallback
    branch is exercised.
    """
    queries_dir = tmp_path / "queries"
    costs_dir = tmp_path / "costs"
    metrics_dir = tmp_path / "metrics"
    queries_dir.mkdir(exist_ok=True)
    costs_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    timestamp = "2026-05-06T12:00:00Z"
    fname_ts = timestamp.replace(":", "_")

    queries_payload: dict = {
        "tier": f"tier-{tier}",
        "timestamp": timestamp,
        "git_sha": "abc1234",
        "model": "google/gemini-2.5-flash",
        "records": [
            {"question_id": "q1", "question": "?", "answer": "a",
             "retrieved_contexts": ["c"], "latency_s": 1.0,
             "cost_usd_at_capture": 0.001, "error": None},
        ],
    }
    if embedder is not None:
        queries_payload["embedder"] = embedder
    if embedder_source is not None:
        queries_payload["embedder_source"] = embedder_source

    queries_dir.joinpath(f"tier-{tier}-{fname_ts}.json").write_text(
        json.dumps(queries_payload)
    )
    costs_dir.joinpath(f"tier-{tier}-eval-{fname_ts}.json").write_text(
        json.dumps({
            "tier": f"tier-{tier}-eval",
            "timestamp": timestamp,
            "queries": [],
            "totals": {"usd": 0.001, "llm_input_tokens": 100,
                       "llm_output_tokens": 50, "embedding_tokens": 0},
        })
    )
    metrics_dir.joinpath(f"tier-{tier}-{fname_ts}.json").write_text(
        json.dumps([
            {"question_id": "q1", "faithfulness": 0.9, "answer_relevancy": 0.85,
             "context_precision": 0.8, "nan_reason": None},
        ])
    )


def _run_compare(tmp_path: Path, tiers: str = "1") -> str:
    """Drive compare._run against tmp_path; return the rendered comparison.md."""
    from types import SimpleNamespace
    from evaluation.harness.compare import _run

    out = tmp_path / "comparison.md"
    args = SimpleNamespace(
        results_dir=str(tmp_path),
        tiers=tiers,
        out=str(out),
    )
    rc = _run(args)
    assert rc == 0, f"compare._run exited with rc={rc}"
    return out.read_text(encoding="utf-8")


def test_compare_emits_embedder_in_provenance_footer(tmp_path):
    """Plan 06-01 Task 5 / Success Criterion 3: comparison.md provenance
    footer contains a per-tier embedder line of the form
    `  - embedder: \\`<emb>\\` (source: \\`<src>\\`)`.
    """
    _seed_results_with_embedder(
        tmp_path, tier=1,
        embedder="openai/text-embedding-3-small",
        embedder_source="openrouter",
    )
    md = _run_compare(tmp_path, tiers="1")
    assert "  - embedder: `openai/text-embedding-3-small` (source: `openrouter`)" in md, (
        f"missing embedder line in provenance footer; markdown:\n{md}"
    )


def test_compare_emits_em_dash_for_legacy_embedder(tmp_path):
    """Plan 06-01 Task 5 / D-BACKCOMPAT / Pitfall 6: legacy QueryLog JSONs
    (no embedder fields) render the embedder line with em-dash placeholders;
    no traceback / KeyError.
    """
    _seed_results_with_embedder(
        tmp_path, tier=1,
        embedder=None,
        embedder_source=None,
    )
    md = _run_compare(tmp_path, tiers="1")
    assert "  - embedder: `—` (source: `—`)" in md, (
        f"em-dash legacy fallback missing; markdown:\n{md}"
    )


def test_compare_emits_embedder_by_tier_table(tmp_path):
    """Plan 06-01 Task 5 / D-Q1 / D-ROADMAP-OVERRIDE: comparison.md
    contains a dedicated 'Embedder by tier' table block above the NaN
    breakdown. Tier 5's row matches Tier 1's (NOT a hosted vector store);
    Tier 2's 'Managed' column reads 'yes' (Google File Search managed).
    """
    truth = {
        1: ("openai/text-embedding-3-small", "openrouter"),
        2: ("gemini-embedding-001", "google-managed"),
        3: ("openai/text-embedding-3-small", "openrouter"),
        4: ("openai/text-embedding-3-small", "openrouter"),
        5: ("openai/text-embedding-3-small", "openrouter"),
    }
    for tier, (emb, src) in truth.items():
        _seed_results_with_embedder(tmp_path, tier=tier, embedder=emb, embedder_source=src)

    md = _run_compare(tmp_path, tiers="1,2,3,4,5")

    # 1. Heading present.
    assert "**Embedder by tier:**" in md, f"missing 'Embedder by tier' heading"

    # 2. Tier 2 row reads 'yes' for Managed (only google-managed source).
    assert "| tier-2 | gemini-embedding-001 | google-managed | yes |" in md, (
        f"tier-2 managed=yes row missing; markdown excerpt:\n{md[-2000:]}"
    )

    # 3. Tier 5 row reads identical to Tier 1's (D-ROADMAP-OVERRIDE).
    tier5_row = "| tier-5 | openai/text-embedding-3-small | openrouter | no |"
    assert tier5_row in md, (
        f"tier-5 row mismatch (D-ROADMAP-OVERRIDE); expected {tier5_row!r}; "
        f"markdown excerpt:\n{md[-2000:]}"
    )

    # 4. Tier 1 row (sanity).
    assert "| tier-1 | openai/text-embedding-3-small | openrouter | no |" in md


def test_cli_help_exits_zero():
    """argparse --help → SystemExit(0); flag list introspectable without invoking main()."""
    from evaluation.harness import compare as compare_mod
    parser = compare_mod.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
