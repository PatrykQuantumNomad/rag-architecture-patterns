"""Phase 8 / Plan 08-01 — offline tests for evaluation.harness.multi_judge_spotcheck.

This file is intentionally split into two logical sections per the TDD trail:

    Section A (Task 0 — pre-flight): a single offline assertion that the
    Phase 7 sweep_sha=`75f6f1b` source captures actually contain the 5
    wanted question IDs in tiers 1, 4, 5. Runs WITHOUT importing the
    module-under-test so that Task 1's RED commit (which DOES import
    multi_judge_spotcheck before it exists) cannot break the pre-flight.

    Section B (Task 1 — RED) is APPENDED in a follow-up commit.

Test-name cross-reference (per VALIDATION.md WARNING #7 / 08-01 Task 1
behavior table). Plan-level test names (left) are authoritative; the
canonical names in `08-VALIDATION.md` (right) are aliases:

    test_signed_delta_with_none                          ← test_signed_delta_with_none
    test_filter_records_deterministic_order              ← (helper-level)
    test_filter_records_missing_id_aborts                ← (helper-level)
    test_read_primary_metrics_pins_to_ts                 ← test_source_sha_pinned (TS half)
    test_read_source_sha_returns_log_git_sha_not_head    ← test_source_sha_pinned (SHA half)
    test_estimate_cost_fallback                          ← (cost block of test_15_cells)
    test_amain_writes_spotcheck_json                     ← test_writes_spotcheck_json + test_cell_schema + test_secondary_provenance + test_15_cells
    test_amain_writes_cost_ledger_to_explicit_dest_dir   ← test_cost_dir_isolation
    test_amain_aborts_on_missing_id                      ← (no canonical alias)
    test_dual_sha_provenance                             ← test_source_sha_pinned (SHA half)
    test_phase_7_captures_have_all_5_ids (Task 0)        ← (pre-flight only — WARNING #5 closure)
    test_module_imports                                  ← (locks D-1, D-2, D-3 module constants)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# D-2 lock (planner verification 2026-05-07 — IDs ARE present per spec):
WANTED_IDS = {
    "single-hop-001",
    "single-hop-002",
    "multi-hop-001",
    "multi-hop-002",
    "multimodal-001",
}
# D-1 lock — one tier per architecture family:
PHASE_7_TIERS = (1, 4, 5)
# Phase 7 sweep invariant:
PHASE_7_SWEEP_TS = "2026-05-07T10:59:10Z"
PHASE_7_TS_FILENAME = "2026-05-07T10_59_10Z"
PHASE_7_SWEEP_SHA = "75f6f1b"


def test_phase_7_captures_have_all_5_ids(capsys):
    """Pre-flight: assert all 5 wanted IDs are present in tiers 1, 4, 5 at sweep_sha=75f6f1b.

    Closes checker WARNING #5 — produces evidence (3 stdout lines, one per tier)
    that D-2 holds BEFORE any other test in this file runs. Skips cleanly on a
    fresh clone if Phase 7 captures aren't on disk.
    """
    queries_dir = _REPO_ROOT / "evaluation" / "results" / "queries"
    paths = {
        tier: queries_dir / f"tier-{tier}-{PHASE_7_TS_FILENAME}.json"
        for tier in PHASE_7_TIERS
    }
    missing_files = [p for p in paths.values() if not p.exists()]
    if missing_files:
        pytest.skip(
            "Phase 7 source capture missing — run /gsd-execute-phase 7 first. "
            f"Missing: {missing_files}"
        )

    failures: list[str] = []
    for tier in PHASE_7_TIERS:
        data = json.loads(paths[tier].read_text(encoding="utf-8"))
        ids_present = {r["question_id"] for r in data["records"]}
        missing_ids = WANTED_IDS - ids_present
        if missing_ids:
            failures.append(
                f"tier-{tier}: missing IDs {sorted(missing_ids)}"
            )
        if data["git_sha"] != PHASE_7_SWEEP_SHA:
            failures.append(
                f"tier-{tier}: git_sha={data['git_sha']!r} (expected {PHASE_7_SWEEP_SHA!r})"
            )
        # Evidence print (visible under -v / -s)
        print(
            f"[Pre-flight] tier-{tier}: "
            f"{len(WANTED_IDS - missing_ids)}/5 wanted IDs present "
            f"at sweep_sha={data['git_sha']}"
        )

    assert not failures, (
        "Phase 7 capture pre-flight failed:\n  " + "\n  ".join(failures)
    )


# ---------------------------------------------------------------------------
# Section B (Task 1 — RED) — these tests import from
# evaluation.harness.multi_judge_spotcheck which does NOT exist yet.
# Collection MUST raise ImportError / ModuleNotFoundError on every test below
# until Task 2 GREEN lands the impl.
# ---------------------------------------------------------------------------

from evaluation.harness import multi_judge_spotcheck as mjs  # noqa: E402
from evaluation.harness.multi_judge_spotcheck import (  # noqa: E402
    DEFAULT_TIERS,
    SECONDARY_JUDGE_DEFAULT,
    WANTED_IDS as MODULE_WANTED_IDS,
    _estimate_cost_fallback,
    _filter_records,
    _read_primary_metrics,
    _read_source_sha,
    _signed_delta,
    amain,
    build_parser,
)
from evaluation.harness.records import EvalRecord, QueryLog, ScoreRecord  # noqa: E402


# Helpers -------------------------------------------------------------------

WANTED_TUPLE = (
    "single-hop-001",
    "single-hop-002",
    "multi-hop-001",
    "multi-hop-002",
    "multimodal-001",
)


def _make_eval_record(qid: str, *, contexts: list[str] | None = None) -> EvalRecord:
    return EvalRecord(
        question_id=qid,
        question=f"Q-{qid}?",
        answer=f"A-{qid}.",
        retrieved_contexts=contexts if contexts is not None else [f"ctx-{qid}"],
        latency_s=0.5,
        cost_usd_at_capture=0.0001,
        error=None,
    )


def _make_query_log(
    tier_str: str, qids: list[str], *, git_sha: str = "75f6f1b",
    timestamp: str = "2026-05-07T10:59:10Z",
) -> QueryLog:
    return QueryLog(
        tier=tier_str,
        timestamp=timestamp,
        git_sha=git_sha,
        model="google/gemini-2.5-flash",
        embedder="openai/text-embedding-3-small",
        embedder_source="openrouter",
        records=[_make_eval_record(q) for q in qids],
    )


def _write_query_log_file(path: Path, log: QueryLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json(indent=2), encoding="utf-8")


def _write_metrics_file(path: Path, qids: list[str], *,
                        f_val: float = 0.50,
                        ar_val: float = 0.60,
                        cp_val: float = 0.70) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "question_id": q,
            "faithfulness": f_val,
            "answer_relevancy": ar_val,
            "context_precision": cp_val,
            "nan_reason": None,
        }
        for q in qids
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_args(tmp_path: Path, *, source_ts: str = "2026-05-07T10:59:10Z",
                tiers: tuple[int, ...] = (1, 4, 5),
                judge: str = "openrouter/anthropic/claude-haiku-4.5",
                question_ids: str = "",
                judge_emb: str = "openrouter/openai/text-embedding-3-small"):
    parser = build_parser()
    return parser.parse_args([
        "--source-ts", source_ts,
        "--tiers", ",".join(str(t) for t in tiers),
        "--judge", judge,
        "--judge-emb", judge_emb,
        "--question-ids", question_ids,
        "--results-dir", str(tmp_path),
        "--yes",
    ])


def _stub_score_query_log_factory(secondary_offsets: dict[str, float] | None = None,
                                  usage: dict | None = None):
    """Return a stub `score_query_log` async fn returning deterministic ScoreRecords.

    `secondary_offsets[qid]` is a float added to the (faithfulness, answer_relevancy,
    context_precision) baseline of (0.55, 0.65, 0.75) per cell — gives a non-zero
    delta vs. the metrics file values (0.50, 0.60, 0.70).
    """
    secondary_offsets = secondary_offsets or {}
    usage = usage or {"input_tokens": 0, "output_tokens": 0, "n_scored": 0}

    async def _stub(log, qa_index, judge_llm, judge_emb, batch_size=10,
                    raise_exceptions=False):
        scores = []
        for r in log.records:
            off = secondary_offsets.get(r.question_id, 0.0)
            scores.append(
                ScoreRecord(
                    question_id=r.question_id,
                    faithfulness=0.55 + off,
                    answer_relevancy=0.65 + off,
                    context_precision=0.75 + off,
                    nan_reason=None,
                )
            )
        usage_out = dict(usage)
        usage_out["n_scored"] = len(log.records)
        return scores, usage_out

    return _stub


def _stub_build_judge(*args, **kwargs):
    """Return a (None, None) tuple — no real LLM/embedder construction in tests."""
    return (None, None)


def _stub_load_golden_qa():
    """Return a stub golden QA list — all wanted IDs with placeholder reference."""
    return [
        {"id": q, "expected_answer": f"ref-{q}"}
        for q in WANTED_TUPLE
    ]


# Tests ---------------------------------------------------------------------


def test_module_imports():
    """Locks D-1, D-2, D-3 module constants."""
    assert MODULE_WANTED_IDS == WANTED_TUPLE
    assert DEFAULT_TIERS == (1, 4, 5)
    assert SECONDARY_JUDGE_DEFAULT == "openrouter/anthropic/claude-haiku-4.5"


def test_signed_delta_with_none():
    """Pitfall 5 / Pattern 4 — None propagates symmetrically."""
    assert _signed_delta(0.75, 0.82) == pytest.approx(-0.07)
    assert _signed_delta(None, 0.5) is None
    assert _signed_delta(0.5, None) is None
    assert _signed_delta(None, None) is None


def test_filter_records_deterministic_order():
    """Pitfall 7 — _filter_records iterates WANTED_IDS, NOT source list order."""
    reverse = list(reversed(WANTED_TUPLE))
    log = _make_query_log("tier-1", reverse)
    out = _filter_records(log, WANTED_TUPLE)
    assert [r.question_id for r in out.records] == list(WANTED_TUPLE)


def test_filter_records_missing_id_aborts():
    """Pitfall 4 — when a wanted ID is missing, returns log with len < 5."""
    log = _make_query_log("tier-1", list(WANTED_TUPLE[:4]))  # only 4 of 5
    out = _filter_records(log, WANTED_TUPLE)
    assert len(out.records) == 4
    assert "multimodal-001" not in {r.question_id for r in out.records}


def test_read_primary_metrics_pins_to_ts(tmp_path):
    """Pattern 3 / Pitfall 3 — pin to ts, NOT mtime."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pinned_path = metrics_dir / "tier-1-2026-05-07T10_59_10Z.json"
    newer_path = metrics_dir / "tier-1-2026-05-07T11_00_00Z.json"
    _write_metrics_file(pinned_path, list(WANTED_TUPLE), f_val=0.10)
    _write_metrics_file(newer_path, list(WANTED_TUPLE), f_val=0.99)
    # Touch newer to ensure mtime is later
    import os
    import time
    time.sleep(0.01)
    os.utime(newer_path, None)

    out = _read_primary_metrics(metrics_dir, 1, "2026-05-07T10:59:10Z")
    # Must load PINNED file (f=0.10), NOT mtime-newest (f=0.99)
    assert out["single-hop-001"]["faithfulness"] == pytest.approx(0.10)


def test_read_source_sha_returns_log_git_sha_not_head(tmp_path, monkeypatch):
    """BLOCKER #3 / Pitfall 3 — helper reads src_log.git_sha, NEVER _git_sha()."""
    queries_dir = tmp_path / "queries"
    log = _make_query_log("tier-1", list(WANTED_TUPLE), git_sha="75f6f1b")
    _write_query_log_file(
        queries_dir / "tier-1-2026-05-07T10_59_10Z.json", log,
    )
    # Simulate HEAD has advanced past Phase 7 — _git_sha() returns DEADBEEF.
    monkeypatch.setattr(mjs, "_git_sha", lambda: "DEADBEEF")

    sha = _read_source_sha(queries_dir, [1], "2026-05-07T10:59:10Z")
    assert sha == "75f6f1b"
    assert sha != "DEADBEEF"

    # Repeat for 3 tiers — consistency check.
    for t in (4, 5):
        _write_query_log_file(
            queries_dir / f"tier-{t}-2026-05-07T10_59_10Z.json",
            _make_query_log(f"tier-{t}", list(WANTED_TUPLE), git_sha="75f6f1b"),
        )
    sha_multi = _read_source_sha(queries_dir, [1, 4, 5], "2026-05-07T10:59:10Z")
    assert sha_multi == "75f6f1b"


def test_estimate_cost_fallback():
    """RESEARCH A6 / D-8 — estimator triggers when usage is zero, passes-through otherwise."""
    # Estimated path
    est = _estimate_cost_fallback(
        {"input_tokens": 0, "output_tokens": 0, "n_scored": 5},
        "anthropic/claude-haiku-4.5", n_scored=5,
    )
    assert est["estimated"] is True
    assert est["input_tokens"] > 0
    assert est["output_tokens"] > 0
    assert est["usd"] > 0.0
    assert est["usd"] < 0.05

    # Real-usage path
    real = _estimate_cost_fallback(
        {"input_tokens": 1000, "output_tokens": 200, "n_scored": 5},
        "anthropic/claude-haiku-4.5", n_scored=5,
    )
    assert real["estimated"] is False
    assert real["input_tokens"] == 1000
    assert real["output_tokens"] == 200
    assert real["usd"] > 0.0


def _setup_amain_inputs(tmp_path: Path, *, missing_qid_for_tier: int | None = None,
                        source_git_sha: str = "75f6f1b"):
    """Wire up a complete tmp_path with queries+metrics for tiers 1, 4, 5."""
    queries_dir = tmp_path / "queries"
    metrics_dir = tmp_path / "metrics"
    for tier in (1, 4, 5):
        qids = list(WANTED_TUPLE)
        if missing_qid_for_tier == tier:
            qids = qids[:4]  # drop multimodal-001
        log = _make_query_log(f"tier-{tier}", qids, git_sha=source_git_sha)
        _write_query_log_file(
            queries_dir / f"tier-{tier}-2026-05-07T10_59_10Z.json", log,
        )
        _write_metrics_file(
            metrics_dir / f"tier-{tier}-2026-05-07T10_59_10Z.json",
            list(WANTED_TUPLE),  # metrics always have all 5 (delta will skip missing src)
        )
    return queries_dir, metrics_dir


def _patch_amain_internals(monkeypatch, *, head_sha: str = "75f6f1b",
                           usage: dict | None = None):
    """Patch the module's external dependencies for offline integration tests."""
    monkeypatch.setattr(mjs, "_git_sha", lambda: head_sha)
    monkeypatch.setattr(mjs, "_build_judge", _stub_build_judge)
    monkeypatch.setattr(mjs, "_load_golden_qa", _stub_load_golden_qa)
    monkeypatch.setattr(
        mjs, "score_query_log",
        _stub_score_query_log_factory(usage=usage),
    )


def test_amain_writes_spotcheck_json(tmp_path, monkeypatch):
    """SC-1 + SC-2 + SC-4 — integration: writes JSON with 15 cells, dual-SHA, max_tokens=8192."""
    import asyncio

    from rich.console import Console

    _setup_amain_inputs(tmp_path)
    _patch_amain_internals(monkeypatch)

    args = _build_args(tmp_path)
    rc = asyncio.run(amain(args, Console()))
    assert rc == 0

    # Locate output JSON
    metrics_dir = tmp_path / "metrics"
    matches = sorted(metrics_dir.glob("multi-judge-spotcheck-*.json"))
    assert len(matches) == 1
    payload = json.loads(matches[0].read_text())

    assert payload["$schema_version"] == "1.0"
    assert payload["secondary_judge"]["max_tokens"] == 8192
    assert payload["secondary_judge"]["model_slug"] == "openrouter/anthropic/claude-haiku-4.5"
    assert payload["secondary_judge"]["model"] == "anthropic/claude-haiku-4.5"
    assert payload["source_capture_git_sha"] == "75f6f1b"
    # Schema check
    assert len(payload["cells"]) == 15
    for cell in payload["cells"]:
        assert set(cell.keys()) >= {"question_id", "tier", "primary", "secondary", "delta"}
        for block_key in ("primary", "secondary", "delta"):
            block = cell[block_key]
            assert "faithfulness" in block
            assert "answer_relevancy" in block
            assert "context_precision" in block


def test_amain_writes_cost_ledger_to_explicit_dest_dir(tmp_path, monkeypatch):
    """Pitfall 2 / D-7 — cost JSON lands at tmp_path/costs/, NEVER evaluation/results/costs/."""
    import asyncio

    from rich.console import Console

    _setup_amain_inputs(tmp_path)
    _patch_amain_internals(
        monkeypatch,
        usage={"input_tokens": 100, "output_tokens": 50, "n_scored": 5},
    )

    # Snapshot pre-existing production-default cost dir state
    prod_costs = _REPO_ROOT / "evaluation" / "results" / "costs"
    pre_existing = (
        set(prod_costs.glob("multi-judge-spotcheck-*.json"))
        if prod_costs.exists()
        else set()
    )

    args = _build_args(tmp_path)
    rc = asyncio.run(amain(args, Console()))
    assert rc == 0

    # Cost JSON in tmp_path
    cost_dir = tmp_path / "costs"
    cost_files = sorted(cost_dir.glob("multi-judge-spotcheck-*.json"))
    assert len(cost_files) == 1, f"Expected 1 cost file in {cost_dir}, got {cost_files}"
    cost_payload = json.loads(cost_files[0].read_text())
    # D-13 schema
    assert "tier" in cost_payload
    assert "timestamp" in cost_payload
    assert "queries" in cost_payload
    assert "totals" in cost_payload

    # Production default UNTOUCHED for new spotcheck files
    if prod_costs.exists():
        post = set(prod_costs.glob("multi-judge-spotcheck-*.json"))
        assert post == pre_existing, (
            f"Cost ledger leaked into production default: {post - pre_existing}"
        )


def test_amain_aborts_on_missing_id(tmp_path, monkeypatch):
    """Pitfall 4 — when source log is missing one wanted ID, amain returns non-zero."""
    import asyncio

    from rich.console import Console

    # Drop multimodal-001 from tier-1 source capture
    _setup_amain_inputs(tmp_path, missing_qid_for_tier=1)
    _patch_amain_internals(monkeypatch)

    args = _build_args(tmp_path)
    rc = asyncio.run(amain(args, Console()))
    assert rc != 0

    # NO spot-check JSON should be written
    matches = sorted((tmp_path / "metrics").glob("multi-judge-spotcheck-*.json"))
    assert matches == []


def test_dual_sha_provenance(tmp_path, monkeypatch, capsys):
    """Pitfall 3 / D-6 — source_capture_git_sha=src, spotcheck_run_git_sha=HEAD; mismatch warns."""
    import asyncio

    from rich.console import Console

    _setup_amain_inputs(tmp_path, source_git_sha="75f6f1b")
    _patch_amain_internals(monkeypatch, head_sha="DEADBEEF")

    args = _build_args(tmp_path)
    console = Console(force_terminal=False)
    rc = asyncio.run(amain(args, console))
    assert rc == 0

    metrics_dir = tmp_path / "metrics"
    matches = sorted(metrics_dir.glob("multi-judge-spotcheck-*.json"))
    assert len(matches) == 1
    payload = json.loads(matches[0].read_text())

    assert payload["source_capture_git_sha"] == "75f6f1b"
    assert payload["spotcheck_run_git_sha"] == "DEADBEEF"

    # Warning must mention sha mismatch (yellow style)
    out = capsys.readouterr().out
    assert "75f6f1b" in out
    assert "DEADBEEF" in out


# ---------------------------------------------------------------------------
# Section C (Plan 08-02 — live smoke backstop) — closes CAP-02 at LIVE level.
# Mirrors the Plan 05-02 / Plan 03-03 live-smoke-backstop pattern.
# Two-tier cost enforcement (BLOCKER #2 fix):
#   - SOFT ceiling $0.30 = ROADMAP SC-3 envelope (escalates as checkpoint).
#   - HARD ceiling $0.50 = runaway protection (test fails outright).
# ---------------------------------------------------------------------------

SOFT_CEILING_USD = 0.30   # ROADMAP SC-3 envelope (BLOCKER #2 — soft tier)
HARD_CEILING_USD = 0.50   # runaway protection (BLOCKER #2 — hard tier)


@pytest.mark.live
def test_live_spotcheck_under_budget(live_eval_keys_ok, tmp_path, capsys):
    """Phase 8 Plan 08-02 — live smoke backstop closing CAP-02 at LIVE level.

    Drives multi_judge_spotcheck.amain against the Phase 7 sweep_sha=75f6f1b
    capture and the real OpenRouter API (Claude Haiku 4.5 secondary judge).

    Two-tier cost enforcement (BLOCKER #2 fix):
      - SOFT ceiling $0.30 (ROADMAP SC-3 envelope): exceedance prints
        ``## CHECKPOINT REACHED — SC-3 envelope exceeded`` with breakdown;
        test fails via dedicated AssertionError that the orchestrator
        treats as a checkpoint (PASS-WITH-DEVIATION outcome path).
      - HARD ceiling $0.50: exceedance fails the test outright (runaway).

    Source-SHA pinning: produced JSON's ``source_capture_git_sha`` MUST equal
    ``75f6f1b`` (the Phase 7 sweep_sha) regardless of current HEAD —
    BLOCKER #3 fix verified end-to-end against real source captures.
    """
    import asyncio
    from rich.console import Console

    from evaluation.harness import multi_judge_spotcheck

    # Stage results dir under tmp_path so live artifacts don't pollute repo.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "queries").mkdir()
    (results_dir / "metrics").mkdir()
    (results_dir / "costs").mkdir()

    # Stage the Phase 7 capture into tmp_path so amain reads from there
    # (NOT from evaluation/results/, so prod data stays clean even on bug).
    src_repo = _REPO_ROOT
    for tier in PHASE_7_TIERS:
        for kind in ("queries", "metrics"):
            src = src_repo / "evaluation" / "results" / kind / \
                f"tier-{tier}-{PHASE_7_TS_FILENAME}.json"
            dst = results_dir / kind / f"tier-{tier}-{PHASE_7_TS_FILENAME}.json"
            dst.write_bytes(src.read_bytes())

    # Snapshot prod costs dir so we can assert no NEW files leak there
    # (Pitfall 2 / D-7 — second-time verification at LIVE level).
    prod_costs = src_repo / "evaluation" / "results" / "costs"
    pre_run_costs = (
        set(prod_costs.glob("multi-judge-spotcheck-*.json"))
        if prod_costs.exists()
        else set()
    )

    # Build args via build_parser to match the production CLI path.
    parser = multi_judge_spotcheck.build_parser()
    args = parser.parse_args([
        "--source-ts", PHASE_7_SWEEP_TS,
        "--tiers", "1,4,5",
        "--results-dir", str(results_dir),
        "--yes",
    ])

    # Invoke amain() in-process (subprocess-free; mirrors Plan 05-02 pattern).
    rc = asyncio.run(multi_judge_spotcheck.amain(args, Console()))
    assert rc == 0, f"multi_judge_spotcheck.amain exited rc={rc}"

    # 1. Spot-check JSON exists and has 15 cells (SC-1 + SC-2 + SC-4).
    spotcheck_files = list((results_dir / "metrics").glob("multi-judge-spotcheck-*.json"))
    assert len(spotcheck_files) == 1, \
        f"Expected 1 spot-check file, got {len(spotcheck_files)}: {spotcheck_files}"
    spotcheck = json.loads(spotcheck_files[0].read_text())

    # Schema and provenance assertions
    assert spotcheck["$schema_version"] == "1.0"
    assert spotcheck["source_capture_git_sha"] == PHASE_7_SWEEP_SHA, (
        f"source_capture_git_sha={spotcheck['source_capture_git_sha']!r}; "
        f"expected {PHASE_7_SWEEP_SHA!r} (Phase 7 sweep_sha — BLOCKER #3 fix invariant)"
    )
    assert spotcheck["source_capture_timestamp"] == PHASE_7_SWEEP_TS
    assert spotcheck["secondary_judge"]["model"] == "anthropic/claude-haiku-4.5"
    assert spotcheck["secondary_judge"]["model_slug"] == "openrouter/anthropic/claude-haiku-4.5"
    assert spotcheck["secondary_judge"]["max_tokens"] == 8192, (
        "secondary_judge.max_tokens must be 8192 (Plan 02-04 lesson — JUDGE_MAX_TOKENS)"
    )
    assert spotcheck["primary_judge"]["model"] == "google/gemini-2.5-flash"

    # 15-cell shape (5 IDs × 3 tiers)
    cells = spotcheck["cells"]
    assert len(cells) == 15, f"Expected 15 cells (5 IDs × 3 tiers), got {len(cells)}"

    # Per-cell: primary + secondary + delta blocks for all 3 metrics
    for c in cells:
        assert {"question_id", "tier", "primary", "secondary", "delta"} <= c.keys(), (
            f"Cell missing required keys: {c.keys()}"
        )
        for block_name in ("primary", "secondary", "delta"):
            block = c[block_name]
            for metric in ("faithfulness", "answer_relevancy", "context_precision"):
                assert metric in block, (
                    f"Missing {metric} in {block_name} of "
                    f"{c['question_id']}/{c['tier']}"
                )

    # Tier coverage check — 5 cells per tier
    by_tier: dict[str, int] = {}
    for c in cells:
        by_tier[c["tier"]] = by_tier.get(c["tier"], 0) + 1
    assert by_tier == {"tier-1": 5, "tier-4": 5, "tier-5": 5}, (
        f"Per-tier cell counts wrong: {by_tier}"
    )

    # 2. Cost ledger written to results_dir/costs/ (Pitfall 2 / D-7).
    cost_files = list((results_dir / "costs").glob("multi-judge-spotcheck-*.json"))
    assert len(cost_files) == 1, \
        f"Expected 1 cost ledger in results_dir/costs/, got {cost_files}"
    cost = json.loads(cost_files[0].read_text())

    # Determine total_usd: prefer estimator.estimated_usd when fallback fired
    # (RESEARCH A6 / D-8 — Phase 7 ragas-judge ledgers consistently show
    # total_usd=0 because LiteLLM usage parser returns 0 for non-OpenAI providers).
    raw_total_usd = float(cost.get("totals", {}).get("usd", 0.0) or 0.0)
    estimator = cost.get("estimator", {})
    estimated_usd = float(estimator.get("estimated_usd", 0.0) or 0.0)
    total_usd = max(raw_total_usd, estimated_usd)

    # 3. Two-tier cost enforcement (BLOCKER #2 fix).
    # 3a. HARD ceiling — runaway protection (test FAILS outright if breached).
    assert total_usd <= HARD_CEILING_USD, (
        f"Live spotcheck spent ${total_usd:.4f} > ${HARD_CEILING_USD:.2f} "
        f"HARD ceiling — runaway"
    )

    # 3b. SOFT ceiling — ROADMAP SC-3 envelope (escalates as checkpoint).
    if total_usd > SOFT_CEILING_USD:
        breakdown_lines = [
            "## CHECKPOINT REACHED — SC-3 envelope exceeded "
            f"(${total_usd:.4f} > ${SOFT_CEILING_USD:.2f})",
            "",
            "Cost breakdown:",
            f"  raw_total_usd:    ${raw_total_usd:.4f}",
            f"  estimated_usd:    ${estimated_usd:.4f}",
            f"  effective_total:  ${total_usd:.4f}",
            f"  SC-3 envelope:    $0.10–$0.30 (ROADMAP)",
            f"  HARD ceiling:     ${HARD_CEILING_USD:.2f} (NOT breached)",
            f"  estimated:        {estimator.get('estimated', False)}",
            "",
            "User decision required:",
            "  - 'accept'  → mark plan PASS-WITH-DEVIATION, record SC-3 deviation",
            "  - 'abort'   → mark plan FAIL, no rerun",
            "  - 'rerun'   → re-invoke live test (additional ≤ $0.50 spend authorized)",
        ]
        print("\n".join(breakdown_lines))
        raise AssertionError(
            f"## CHECKPOINT REACHED — SC-3 envelope exceeded "
            f"(${total_usd:.4f} > ${SOFT_CEILING_USD:.2f}); see captured output"
        )

    # 4. NO production-default cost path written (Pitfall 2 / D-7 — second pass).
    if prod_costs.exists():
        post_run_costs = set(prod_costs.glob("multi-judge-spotcheck-*.json"))
        new_prod_costs = post_run_costs - pre_run_costs
        assert not new_prod_costs, (
            f"Cost ledger leaked to prod path: {new_prod_costs}"
        )

    # Print summary for SUMMARY provenance (visible under -v / -s).
    print(
        f"\n[live spotcheck] total_usd=${total_usd:.6f} "
        f"raw=${raw_total_usd:.6f} estimated=${estimated_usd:.6f} "
        f"source_sha={spotcheck['source_capture_git_sha']} "
        f"n_cells={len(cells)} "
        f"by_tier={by_tier}"
    )
