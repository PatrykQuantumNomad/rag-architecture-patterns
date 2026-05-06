"""HARN-01 + HARN-02: pipeline.py composition + single-tier rerun semantics.

Tests written BEFORE evaluation/harness/pipeline.py exists (TDD RED).
Mirrors fixture patterns from test_eval_freeze.py + test_eval_run.py.

Run offline:
    uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live' -x
"""
from __future__ import annotations

import builtins
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import pytest
from rich.console import Console

from evaluation.harness import compare, freeze, run, score
# The following import will FAIL until Task 2 ships pipeline.py — that's the RED state:
from evaluation.harness import pipeline


# --- Helpers ----------------------------------------------------------------


class _SilentConsole:
    """Drop-in stand-in for rich.console.Console used in tests."""

    def print(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def rule(self, *args, **kwargs):
        pass


def _base_args(**overrides) -> Namespace:
    """Produce a Namespace with all 12 pipeline argparse fields populated.

    Override per test as needed.
    """
    defaults = dict(
        tiers="1",
        limit=None,
        smoke_question_ids=None,
        tier_4_from_cache=None,
        results_dir="evaluation/results",
        judge_model=score.JUDGE_LLM_SLUG_DEFAULT,
        judge_emb=score.JUDGE_EMB_SLUG_DEFAULT,
        batch_size=10,
        mode="hybrid",
        tier1_k=5,
        freeze=None,
        yes=True,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _patch_loaders(monkeypatch):
    """Stub _load_golden_qa so tests don't depend on disk."""
    fake_qa = [
        {"id": "q1", "question": "?", "expected_answer": "!",
         "modality_tag": "text", "hop_count_tag": "single-hop"},
    ]
    monkeypatch.setattr(run, "_load_golden_qa", lambda: fake_qa)


def _patch_sha_ts(monkeypatch, sha="deadbeef", ts="2026-05-05T12:00:00Z"):
    monkeypatch.setattr(run, "_git_sha", lambda: sha)
    monkeypatch.setattr(run, "_ts", lambda: ts)
    return sha, ts


# --- Tests ------------------------------------------------------------------


def test_build_parser_help_exits_zero():
    """Test 1: --help exits 0."""
    parser = pipeline.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_build_parser_defaults():
    """Test 2: argparse defaults match locked surface."""
    parser = pipeline.build_parser()
    args = parser.parse_args([])
    assert args.tiers == "1,2,3,4,5"
    assert args.limit is None
    assert args.smoke_question_ids is None
    assert args.tier_4_from_cache is None
    assert args.results_dir == "evaluation/results"
    assert args.judge_model == score.JUDGE_LLM_SLUG_DEFAULT
    assert args.judge_emb == score.JUDGE_EMB_SLUG_DEFAULT
    assert args.batch_size == 10
    assert args.mode == "hybrid"
    assert args.tier1_k == 5
    assert args.freeze is None
    assert args.yes is False


def test_pipeline_cost_estimate():
    """Test 3: _pipeline_cost_estimate returns (capture, judge, total) with judge factor."""
    cap, jud, tot = pipeline._pipeline_cost_estimate([1, 2, 3], 10)
    assert jud == pytest.approx(0.003 * 10 * 3)
    assert cap > 0
    assert tot == pytest.approx(cap + jud)


def test_sha_propagation(monkeypatch):
    """Test 4 (HARN-01): sweep_sha + sweep_ts thread to run.amain via override kwargs."""
    import asyncio

    sweep_sha, sweep_ts = _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    captured = {}

    async def _capture_run_args(args, console):
        captured["run_args"] = args
        return 0

    async def _no_op_score(args, console):
        return 0

    def _no_op_compare(args):
        return 0

    monkeypatch.setattr(run, "amain", _capture_run_args)
    monkeypatch.setattr(score, "amain", _no_op_score)
    monkeypatch.setattr(compare, "_run", _no_op_compare)

    args = _base_args(yes=True, freeze=None)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert captured["run_args"].git_sha_override == sweep_sha
    assert captured["run_args"].ts_override == sweep_ts


def test_full_sweep_order(monkeypatch):
    """Test 5 (HARN-01): run -> score -> compare -> freeze in order."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    order: list[str] = []

    async def fake_run_amain(args, console):
        order.append("run.amain")
        return 0

    async def fake_score_amain(args, console):
        order.append("score.amain")
        return 0

    def fake_compare_run(args):
        order.append("compare._run")
        return 0

    def fake_freeze_freeze(**kw):
        order.append("freeze.freeze")
        return Path("/tmp/x.md")

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fake_score_amain)
    monkeypatch.setattr(compare, "_run", fake_compare_run)
    monkeypatch.setattr(freeze, "freeze", fake_freeze_freeze)

    args = _base_args(tiers="1", freeze="v1.0", yes=True)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert order == ["run.amain", "score.amain", "compare._run", "freeze.freeze"]


def test_single_cost_prompt(monkeypatch):
    """Test 6 (HARN-01 SC4): cost prompt fires exactly once; inner prompts suppressed via yes=True."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    counter = {"n": 0}

    def fake_input(prompt=""):
        counter["n"] += 1
        return "y"

    monkeypatch.setattr(builtins, "input", fake_input)

    captured = {}

    async def fake_run_amain(args, console):
        captured["run_args"] = args
        return 0

    async def fake_score_amain(args, console):
        captured["score_args"] = args
        return 0

    def fake_compare_run(args):
        return 0

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fake_score_amain)
    monkeypatch.setattr(compare, "_run", fake_compare_run)

    args = _base_args(yes=False, freeze=None)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert counter["n"] == 1
    assert captured["run_args"].yes is True
    assert captured["score_args"].yes is True


def test_cost_prompt_abort(monkeypatch):
    """Test 7: 'n' to cost prompt -> exit 1, no stage invoked."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")

    async def fail_run(*a, **k):
        raise AssertionError("run.amain should not be called when user aborts cost prompt")

    async def fail_score(*a, **k):
        raise AssertionError("score.amain should not be called when user aborts")

    def fail_compare(*a, **k):
        raise AssertionError("compare._run should not be called when user aborts")

    def fail_freeze(*a, **k):
        raise AssertionError("freeze should not be called when user aborts")

    monkeypatch.setattr(run, "amain", fail_run)
    monkeypatch.setattr(score, "amain", fail_score)
    monkeypatch.setattr(compare, "_run", fail_compare)
    monkeypatch.setattr(freeze, "freeze", fail_freeze)

    args = _base_args(yes=False, freeze="v1.0")
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 1


def test_stage_exit_propagation(monkeypatch):
    """Test 8: run.amain returns 2 -> pipeline returns 2; downstream stages NOT called."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    async def fake_run_amain(args, console):
        return 2

    async def fail_score(args, console):
        raise AssertionError("score.amain should not be called when run fails")

    def fail_compare(args):
        raise AssertionError("compare._run should not be called when run fails")

    def fail_freeze(**kw):
        raise AssertionError("freeze should not be called when run fails")

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fail_score)
    monkeypatch.setattr(compare, "_run", fail_compare)
    monkeypatch.setattr(freeze, "freeze", fail_freeze)

    args = _base_args(yes=True, freeze="v1.0")
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 2


def test_freeze_passthrough(monkeypatch):
    """Test 9: --freeze v1.0 -> freeze.freeze called once with locked kwargs."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    async def ok_run(args, console):
        return 0

    async def ok_score(args, console):
        return 0

    def ok_compare(args):
        return 0

    captured: dict = {}

    def fake_freeze_fn(**kw):
        captured["call_count"] = captured.get("call_count", 0) + 1
        captured["kwargs"] = kw
        return Path("/tmp/fake-frozen.md")

    monkeypatch.setattr(run, "amain", ok_run)
    monkeypatch.setattr(score, "amain", ok_score)
    monkeypatch.setattr(compare, "_run", ok_compare)
    monkeypatch.setattr(freeze, "freeze", fake_freeze_fn)

    args = _base_args(freeze="v1.0", results_dir="/some/dir", yes=True)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert captured["call_count"] == 1
    kw = captured["kwargs"]
    assert kw["version"] == "v1.0"
    assert kw["force"] is False
    assert kw["results_dir"] == Path("/some/dir")
    assert kw["source"] is None


def test_freeze_refusal_handling(monkeypatch):
    """Test 10: freeze raises FileExistsError -> pipeline returns 2."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    async def ok_run(args, console):
        return 0

    async def ok_score(args, console):
        return 0

    def ok_compare(args):
        return 0

    def fail_freeze(**kw):
        raise FileExistsError("already frozen — bump version or pass --force")

    monkeypatch.setattr(run, "amain", ok_run)
    monkeypatch.setattr(score, "amain", ok_score)
    monkeypatch.setattr(compare, "_run", ok_compare)
    monkeypatch.setattr(freeze, "freeze", fail_freeze)

    args = _base_args(freeze="v1.0", yes=True)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 2


def test_yes_flag_consolidation(monkeypatch):
    """Test 11: --yes suppresses pipeline prompt + threads yes=True into stages."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    def fail_input(prompt=""):
        raise AssertionError("--yes should suppress prompt — input() must not be called")

    monkeypatch.setattr(builtins, "input", fail_input)

    captured: dict = {}

    async def fake_run_amain(args, console):
        captured["run_args"] = args
        return 0

    async def fake_score_amain(args, console):
        captured["score_args"] = args
        return 0

    def ok_compare(args):
        return 0

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fake_score_amain)
    monkeypatch.setattr(compare, "_run", ok_compare)

    args = _base_args(yes=True, freeze=None)
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert captured["run_args"].yes is True
    assert captured["score_args"].yes is True


def test_tier_4_from_cache_passthrough(monkeypatch):
    """Test 12 (HARN-02): --tier-4-from-cache threads through to run_args."""
    import asyncio

    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    captured: dict = {}

    async def fake_run_amain(args, console):
        captured["run_args"] = args
        return 0

    async def ok_score(args, console):
        return 0

    def ok_compare(args):
        return 0

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", ok_score)
    monkeypatch.setattr(compare, "_run", ok_compare)

    args = _base_args(
        tier_4_from_cache="/path/to/cache.json",
        tiers="4",
        yes=True,
    )
    rc = asyncio.run(pipeline.amain(args, _SilentConsole()))
    assert rc == 0
    assert captured["run_args"].tier_4_from_cache == "/path/to/cache.json"


def test_no_subprocess_calls():
    """Test 13 (Success Criterion 3): pipeline.py source contains no subprocess/os.system/os.popen."""
    src = Path(__file__).resolve().parent.parent.parent / "evaluation" / "harness" / "pipeline.py"
    text = src.read_text()
    assert "subprocess" not in text, "pipeline.py must not reference subprocess"
    assert "os.system" not in text, "pipeline.py must not reference os.system"
    assert "os.popen" not in text, "pipeline.py must not reference os.popen"


# --- Integration tests ------------------------------------------------------


def _build_results_fixture(tmp_path: Path, tiers: tuple[int, ...]) -> Path:
    """Build a tmp results_dir with pre-existing tier-{tiers} files (for HARN-02)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "comparison.md").write_text("PRIOR ROLLUP\n")
    queries_dir = results_dir / "queries"
    costs_dir = results_dir / "costs"
    metrics_dir = results_dir / "metrics"
    queries_dir.mkdir()
    costs_dir.mkdir()
    metrics_dir.mkdir()

    for tier in tiers:
        (queries_dir / f"tier-{tier}-2026-05-04T10_00_00Z.json").write_text(json.dumps({
            "tier": f"tier-{tier}",
            "git_sha": "OLDSHA",
            "timestamp": "2026-05-04T10:00:00Z",
            "model": "google/gemini-2.5-flash",
            "records": [{
                "question_id": "q1",
                "question": "?",
                "answer": "stub",
                "retrieved_contexts": ["ctx"],
                "latency_s": 1.0,
            }],
        }))
        (costs_dir / f"tier-{tier}-eval-2026-05-04T10_00_00Z.json").write_text(json.dumps({
            "totals": {"usd": 0.001},
            "queries": [],
        }))
        (metrics_dir / f"tier-{tier}-2026-05-04T10_00_00Z.json").write_text(json.dumps([
            {"question_id": "q1", "faithfulness": 0.8,
             "answer_relevancy": 0.9, "context_precision": 0.7, "nan_reason": None},
        ]))

    (costs_dir / "ragas-judge-2026-05-04T10_00_00Z.json").write_text(json.dumps({
        "queries": [{"kind": "llm", "model": "google/gemini-2.5-flash"}],
    }))

    # Older mtime — one day ago
    epoch_old = time.time() - 86400
    for p in [
        *queries_dir.glob("*.json"),
        *costs_dir.glob("*.json"),
        *metrics_dir.glob("*.json"),
        results_dir / "comparison.md",
    ]:
        os.utime(p, (epoch_old, epoch_old))

    return results_dir


def test_e2e_with_stub_stages(tmp_path, monkeypatch):
    """Test 14 (integration): end-to-end with stub stages writing marker files."""
    _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    async def fake_run_amain(args, console):
        (tmp_path / "RAN_RUN.txt").write_text("ran")
        return 0

    async def fake_score_amain(args, console):
        (tmp_path / "RAN_SCORE.txt").write_text("ran")
        return 0

    def fake_compare_run(args):
        (tmp_path / "RAN_COMPARE.txt").write_text("ran")
        return 0

    def fake_freeze_freeze(**kw):
        (tmp_path / "RAN_FREEZE.txt").write_text("ran")
        return tmp_path / "frozen-fake.md"

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fake_score_amain)
    monkeypatch.setattr(compare, "_run", fake_compare_run)
    monkeypatch.setattr(freeze, "freeze", fake_freeze_freeze)

    rc = pipeline.main([
        "--tiers", "1",
        "--freeze", "v1.0",
        "--yes",
        "--results-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "RAN_RUN.txt").exists()
    assert (tmp_path / "RAN_SCORE.txt").exists()
    assert (tmp_path / "RAN_COMPARE.txt").exists()
    assert (tmp_path / "RAN_FREEZE.txt").exists()


def test_single_tier_rerun_preserves_others(tmp_path, monkeypatch):
    """Test 15 (HARN-02 + W5): single tier-4 rerun preserves other tiers byte-identically;
    new tier-4 JSON contains the propagated SHA on disk."""
    sweep_sha, sweep_ts = _patch_sha_ts(monkeypatch)
    _patch_loaders(monkeypatch)

    # Build pre-existing fixture with tier-{1,2,3,5} (NOT 4)
    results_dir = _build_results_fixture(tmp_path, tiers=(1, 2, 3, 5))

    # Capture pre-existing bytes for byte-identical assertion
    before: dict[Path, bytes] = {}
    for sub in ("queries", "costs", "metrics"):
        for p in (results_dir / sub).glob("*.json"):
            before[p] = p.read_bytes()
    # Note: comparison.md WILL be regenerated, so don't capture it for byte-identity.

    new_tier4_path: dict = {}

    async def fake_run_amain(args, console):
        sha = args.git_sha_override
        ts = args.ts_override
        new_tier4 = Path(args.output_dir) / "queries" / "tier-4-2026-05-05T12_00_00Z.json"
        new_tier4.parent.mkdir(parents=True, exist_ok=True)
        new_tier4.write_text(json.dumps({
            "tier": "tier-4",
            "git_sha": sha,
            "timestamp": ts,
            "model": "google/gemini-2.5-flash",
            "records": [{
                "question_id": "q1",
                "question": "?",
                "answer": "stub",
                "retrieved_contexts": ["ctx"],
                "latency_s": 1.0,
            }],
        }))
        # Write a tier-4 cost file too so aggregate_tier returns a row
        new_cost = Path(args.output_dir) / "costs" / "tier-4-eval-2026-05-05T12_00_00Z.json"
        new_cost.write_text(json.dumps({"totals": {"usd": 0.001}, "queries": []}))
        # Bump mtime fresher than pre-existing files
        now = time.time()
        os.utime(new_tier4, (now, now))
        os.utime(new_cost, (now, now))
        new_tier4_path["path"] = new_tier4
        return 0

    async def fake_score_amain(args, console):
        new_metric = Path(args.output_dir) / "metrics" / "tier-4-2026-05-05T12_00_00Z.json"
        new_metric.parent.mkdir(parents=True, exist_ok=True)
        new_metric.write_text(json.dumps([
            {"question_id": "q1", "faithfulness": 0.95,
             "answer_relevancy": 0.95, "context_precision": 0.95, "nan_reason": None},
        ]))
        now = time.time()
        os.utime(new_metric, (now, now))
        return 0

    monkeypatch.setattr(run, "amain", fake_run_amain)
    monkeypatch.setattr(score, "amain", fake_score_amain)
    # Do NOT mock compare._run — let real one regenerate comparison.md.

    # Provide a minimal golden_qa.json that compare._run loads from _REPO_ROOT
    # (compare.py reads <repo>/evaluation/golden_qa.json). The real file exists,
    # so we leave it alone.

    rc = pipeline.main([
        "--tiers", "4",
        "--tier-4-from-cache", "/dev/null",
        "--results-dir", str(results_dir),
        "--yes",
    ])
    assert rc == 0, "pipeline returned non-zero"

    # (b) byte-identical preservation of pre-existing tier-{1,2,3,5} files
    for path, original_bytes in before.items():
        assert path.read_bytes() == original_bytes, f"file mutated: {path}"

    # (c) comparison.md regenerated and contains all 5 tier rows
    cmp_md = (results_dir / "comparison.md").read_text()
    for label in ("tier-1", "tier-2", "tier-3", "tier-4", "tier-5"):
        assert label in cmp_md, f"comparison.md missing {label}"

    # (e) WARNING W5 closure: SHA on disk in new tier-4 JSON
    new_tier4 = new_tier4_path["path"]
    payload = json.loads(new_tier4.read_text())
    assert payload["git_sha"] == sweep_sha, (
        f"propagated SHA missing on disk: expected {sweep_sha}, got {payload.get('git_sha')}"
    )
    assert payload["timestamp"] == sweep_ts
