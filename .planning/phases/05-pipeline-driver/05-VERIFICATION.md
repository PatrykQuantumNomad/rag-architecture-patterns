---
phase: 05-pipeline-driver
verified: 2026-05-06T12:05:00Z
status: passed
score: 4/4
overrides_applied: 0
---

# Phase 5: Pipeline Driver — Verification Report

**Phase Goal:** Capture -> score -> compare -> freeze runs as one command with a single git SHA and ISO timestamp captured at start, and re-running a single tier does not invalidate the captured runs of the other four.
**Verified:** 2026-05-06T12:05:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` runs capture → score → compare → freeze in sequence with one git SHA propagated to every per-tier output JSON | VERIFIED | `pipeline.py` lines 95-146: `sweep_sha = run._git_sha()` + `sweep_ts = run._ts()` captured ONCE at `amain` entry; threaded into `run.amain` via `git_sha_override`/`ts_override` kwargs; `run.py` lines 163-164 apply them; live test `test_pipeline_live_tier5_smoke` PASS verdict confirms queries.git_sha=fd287f0 matches HEAD |
| 2 | `pipeline --tiers 4 --tier-4-from-cache <path>` reruns only Tier 4; tiers 1,2,3,5 retain previous run data picked up by `_latest()` mtime resolution in `compare.py` | VERIFIED | `_build_compare_args` hardcodes `tiers="1,2,3,4,5"` (D-Q4, pipeline.py line 89); `compare._latest` exists and is mtime-sorted (compare.py line 48); Test 15 (`test_single_tier_rerun_preserves_others`) uses a real `compare._run` and asserts pre-existing tier-{1,2,3,5} files are byte-identical post-rerun AND comparison.md contains all 5 tier rows |
| 3 | `pipeline.py` calls `run.amain()` / `score.amain()` / `compare._run()` as in-process function calls (not subprocesses) | VERIFIED | Zero occurrences of `subprocess`, `os.system`, `os.popen` in pipeline.py (grep confirmed empty); Test 13 (`test_no_subprocess_calls`) reads pipeline.py source text and asserts absence; actual calls: `await run.amain(...)`, `await score.amain(...)`, `await asyncio.to_thread(compare._run, ...)`, `await asyncio.to_thread(freeze.freeze, ...)` |
| 4 | Cost-surprise prompts fire once per pipeline run (not five times per tier) when running a full sweep | VERIFIED | `_build_run_args` passes `yes=True` (line 60); `_build_score_args` passes `yes=True` (line 79); pipeline's single `if not args.yes:` prompt (lines 103-115) is the only cost gate; Test 6 (`test_single_cost_prompt`) and Test 11 (`test_yes_flag_consolidation`) verify this |

**Score:** 4/4 truths verified

### Deferred Items

None.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `evaluation/harness/pipeline.py` | ≤220 raw LOC, in-process composition driver | VERIFIED | `wc -l` = 189 (14% buffer under 220 cap) |
| `evaluation/harness/run.py` | Additive `git_sha_override` + `ts_override` kwargs on `_capture_tier`; ≤10 raw insertions | VERIFIED | Commit aa1640c: +10 raw insertions / -3 deletions; `_capture_tier` signature lines 152-153; `amain` plumb-through lines 320-323 |
| `evaluation/tests/test_eval_pipeline.py` | 15 offline tests + 1 live-marked test | VERIFIED | 15 functions (`test_build_parser_*` through `test_single_tier_rerun_preserves_others`) + `test_pipeline_live_tier5_smoke` at line 615 with `@pytest.mark.live` |
| `.planning/phases/05-pipeline-driver/05-01-SUMMARY.md` | Exists, Self-Check: PASSED | VERIFIED | File exists; "Self-Check: PASSED" at line 177 |
| `.planning/phases/05-pipeline-driver/05-02-SUMMARY.md` | Exists, Self-Check: PASSED | VERIFIED | File exists; "Self-Check: PASSED" at line 221 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pipeline.amain` | `run.amain` | Direct async call with synthesized Namespace | WIRED | Line 117: `await run.amain(_build_run_args(args, sweep_sha, sweep_ts), console)` |
| `pipeline.amain` | `score.amain` | Direct async call with synthesized Namespace | WIRED | Line 122: `await score.amain(_build_score_args(args), console)` |
| `pipeline.amain` | `compare._run` | `asyncio.to_thread` (sync wrapper) | WIRED | Line 127: `await asyncio.to_thread(compare._run, _build_compare_args(args))` |
| `pipeline.amain` | `freeze.freeze` | `asyncio.to_thread` (sync wrapper) | WIRED | Lines 134-139; Phase 4 contract signature `freeze(version, force, results_dir, source)` honored verbatim |
| `pipeline._build_run_args` | `run._capture_tier` via `run.amain` | `git_sha_override` + `ts_override` kwargs on Namespace | WIRED | `run.py` lines 320-323: `sha_o = getattr(args, "git_sha_override", None)`; forwarded to `_capture_tier` with keyword args |
| `_capture_tier` SHA path | per-tier output JSON `git_sha` field | `git_sha = git_sha_override or _git_sha()` | WIRED | `run.py` line 164; Test 15 integration asserts `payload["git_sha"] == sweep_sha` from file on disk; live test asserts `queries.git_sha == "fd287f0"` |
| `_build_compare_args` | all-5-tier rollup | Hardcoded `tiers="1,2,3,4,5"` | WIRED | Line 89 of pipeline.py |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `pipeline.py` (amain) | `sweep_sha` | `run._git_sha()` → `subprocess(['git','rev-parse','--short','HEAD'])` | Yes — live test confirms `fd287f0` (non-"unknown") | FLOWING |
| per-tier output JSON `git_sha` | `git_sha_override` forwarded from `sweep_sha` | pipeline.amain → run.amain → _capture_tier kwargs | Yes — Test 15 + live test both confirm SHA on disk | FLOWING |
| `comparison.md` | tier rows from `aggregate_tier` | `compare._latest()` picks freshest per-tier metrics/queries files by mtime | Yes — HARN-02 integration test uses real compare._run and verifies all 5 rows | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `--help` exits 0, all 12 flags rendered | `uv run python -m evaluation.harness.pipeline --help` | All 12 flags present in output | PASS |
| Unsupported tier exits non-zero | `uv run python -m evaluation.harness.pipeline --tiers 99 --yes` | "Unsupported tier(s): [99]" + "Pipeline halted at capture stage (exit 2)" | PASS |
| 15 offline tests pass | `uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live' --no-header -q` | `15 passed, 1 deselected in 2.82s` | PASS |
| run.py regression (7 tests) | `uv run pytest evaluation/tests/test_eval_run.py -m 'not live' --no-header -q` | `7 passed in <0.5s` (part of 22 combined) | PASS |
| No subprocess calls in pipeline.py | `grep -E 'subprocess\|os\.system\|os\.popen' pipeline.py` | Empty result | PASS |
| Byte-identical guard (score/compare/freeze) | `git diff aa1640c..HEAD -- harness/{score,compare,freeze}.py \| wc -c` | 0 bytes | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| HARN-01 | 05-01 + 05-02 | Single git SHA + ISO timestamp propagated across all per-tier outputs in one pipeline command | SATISFIED | pipeline.py captures SHA once; threaded to run.py; live test `test_pipeline_live_tier5_smoke` PASS (queries.git_sha=fd287f0, total_usd=$0.007010) |
| HARN-02 | 05-01 | Single-tier rerun preserves other tiers via `_latest()` mtime resolution | SATISFIED | Test 15 integration: pre-existing tier-{1,2,3,5} files byte-identical post `--tiers 4` rerun; comparison.md contains all 5 rows |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `run.py` / `score.py` | 273 / 518 | `tracker.persist()` ignores `--output-dir` / `--results-dir` (hardcoded `DEFAULT_COSTS_DIR`) | Info | Cost files always land in `evaluation/results/costs/` regardless of caller's dir argument. Documented as v1.1 hardening item in 05-02-SUMMARY.md. Does not affect pipeline correctness; test works around it via monkeypatch. |

No blockers. No STUB or MISSING artifacts detected.

### Human Verification Required

None. All must-haves are verifiable programmatically. The live smoke test (`test_pipeline_live_tier5_smoke`) ran against the real OpenRouter API on 2026-05-06 with verdict PASS (rc=0, total_usd=$0.007010, queries.git_sha=fd287f0 matching HEAD, 1/1/1/1 tier-5 artifacts on disk, comparison.md regenerated with tier-5 row).

### Gaps Summary

No gaps. All 4 success criteria are verified end-to-end at code + test + (for SC1/HARN-01) live-API levels.

---

## Phase 6 / Phase 7 Dependency Health

Phase 5 is a direct dependency for Phase 7 (Full 5-Tier Rerun). The single command of record `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` is proven functional. Two v1.1 carry-forward items from Phase 5 do NOT block Phase 7:

1. **`tracker.persist()` ignores `--results-dir`** — cost files land in `evaluation/results/costs/` regardless of `--results-dir`. Phase 7 uses the default `evaluation/results/` dir so this is transparent; no impact.
2. **LiteLLM judge token-usage underreporting** — `ragas-judge-*.json` shows `totals.usd=$0.0` despite real spend ~$0.001/run. Phase 7's spend tracking will undercount judge costs by a small margin; within the documented $1–3 budget tolerance.

Neither item undermines the SHA-propagation invariant (HARN-01) or the single-tier-rerun invariant (HARN-02) that Phase 7 depends on.

---

_Verified: 2026-05-06T12:05:00Z_
_Verifier: Claude (gsd-verifier)_
