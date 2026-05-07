---
phase: 08-multi-judge-spot-check
verified: 2026-05-07T17:00:00Z
status: passed
score: 4/4
overrides_applied: 0
gaps: []
---

# Phase 8: Multi-Judge Spot-Check — Verification Report

**Phase Goal:** 5 questions × 3 tiers (15 cells) re-scored with a non-Gemini judge (Claude Haiku or GPT-4.1-mini) so the family-bias disclosure in the frozen doc cites a measured delta, not just published-paper magnitudes.
**Verified:** 2026-05-07T17:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Step 0: Previous Verification

No prior VERIFICATION.md found. Initial mode.

---

## Forward-Contract Guard (NON-NEGOTIABLE)

**Command:** `git diff ce0a8b0 HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py shared/cost_tracker.py shared/pricing.py | wc -c`

**Result:** `0`

VERDICT: PASS. All 9 RAW-LOCKed files are byte-identical to the pre-Phase-8 baseline (ce0a8b0). No source mutations outside the new module.

---

## Self-Check Verification

| Plan | Self-Check Claim | Verified |
|------|-----------------|----------|
| 08-01-SUMMARY.md | Self-Check: PASSED | CONFIRMED — all artifacts, commits, must-haves listed are present |
| 08-02-SUMMARY.md | Self-Check: PASSED | CONFIRMED — live test verdict CLEAN PASS, commits verified |

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | User can run re-score command with `--judge <non-gemini-slug>` and see structured JSON at `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json` | VERIFIED | CLI `--judge` flag confirmed via `--help` output. `amain` writes to `{results_dir}/metrics/multi-judge-spotcheck-{TS}.json` (line 179 of module). Live test (08-02) confirmed 1 file written at `tmp_path/results/metrics/multi-judge-spotcheck-2026-05-07T16_36_05Z.json`. Note: JSONs land in tmp_path during tests (gitignored prod dir is design intent — see below). |
| SC-2 | Per-cell structure contains primary-judge score, secondary-judge score, and delta across faithfulness, answer_relevancy, context_precision | VERIFIED | `_build_cell` at lines 118-124 constructs `{primary: {f, ar, cp, nan_reason}, secondary: {f, ar, cp, nan_reason}, delta: {f, ar, cp}}`. Offline test `test_amain_writes_spotcheck_json` asserts all 3 blocks on all 15 cells. Live test asserts same shape. `_METRICS = ("faithfulness", "answer_relevancy", "context_precision")` at line 48. |
| SC-3 | Spend within $0.10–$0.30, recorded in `costs/multi-judge-spotcheck-{TS}.json` | VERIFIED | Live test verdict: `total_usd=$0.12225` (40.75% of $0.30 envelope). Cost ledger written at `tmp_path/results/costs/multi-judge-spotcheck-{TS}.json`. D-8 fallback estimator triggered (LiteLLM usage parser returns 0 for OpenRouter) — `estimated_usd=0.12225` is the authoritative cost figure. Schema: `{tier, timestamp, queries, totals, estimator:{estimated:true, method:fixed_per_cell, tokens_per_cell, estimated_input_tokens, estimated_output_tokens, estimated_usd}}`. Note: prod `evaluation/results/costs/` untouched (tmp_path pattern is by-design per Phase 7 pattern). |
| SC-4 | Secondary judge model + version recorded in JSON (no opaque "Claude") | VERIFIED | `secondary_judge` block in payload (lines 187-191): `model=_strip_openrouter_prefix(args.judge)` → `"anthropic/claude-haiku-4.5"`, `model_slug=args.judge` → `"openrouter/anthropic/claude-haiku-4.5"`, `embedder=_strip_openrouter_prefix(args.judge_emb)` → `"openai/text-embedding-3-small"`, `max_tokens=JUDGE_MAX_TOKENS` → `8192`. Live test asserts all 4 fields. |

**Score:** 4/4 truths verified

---

## SC-1 Tmp-Path Design Note

The SUMMARY context explains that spot-check JSONs are staged to `tmp_path` during tests rather than persisted to `evaluation/results/` (which is gitignored). This is intentional and correct:

- `evaluation/results/metrics/*.json` and `evaluation/results/costs/*.json` are gitignored (`.gitignore` lines 3-5 per SUMMARY)
- The live test uses `tmp_path/results/` as the staging area and explicitly asserts no new files leak to the production cost dir
- Re-running `python -m evaluation.harness.multi_judge_spotcheck --source-ts 2026-05-07T10:59:10Z --results-dir evaluation/results` would produce the persistent JSON in `evaluation/results/metrics/` as SC-1 describes — the CLI path exists and the argument is wired correctly
- The SUMMARY's documented JSON shape + cost ledger contents serve as provenance evidence

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `evaluation/harness/multi_judge_spotcheck.py` | New module, 227 raw LOC, 6 public helpers + CLI | VERIFIED | File exists, `wc -l` = 227, all 6 helpers importable: `_signed_delta`, `_filter_records`, `_read_primary_metrics`, `_read_source_sha`, `_estimate_cost_fallback`, `amain` |
| `evaluation/tests/test_eval_multi_judge_spotcheck.py` | 12 offline tests + 1 live test | VERIFIED | 12 offline collected and passing, 1 live test at Section C with `@pytest.mark.live`; total file 683 LOC |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| CLI (`--judge` flag) | `amain` | `build_parser` → `args.judge` | WIRED | `--judge` declared at line 215, consumed at lines 146, 149, 188, 195 |
| `amain` | output JSON | `out_path.write_text` (line 194) | WIRED | Path = `metrics_dir / f"multi-judge-spotcheck-{_ts_for_filename(spotcheck_run_ts)}.json"` |
| `amain` | cost ledger | `cost_tracker.persist(dest_dir=...)` (line 195) | WIRED | Explicit dest_dir = `Path(args.results_dir) / "costs"` — Pitfall 2 / D-7 fix confirmed |
| `_read_source_sha` | `QueryLog.git_sha` | `read_query_log(path).git_sha` (line 81) | WIRED | NEVER calls `_git_sha()` — BLOCKER #3 fix; grep-guard confirms buggy pattern absent (exit_code=1) |
| `secondary_judge.max_tokens` | `JUDGE_MAX_TOKENS=8192` | line 190, 47 | WIRED | Constant set at line 47, wired into payload at line 190; live test asserts `== 8192` |
| `_estimate_cost_fallback` | fallback estimator block | triggered when `in_tok == 0 and n_scored > 0` (line 174) | WIRED | Cost payload gets `estimator` dict written back at lines 199-206 |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `amain` → `cells[]` | `secondary_scores` from `score_query_log` | real OpenRouter API (live); stub in offline tests | Yes (live test: 15/15 cells, 0/15 nan_reason) | FLOWING |
| `amain` → `primary` block | `primary_by_qid` from `_read_primary_metrics` | Phase 7 captured metrics JSON files | Yes (reads pinned ts file, not mtime-newest) | FLOWING |
| `amain` → `source_capture_git_sha` | `_read_source_sha` → `read_query_log().git_sha` | Phase 7 QueryLog JSON | Yes (returns `75f6f1b`, never `_git_sha()`) | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CLI `--judge` flag present | `uv run python -m evaluation.harness.multi_judge_spotcheck --help` | Shows `--judge JUDGE` flag | PASS |
| All 8 CLI flags present | `--help` output | `--source-ts`, `--tiers`, `--judge`, `--judge-emb`, `--question-ids`, `--results-dir`, `--batch-size`, `--yes` — all 8 present | PASS |
| 12 offline tests pass | `uv run pytest -m 'not live' evaluation/tests/test_eval_multi_judge_spotcheck.py -v` | `12 passed, 1 deselected in 2.66s` | PASS |
| Full offline regression | `uv run pytest -m 'not live' evaluation/tests/ --ignore=evaluation/tests/test_eval_adapters.py` | `128 passed, 6 deselected in 6.83s` | PASS |
| BLOCKER #3 grep-guard | `grep -nE 'source_capture_git_sha\s*=\s*_git_sha\s*\(' multi_judge_spotcheck.py` | exit_code=1 (no match) | PASS |
| LOC budget | `wc -l evaluation/harness/multi_judge_spotcheck.py` | `227` (≤230 budget) | PASS |
| Forward-contract guard | `git diff ce0a8b0 HEAD -- {9 RAW-LOCKed files} | wc -c` | `0` | PASS |
| All 6 Phase 8 commits exist | `git log --oneline 54e62c0 1e9db22 3baa8a8 6ab7396 3f37e4b 676f07d` | All 6 resolved | PASS |

---

## Module API Surface Verification

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| `_signed_delta` | public helper | importable, tested | VERIFIED |
| `_filter_records` | public helper | importable, tested | VERIFIED |
| `_read_primary_metrics` | public helper | importable, tested | VERIFIED |
| `_read_source_sha` | public helper — reads `src_log.git_sha`, NEVER `_git_sha()` | importable; dedicated test with `_git_sha` monkeypatched to `DEADBEEF` asserts returns `75f6f1b` | VERIFIED |
| `_estimate_cost_fallback` | public helper | importable, tested | VERIFIED |
| `amain` | public async entrypoint | importable, 4 integration tests | VERIFIED |
| `WANTED_IDS` | `("single-hop-001", "single-hop-002", "multi-hop-001", "multi-hop-002", "multimodal-001")` | Matches exactly | VERIFIED |
| `DEFAULT_TIERS` | `(1, 4, 5)` | Matches exactly | VERIFIED |
| `SECONDARY_JUDGE_DEFAULT` | `"openrouter/anthropic/claude-haiku-4.5"` | Matches exactly | VERIFIED |
| `JUDGE_MAX_TOKENS` | `8192` | `8192` | VERIFIED |
| `PRIMARY_JUDGE_MODEL` | `"google/gemini-2.5-flash"` | Matches | VERIFIED |
| `PRIMARY_EMBEDDER` | `"openai/text-embedding-3-small"` | Matches | VERIFIED |

---

## JSON Output Schema (SC-2 Completeness)

Verified from `amain` source (lines 180-193) + offline integration test assertions:

```
{
  "$schema_version": "1.0",
  "spotcheck_run_timestamp": "<ISO>",
  "spotcheck_run_git_sha": "<current HEAD>",
  "source_capture_timestamp": "<Phase-7 ts>",
  "source_capture_git_sha": "<Phase-7 sweep_sha = 75f6f1b>",
  "primary_judge": {"model": "google/gemini-2.5-flash", "embedder": "openai/text-embedding-3-small"},
  "secondary_judge": {
    "model": "anthropic/claude-haiku-4.5",        ← SC-4: specific version, not opaque "Claude"
    "model_slug": "openrouter/anthropic/claude-haiku-4.5",
    "embedder": "openai/text-embedding-3-small",
    "max_tokens": 8192
  },
  "cells": [                                        ← 15 cells (5 IDs × 3 tiers)
    {
      "question_id": "...",
      "tier": "tier-N",
      "primary": {"faithfulness": F, "answer_relevancy": A, "context_precision": C, "nan_reason": null},
      "secondary": {"faithfulness": F, "answer_relevancy": A, "context_precision": C, "nan_reason": null},
      "delta": {"faithfulness": D, "answer_relevancy": D, "context_precision": D}
    }, ...
  ],
  "aggregate": {
    "by_tier": {"tier-1": {...}, "tier-4": {...}, "tier-5": {...}},
    "overall": {"n_cells": 15, "n_skipped_due_to_nan": 0, "mean_delta_faithfulness": ..., ...}
  }
}
```

All SC-2 required fields (primary/secondary/delta × faithfulness/answer_relevancy/context_precision) confirmed present in both offline integration test assertions and live test assertions.

---

## Dual-SHA Provenance Verification

| Field | Value at Live Run | Source | Invariant |
|-------|------------------|--------|-----------|
| `source_capture_git_sha` | `"75f6f1b"` | `QueryLog.git_sha` via `_read_source_sha` | BLOCKER #3 fix — NEVER `_git_sha()` |
| `spotcheck_run_git_sha` | `"3f37e4b"` | `_git_sha()` (current HEAD at invocation) | Expected — HEAD advanced post-capture |
| Mismatch warning | Yellow console warning printed | `source_capture_git_sha != spotcheck_run_git_sha` check at line 140 | WIRED |

Live test confirms `source_capture_git_sha == "75f6f1b"` (Phase 7 sweep_sha) regardless of current HEAD. BLOCKER #3 fix verified end-to-end.

---

## Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| CAP-02 | Multi-judge spot-check producing measured Gemini vs secondary-judge delta | SATISFIED | Closed at unit + integration (08-01) + live (08-02) levels; 15-cell JSON with per-cell delta produced; total spend $0.12225 |

---

## Anti-Patterns Scan

Files scanned: `evaluation/harness/multi_judge_spotcheck.py`, `evaluation/tests/test_eval_multi_judge_spotcheck.py`

| File | Pattern | Finding | Verdict |
|------|---------|---------|---------|
| `multi_judge_spotcheck.py` | TODO/FIXME/placeholder | None found | CLEAN |
| `multi_judge_spotcheck.py` | `return null` / empty stubs | None; all helpers have real logic | CLEAN |
| `multi_judge_spotcheck.py` | Hardcoded empty data to rendering | `WANTED_IDS` is hardcoded but by design (D-2); not a stub | INFO — intentional |
| `multi_judge_spotcheck.py` | `source_capture_git_sha = _git_sha(...)` | ABSENT (grep-guard exit_code=1) | CLEAN |
| `test_eval_multi_judge_spotcheck.py` | Empty implementations | Stubs (`_stub_score_query_log_factory`, `_stub_build_judge`, `_stub_load_golden_qa`) exist but are test fixtures, not production code | INFO — appropriate test patterns |

No blockers. No warnings.

---

## Human Verification Required

None. All 4 ROADMAP SCs are verifiable programmatically through:
- Module source code inspection (SC-2 schema, SC-4 provenance fields)
- Offline integration test assertions (SC-1, SC-2, SC-4)
- Live test transcript in 08-02-SUMMARY (SC-3 spend, SC-1 path, SC-2 15-cell shape)
- Forward-contract guard returning 0

---

## Gaps Summary

No gaps. All 4 ROADMAP success criteria are VERIFIED against the codebase.

---

## Phase 8 Verification Conclusion

**CAP-02 status:** CLOSED end-to-end (unit + integration + live).

**Phase 8 is ready to be marked complete in ROADMAP.md. Phase 9 (Frozen Handoff Doc) is unblocked.**

**Key Phase 9 contract facts established by Phase 8:**
- Spot-check JSON path pattern: `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`
- Cost ledger path pattern: `evaluation/results/costs/multi-judge-spotcheck-{TS}.json`
- Schema version: `1.0` (locked)
- Source citation SHA for frozen doc: `source_capture_git_sha = "75f6f1b"` (Phase 7 sweep)
- Effective spend for disclosure: `estimator.estimated_usd = $0.12225` (not `totals.usd = $0.00` — D-8 LiteLLM known limitation)
- Per-tier mean deltas for family-bias disclosure: tier-1 Δ_F=-0.035, tier-4 Δ_F=+0.010, tier-5 Δ_F=-0.164; overall Δ_F=-0.063; context_precision universally negative (overall -0.172)

---

_Verified: 2026-05-07T17:00:00Z_
_Verifier: Claude (gsd-verifier)_
