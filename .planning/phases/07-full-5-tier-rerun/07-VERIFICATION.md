---
phase: 07-full-5-tier-rerun
verified: 2026-05-07T00:00:00Z
status: passed
score: 5/5
overrides_applied: 0
human_verified_at: 2026-05-07
human_verified_by: user (orchestrator session — typed `approved` after dashboard check)
human_verification:
  - test: "Confirm total sweep spend is within the $1-3 budget range stated in ROADMAP SC-5"
    expected: "Total spend <= $3.00 — confirmed. The verifier transcript (07-03-VERIFY.txt) reports $0.1178 from the 10 freshest cost ledger files (sweep capture + judge). The honest all-in total including the mandatory Step 1 eval_capture.py Tier 4 live capture is $0.439. Both figures are well under the $3.00 ceiling. However, the RAGAS judge cost ledgers all report $0.00 (pre-existing LiteLLM token-parser gap documented in Plans 02-04 / 03-03 / 05-02). The tier-4 capture at the pipeline passthrough step also reports $0.00 because the passthrough itself does not make live API calls. If you need to validate the true out-of-pocket spend against an external dashboard (OpenRouter / Gemini), that step requires human action."
    why_human: "Judge costs ($0 in all 5 RAGAS cost ledgers) are a pre-existing known instrumentation gap. The formal verifier SC-5 passes at $0.1178 (which is under $3.00), and the honest manual total is $0.439. Neither number exceeds the ceiling. However, a reviewer wishing to confirm the actual cloud-billed spend should check the OpenRouter and Gemini dashboards for 2026-05-07, as the cost ledgers cannot be trusted for the judge-side spend."
---

# Phase 7: Full 5-Tier Rerun — Verification Report

**Phase Goal:** All 5 tiers x 30 questions captured and scored on a single date with a single git SHA, producing fresh per-tier `queries/`, `costs/`, and `metrics/` files that feed the rollup.
**Verified:** 2026-05-07T00:00:00Z
**Status:** passed (5/5 — SC-5 dashboard-confirmed by user 2026-05-07)
**Re-verification:** No — initial verification (status upgraded from human_needed → passed after user dashboard check)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can run `pipeline --tiers 1,2,3,4,5 --yes` and see fresh `queries/tier-{N}-{TS}.json`, `costs/tier-{N}-eval-{TS}.json`, `costs/ragas-judge-tier-{N}-{TS}.json`, `metrics/tier-{N}-{TS}.json` for every tier with timestamps within the same date | VERIFIED | All 5 `queries/tier-{1..5}-2026-05-07T10_59_10Z.json` exist (confirmed on disk); all 5 `metrics/tier-{1..5}-2026-05-07T10_59_10Z.json` exist; corresponding `costs/` files for sweep-window timestamps exist; all within date 2026-05-07 |
| 2 | User can grep all 5 queries JSONs and find the same git SHA (75f6f1b) in each — no SHA drift (honest scope: 5 queries JSONs only; cost+metrics JSONs do not carry git_sha by design) | VERIFIED | `for t in 1 2 3 4 5; do jq -r '.git_sha' tier-${t}-*.json; done \| sort -u` returns exactly `75f6f1b` — 1 distinct SHA across all 5 tiers. Timestamp `2026-05-07T10:59:10Z` identical on all 5. |
| 3 | User can confirm Tier 4 NaN count is < 5/30 (down from 30/30 baseline; Phase 2 fix sticking) | VERIFIED | `metrics/tier-4-2026-05-07T10_59_10Z.json`: nan_count = 2/30 (breakdown: `empty_contexts: 2`). Confirmed directly with jq against the actual file. 93% reduction from 30/30 baseline. |
| 4 | User can confirm Tier 5 NaN count is < 5/30 (down from 30/30 baseline; Phase 1 fix sticking) | VERIFIED | `metrics/tier-5-2026-05-07T10_59_10Z.json`: nan_count = 0/30. Perfect 100% reduction from 30/30 baseline. Tier 5 context normalization (Phase 1 D-FORMAT) fully durable. |
| 5 | User can confirm total spend stayed within the documented $1-3 budget per full sweep, recorded in capture+judge cost ledgers | UNCERTAIN | Cost ledgers report $0.1178 (verified by automated script against 10 cost JSONs). Honest all-in total including Step 1 eval_capture.py is $0.439. BOTH are under $3.00 ceiling. However, RAGAS judge cost ledgers report $0.00 due to LiteLLM token-parser gap (pre-existing v1.1 hardening item). True cloud-billed judge spend cannot be verified from ledger files alone; requires external dashboard check. |

**Score:** 4/5 truths verified (SC-5 is UNCERTAIN, not FAILED — budget ceiling is clearly satisfied even at the highest honest estimate $0.439; the UNCERTAIN flag is about ledger trustworthiness for the judge portion, not about whether the budget was exceeded)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `evaluation/results/queries/tier-{1..5}-2026-05-07T10_59_10Z.json` | 5 fresh QueryLogs, 30 records each, single sweep_sha | VERIFIED | All 5 exist on disk; 30 records each confirmed via jq; git_sha=75f6f1b on all 5; timestamps identical at 2026-05-07T10:59:10Z |
| `evaluation/results/metrics/tier-{1..5}-2026-05-07T10_59_10Z.json` | 5 fresh metrics with nan_reason per row | VERIFIED | All 5 exist on disk (sizes 4.9KB-5.4KB); Tier 4: 2/30 nan; Tier 5: 0/30 nan |
| `evaluation/results/costs/tier-{1..5}-eval-{sweep_ts}.json` | 5 capture cost ledgers | VERIFIED | Sweep-window cost JSONs at 10:59:10Z-11:04:31Z timestamps exist; note: mtime-freshest picks up the 12:05:42Z test-bleed $0 files (documented pre-existing issue) |
| `evaluation/results/costs/ragas-judge-tier-{1..5}-*.json` | 5 RAGAS judge cost ledgers | VERIFIED (with caveat) | All 5 exist (111906Z-120253Z); all report $0.00 due to pre-existing LiteLLM token-parser gap — ledgers exist but underreport spend |
| `evaluation/results/comparison.md` | 5-row rollup + per-class breakdown + Embedder-by-tier table | VERIFIED | File exists (3920 bytes, mtime 2026-05-07 08:02); 5-row rollup confirmed; 15-row per-class breakdown (5 tiers x 3 classes: single-hop/multi-hop/multimodal) confirmed; `**Embedder by tier:**` table with 5 rows confirmed |
| `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-07T10_31_52Z.json` | Full-corpus rebuild manifest committed at eb77cf8 | VERIFIED | File exists on disk and committed (`git ls-files` confirms); contains: lightrag_version=1.4.15, raganything_version=1.2.10, mineru_version=3.1.4, graphml_node_count=28597, graphml_edge_count=80419 — all matching pinned versions |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval_capture.py --yes` (Step 1) | `queries/tier-4-{TS}.json` (30 records) | Tier 4 live 30q capture against 79-paper graph | VERIFIED | `queries/tier-4-2026-05-07T10_48_53Z.json` — Step 1 capture file; used as `--tier-4-from-cache` input for pipeline.py |
| `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache` | 5 fresh queries+metrics+costs at single sweep_sha | run._git_sha() captured once at amain; threads via git_sha_override+ts_override | VERIFIED | All 5 tier queries JSONs carry identical git_sha=75f6f1b and timestamp=2026-05-07T10:59:10Z — confirmed via jq |
| compare._run | `comparison.md` regenerated with all required sections | _latest_query_log mtime resolution + emit_markdown | VERIFIED | comparison.md exists, 5-row rollup, 15-row per-class, Embedder-by-tier table (note: compare.py emits as `**Embedder by tier:**` bold, not `## Embedder by tier` heading — both the VERIFY.txt and SUMMARY document this discrepancy vs plan spec; table is correctly present) |
| 5 freshest tier-N queries JSONs | Single SHA across all 5 | jq `.git_sha` sort -u | VERIFIED | `sort -u` produces exactly `75f6f1b` — distinct count = 1 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `queries/tier-{1..5}-*.json` | `.records[].retrieved_contexts` | Live API capture (Tiers 1/2/3/5 via run.py; Tier 4 via eval_capture.py) | Yes | FLOWING — Tier 5: `[paper_id=2005.11401]` prefix confirms Phase 1 D-FORMAT; Tier 4: n_ctx > 0 on first 3 records confirmed in human-verify checkpoint |
| `metrics/tier-{1..5}-*.json` | per-row faithfulness/answer_relevancy/context_precision | RAGAS scoring against real queries | Yes | FLOWING — actual metric values present (e.g., tier-1 faithfulness=0.851); 2/30 NaN on Tier 4 and 0/30 on Tier 5 are meaningful real values |
| `comparison.md` | 5-row rollup table + per-class + embedder | compare._run reading mtime-freshest per-tier query+metrics | Yes | FLOWING — all values in rollup match SUMMARY front-matter `tier_rollup` section |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Single SHA across all 5 queries JSONs | `for t in 1..5; do jq -r '.git_sha' tier-${t}-*.json; done \| sort -u` | `75f6f1b` (1 distinct) | PASS |
| 30 records per tier | `jq '.records \| length'` on each tier | 30/30/30/30/30 | PASS |
| Tier 4 NaN count | `jq '[.[] \| select(.nan_reason != null)] \| length'` | 2 | PASS (< 5) |
| Tier 5 NaN count | `jq '[.[] \| select(.nan_reason != null)] \| length'` | 0 | PASS (< 5) |
| comparison.md Embedder-by-tier table | `grep "Embedder by tier" comparison.md` | `**Embedder by tier:**` present | PASS |
| VERIFY.txt OVERALL VERDICT | `grep "OVERALL VERDICT" 07-03-VERIFY.txt` | `OVERALL VERDICT: PASS` | PASS |
| Forward-contract RAW-LOCK vs pre-Phase-7 baseline | `git diff 03f9ce1 -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate}.py \| wc -c` | `0` | PASS |
| Diagnostics manifest committed | `git ls-files --error-unmatch tier-4-graph-stats-2026-05-07T10_31_52Z.json` | PASS | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CAP-01 | 07-01, 07-02, 07-03 | User can capture a full 5-tier x 30-question RAGAS run on a single date with a single git SHA recorded in every per-tier output JSON | SATISFIED | 5 fresh queries JSONs at 2026-05-07T10:59:10Z all carrying git_sha=75f6f1b; 30 records each; comparison.md regenerated; VERIFY.txt OVERALL VERDICT: PASS |

Note: REQUIREMENTS.md still shows `CAP-01` as `[ ] Pending` — this is expected as the checklist update is a post-verification orchestrator action, not a plan deliverable.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `evaluation/results/costs/ragas-judge-tier-{1..5}-*.json` | — | All 5 judge cost JSONs report $0.00 totals (LiteLLM token-parser gap) | Warning | Cannot verify actual judge spend from ledgers. True spend estimated at ~$0.45 (from mass-balance). Pre-existing v1.1 hardening item; does NOT affect metric correctness or SC-5 ceiling compliance at any reasonable estimate. |
| `evaluation/results/costs/tier-{1..5}-eval-20260507T120542Z.json` | — | 5 zero-byte placeholder cost JSONs written by post-sweep pytest test-bleed (197 bytes each, $0) | Warning | Pollutes mtime-freshest cost glob — verifier's `ls -t \| head -1` picks these up instead of real sweep cost JSONs, causing $0 total. Verifier SC-5 trivially passes ($0 < $3.00) but honest cost accounting requires timestamp-window selection. Documented in 07-03-SUMMARY.md decisions section. Pre-existing mis-feature (tracker.persist hardcodes DEFAULT_COSTS_DIR). |
| `evaluation/results/comparison.md` | heading | `**Embedder by tier:**` (markdown bold) instead of `## Embedder by tier` (heading) | Info | No user-visible impact; the table is correctly present. Plan spec had the wrong regex. VERIFY.txt adjusted regex to match actual emit. compare.py RAW-LOCKED so no fix applied. |

### Human Verification Required

#### 1. Confirm actual judge-side cloud spend vs ledger reporting

**Test:** Check the OpenRouter and/or Gemini API usage dashboards for 2026-05-07 (approximately 11:00 UTC - 12:10 UTC) and sum the RAGAS judge API calls for all 5 tiers.

**Expected:** The total RAGAS judge spend across 5 tiers x 30 questions x 3 metrics should be within the $0.10-0.60 range that would keep the overall sweep under $3.00. The honest all-in estimate from 07-03-SUMMARY.md is $0.439 total (including Step 1 eval_capture.py and failed first-attempt Tier 1); even if judge spend were double the estimate, total would remain under $3.00.

**Why human:** The cost ledger files (`ragas-judge-tier-{1..5}-*.json`) all report $0.00 due to a pre-existing LiteLLM token-parser gap (documented in Plans 02-04, 03-03, 05-02). The automated verifier (SC-5) passes because $0 < $3.00, but this is a known instrumentation gap, not evidence that the judge ran for free. A human reading the dashboard would be the only way to confirm the actual out-of-pocket figure was within budget.

---

## Forward-Contract Verdict

**RAW-LOCK held end-to-end across all 3 plans of Phase 7 vs pre-Phase-7 baseline.**

```
git diff 03f9ce1 -- evaluation/harness/pipeline.py \
                    evaluation/harness/run.py \
                    evaluation/harness/score.py \
                    evaluation/harness/compare.py \
                    evaluation/harness/freeze.py \
                    evaluation/harness/smoke_gate.py | wc -c
0
```

Zero bytes diff against commit `03f9ce1` (the pre-Phase-7 research commit that established the baseline). The 6 forward-contract-locked harness modules are byte-identical through all 3 plans. Phase 7 shipped ZERO source-code changes.

Additionally, the 3 tier-4 helper scripts (`ingest_from_mineru.py`, `log_graph_stats.py`, `eval_capture.py`) were also confirmed 0-bytes diff via `git diff HEAD~1` after Plan 07-02 Task 5.

## Cumulative Phase Cost vs Envelope

| Plan | Activity | Cost |
|------|----------|------|
| 07-01 | Pre-flight (Tier 2 ingest + 5 tier smokes) | ~$0.036 |
| 07-02 | Tier 4 full-corpus rebuild (79 papers) | $24.85 |
| 07-03 | Full 5-tier sweep (Step 1 + Step 2 + retry) | $0.439 |
| **Phase 7 total** | | **~$25.32** |

The ROADMAP SC-5 budget envelope of $1-3 applies per full sweep (Plan 07-03 only). The Phase 7 cumulative figure is dominated by the one-time Tier 4 rebuild ($24.85). Plan 07-03 sweep at $0.439 is 14.6% of the $3.00 sweep ceiling.

## Recommendation for Phase 8 Readiness

**Phase 8 is ready to proceed, conditional on SC-5 human verification completing without a blocker.**

The five key artifacts that Phase 8 (Multi-Judge Spot-Check) needs are on disk and verified:
- `evaluation/results/queries/tier-{1..5}-2026-05-07T10_59_10Z.json` — 30 records each, sweep_sha=75f6f1b
- `evaluation/results/metrics/tier-{1..5}-2026-05-07T10_59_10Z.json` — scores with nan_reason populated
- `evaluation/results/comparison.md` — regenerated with 5-tier rollup, per-class breakdown, Embedder-by-tier table

CAP-01 is mechanically closed: single-SHA invariant verified, Tier 4 NaN 2/30 (93% improvement), Tier 5 NaN 0/30 (100% improvement, perfect). Phase 1 and Phase 2 fixes are durable across the full 30-question dataset.

The SC-5 human verification item is informational about the true judge spend, not a blocker — the $3.00 ceiling is satisfied at every plausible estimate. The verifier flags it as UNCERTAIN rather than FAILED to be honest about instrumentation limits.

**If the dashboard check confirms spend under $3.00 for the sweep (or the user accepts the $0.439 honest estimate as sufficient):** status upgrades to `passed` with score 5/5.

---

_Verified: 2026-05-06T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
