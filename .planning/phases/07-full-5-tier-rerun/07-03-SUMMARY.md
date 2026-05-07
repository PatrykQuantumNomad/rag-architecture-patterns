---
phase: 07-full-5-tier-rerun
plan: 03
status: complete
subsystem: evaluation-harness
tags: [pipeline-driver, ragas, cap-01, full-sweep, single-sha-invariant, embedder-provenance]

# Dependency graph
requires:
  - phase: 07-full-5-tier-rerun
    provides: Plan 07-01 (pre-flight 5/5 tier smokes; Open Q1 verdict PASS) + Plan 07-02 (79-paper Tier 4 graph rebuild — 28597 nodes / 80419 edges)
  - phase: 06-embedder-provenance-capture
    provides: Embedder-by-tier table emission (D-Q1) wired into compare.py
  - phase: 05-pipeline-driver
    provides: pipeline.py capture→score→compare in one command (HARN-01 single-SHA propagation)
  - phase: 02-tier-4-multimodal-fix
    provides: Tier 4 NaN reduction (judge max_tokens=8192; empty_contexts NaN reason)
  - phase: 01-tier-5-adapter-fix
    provides: Tier 5 NaN reduction (D-FORMAT [paper_id=...] context normalization)
provides:
  - Five fresh per-tier QueryLogs at single sweep_sha=75f6f1b (CAP-01 closed)
  - Tier 4 NaN dropped from 30/30 baseline to 2/30 (Phase 2 fix sticking)
  - Tier 5 NaN dropped from 30/30 baseline to 0/30 (Phase 1 fix sticking — PERFECT score)
  - Regenerated comparison.md with 5-tier rollup + per-class breakdown + embedder-by-tier table
  - Plan 07-03-VERIFY.txt audit transcript with all 5 ROADMAP success criteria PASS
affects: [phase-08-multi-judge, phase-09-frozen-handoff-doc]

# Tech tracking
tech-stack:
  added: []  # No source code changes; RAW-LOCK preserved
  patterns:
    - "Single-sweep_sha invariant scoped to 5 queries JSONs (cost+metrics intentionally do not carry git_sha)"
    - "Pipeline rerun strategy on transient cloud-quota errors: rerun without committing, sweep_sha stays HEAD"

key-files:
  created:
    - .planning/phases/07-full-5-tier-rerun/07-03-SUMMARY.md
    - .planning/phases/07-full-5-tier-rerun/07-03-VERIFY.txt
  modified: []  # All source untouched

# Sweep provenance (load-bearing for Phase 8 + Phase 9)
sweep_metadata:
  sweep_sha: "75f6f1b"
  sweep_ts: "2026-05-07T10:59:10Z"
  date: "2026-05-07"
  tier_4_full_rebuild_manifest: "evaluation/results/diagnostics/tier-4-graph-stats-2026-05-07T10_31_52Z.json"
  tier_4_graph_nodes: 28597
  tier_4_graph_edges: 80419
  tier_4_papers: 79

# Per-tier QueryLog records (all carry sweep_sha)
queries_artifacts:
  - tier: 1
    path: "evaluation/results/queries/tier-1-2026-05-07T10_59_10Z.json"
    records: 30
    embedder: "openai/text-embedding-3-small"
    embedder_source: "openrouter"
  - tier: 2
    path: "evaluation/results/queries/tier-2-2026-05-07T10_59_10Z.json"
    records: 30
    embedder: "gemini-embedding-001"
    embedder_source: "google-managed"
  - tier: 3
    path: "evaluation/results/queries/tier-3-2026-05-07T10_59_10Z.json"
    records: 30
    embedder: "openai/text-embedding-3-small"
    embedder_source: "openrouter"
  - tier: 4
    path: "evaluation/results/queries/tier-4-2026-05-07T10_59_10Z.json"
    records: 30
    embedder: "openai/text-embedding-3-small"
    embedder_source: "openrouter"
    cache_source: "evaluation/results/queries/tier-4-2026-05-07T10_48_53Z.json (eval_capture.py)"
  - tier: 5
    path: "evaluation/results/queries/tier-5-2026-05-07T10_59_10Z.json"
    records: 30
    embedder: "openai/text-embedding-3-small"
    embedder_source: "openrouter"

# NaN counts (Phase 1 / Phase 2 fix-stickiness verification)
nan_counts:
  tier_4: { count: 2, total: 30, breakdown: { empty_contexts: 2 }, baseline: 30, improvement: "93% reduction" }
  tier_5: { count: 0, total: 30, breakdown: {}, baseline: 30, improvement: "100% reduction (perfect)" }

# Headline scores (5-tier rollup from comparison.md)
tier_rollup:
  - tier: 1, faithfulness: 0.851, answer_relevancy: 0.458, context_precision: 0.459, mean_latency_s: 2.78, n: 30, n_nan: 0
  - tier: 2, faithfulness: 0.427, answer_relevancy: 0.814, context_precision: 0.125, mean_latency_s: 6.41, n: 30, n_nan: 1
  - tier: 3, faithfulness: 0.968, answer_relevancy: 0.759, context_precision: 0.883, mean_latency_s: 1.16, n: 30, n_nan: 0
  - tier: 4, faithfulness: 0.791, answer_relevancy: 0.263, context_precision: 0.238, mean_latency_s: 6.61, n: 30, n_nan: 2
  - tier: 5, faithfulness: 0.768, answer_relevancy: 0.688, context_precision: 0.252, mean_latency_s: 9.13, n: 30, n_nan: 0

# Cost provenance (honest accounting)
cost_provenance:
  step_1_eval_capture_usd: 0.286902      # tier-4-eval-20260507T104853Z.json (eval_capture.py 30q live)
  step_2_capture_subtotal_usd: 0.117844  # 5 fresh tier-N-eval-20260507T1[01]*.json
  step_2_judge_subtotal_usd: 0.000000    # 5 fresh ragas-judge-tier-*.json (LiteLLM token-parser gap; Plan 02-04)
  failed_first_attempt_tier_1_usd: 0.034083  # Tier 1 completed before Gemini 429 stopped Tier 2
  total_usd: 0.438829
  estimate_usd: 0.84
  ceiling_usd: 3.00
  ceiling_status: "WELL UNDER (14.6% of ceiling)"

key-decisions:
  - "Auto-fixed Embedder-by-tier verifier regex mismatch: compare.py emits **Embedder by tier:** (markdown bold) not ## Embedder by tier (heading); RAW-LOCK preserved (Rule 3 deviation)"
  - "Reran pipeline.py once after transient Gemini 429 RESOURCE_EXHAUSTED on Tier 2 multimodal-005 embedding (Rule 3 blocking issue auto-fix); single-SHA invariant preserved because no commits between attempts and HEAD stayed at 75f6f1b"
  - "Honored single-SHA scoping caveat: verifier asserts SHA equality across 5 queries JSONs only (cost+metrics carry no git_sha by design per 07-RESEARCH.md)"
  - "Documented LiteLLM token-parser gap (Plan 02-04 known issue) producing $0 judge cost reports; verifier defensive .totals.usd // 0 handles correctly; ceiling check trivially passes"
  - "Documented test-bleed cost JSON write at 12:05:42Z (mtime-freshest mtime collision, not impacting verdict)"

patterns-established:
  - "Pattern: Plan 07 ordering — Plan 03 (single-sweep-sha capture/score/compare) MUST follow Plan 02 (Tier 4 graph rebuild) without intervening source-code commits to preserve semantic SHA invariant"
  - "Pattern: When verifying a freshness-mtime-based artifact pipeline, prefer ts-window grep over `ls -t | head -1` to avoid post-sweep test-bleed pollution"

# Metrics
duration: ~75min  (08:48 - 08:05Z = 1h 17m, includes 1 transient retry)
completed: 2026-05-07
---

# Phase 7 Plan 03: Full 5-Tier Sweep — CAP-01 Closure

**Single-sweep-SHA capture + score + compare across 5 tiers × 30 questions on a 79-paper corpus, with Tier 4 NaN dropped from 30/30 to 2/30 and Tier 5 from 30/30 to 0/30 — sweep_sha=75f6f1b, total spend $0.44 vs $3.00 ceiling.**

## Performance

- **Duration:** ~75 min wall (Step 1: ~6 min, Step 2 [retry incl.]: ~64 min, verifier+SUMMARY: ~5 min)
- **Started:** 2026-05-07T10:48:46Z (Step 1 pre-capture HEAD record)
- **Completed:** 2026-05-07T12:05:42Z (post-pipeline pytest finish)
- **Tasks:** 5/6 executed (Task 1 cost-ack pre-approved by orchestrator; Tasks 2-6 auto)
- **Files committed:** 2 (07-03-SUMMARY.md, 07-03-VERIFY.txt)
- **Source files modified:** 0 (forward-contract guard: 0 bytes diff on 9 RAW-LOCKED files)

## Accomplishments

- **CAP-01 invariant closed:** all 5 freshest tier-N queries JSONs carry sweep_sha=`75f6f1b`. SHA-equality grep returns 1 distinct value.
- **Tier 4 NaN reduction stuck:** 2/30 (down from 30/30 baseline) — `empty_contexts: 2`. Phase 2 multimodal fix is durable.
- **Tier 5 NaN reduction stuck:** 0/30 (down from 30/30 baseline) — **PERFECT SCORE**. Phase 1 D-FORMAT context normalization durable.
- **comparison.md regenerated** with 5-tier rollup + 15-row per-class breakdown + 5-row Embedder-by-tier table (Phase 6 D-Q1).
- **Total spend $0.439** vs $0.84 estimate vs $3.00 ceiling (14.6% of ceiling).
- **Forward-contract guard intact:** 9 RAW-LOCKED source files (6 harness modules + 3 tier-4 scripts) byte-identical post-sweep.
- **Offline regression baseline preserved:** 116 passed, 1 deselected (matches Plan 07-02 baseline; no regressions).

## ROADMAP Success Criteria — Verifier Verdicts

| SC | Description | Result | Evidence |
|----|-------------|--------|----------|
| SC-1 | Pipeline ran capture → score → compare in one command | **PASS** | `comparison.md` regenerated; 10 tier rows; Embedder-by-tier table present |
| SC-2 | Single git SHA across 5 freshest tier-N queries JSONs | **PASS** | All 5 carry `git_sha=75f6f1b` (distinct count = 1) |
| SC-3 | Tier 4 NaN < 5/30 | **PASS** | 2/30 (`empty_contexts: 2`) |
| SC-4 | Tier 5 NaN < 5/30 | **PASS** | 0/30 (perfect — no NaN reasons present) |
| SC-5 | Total spend ≤ $3 | **PASS** | $0.439 honest sweep total (verifier reported $0.118 from sweep-window cost JSONs) |
| Bonus | Same date across 5 captures | **PASS** | All 5 timestamped 2026-05-07 |

Full transcript: `.planning/phases/07-full-5-tier-rerun/07-03-VERIFY.txt`

## Task Commits

Plan 07-03 produces ONE end-of-plan commit per the plan's "no intervening commits" invariant:

1. **Task 1 (cost-ack)** — Pre-approved by orchestrator (no commit)
2. **Task 2 (Step 1 — Tier 4 30q live capture)** — No commit (per Pitfall 2: pipeline.py reads HEAD at amain entry; commits between Step 1 and Step 2 break semantic SHA provenance)
3. **Task 3 (Step 2 — pipeline sweep)** — No commit (same)
4. **Task 4 (Step 3 — post-sweep verifier)** — No commit (transcript captured for SUMMARY)
5. **Task 5 (Step 4 — human-verify checkpoint)** — Auto-approved (yolo mode); spot-checks confirmed numbers plausible
6. **Task 6 (Step 5 — forward-contract + offline regression + SUMMARY)** — `<filled-by-final-commit>` (docs(07-03): full-sweep COMPLETE …)

## Files Created/Modified

**Created (committed):**
- `.planning/phases/07-full-5-tier-rerun/07-03-SUMMARY.md` — this file
- `.planning/phases/07-full-5-tier-rerun/07-03-VERIFY.txt` — verbatim verifier transcript

**Created (gitignored runtime artifacts):**
- `evaluation/results/queries/tier-{1..5}-2026-05-07T10_59_10Z.json` — 5 fresh per-tier QueryLogs at sweep_sha
- `evaluation/results/queries/tier-4-2026-05-07T10_48_53Z.json` — Step 1 Tier 4 live capture (eval_capture.py)
- `evaluation/results/metrics/tier-{1..5}-2026-05-07T10_59_10Z.json` — 5 fresh per-tier metrics
- `evaluation/results/costs/tier-{1..5}-eval-20260507T1[01]*.json` — 5 fresh capture cost ledgers
- `evaluation/results/costs/ragas-judge-tier-{1..5}-20260507T1[12]*.json` — 5 fresh judge cost ledgers
- `evaluation/results/comparison.md` — regenerated rollup with 5-tier rollup + per-class breakdown + Embedder-by-tier table

**Modified:** None (source files RAW-LOCKED; verified via 0-bytes diff in PART A of Task 6).

## Decisions Made

1. **Embedder-by-tier verifier regex mismatch** (Rule 3 deviation): The plan's verifier uses regex `^## Embedder by tier` but `compare.py:278` actually emits `**Embedder by tier:**` (markdown bold-italic, not heading). The table IS correctly emitted with all 5 rows; the verifier check itself was wrong. Per RAW-LOCK constraint, I adjusted the verifier regex to `^\*\*Embedder by tier:\*\*` rather than touching `compare.py`. The output table is unchanged and correct.

2. **Pipeline retry on transient Gemini 429** (Rule 3 deviation): The first pipeline.py invocation failed at Tier 2 query 25/30 (multimodal-005) with `google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'message': 'Failed to embed content.'}}`. This is a transient cloud-quota error, not a code defect. I reran pipeline.py without making any commits. HEAD remained at `75f6f1b`, so sweep_sha stayed identical between attempts; the second invocation completed successfully with all 5 tier captures landing at sweep_ts `2026-05-07T10:59:10Z`. The single-SHA invariant was preserved per the plan's load-bearing invariant 2 ("no commits between Step 1 and Step 2"). The wasted ~$0.034 from the first-attempt Tier 1 completion is included in honest cost accounting.

3. **Honest cost-window selection for SUMMARY** (Rule 1 fix): The plan's verifier uses `ls -t | head -1` for cost JSON freshness, which got polluted by post-sweep pytest test bleed-through (test wrote 5 empty placeholder cost JSONs at `tier-N-eval-20260507T120542Z.json` — 197 bytes each, $0 — because of the pre-existing `shared.cost_tracker.DEFAULT_COSTS_DIR` hardcoded-path mis-feature documented in Plan 02-04). The verifier's $0 result trivially PASSES SC-5 ($3 ceiling), but for honest cost provenance in this SUMMARY I computed the total from the actual sweep-window cost JSONs (timestamps `10:59:10Z` to `12:02:53Z`), giving $0.118 (sweep) + $0.287 (Step 1) + $0.034 (failed first-attempt Tier 1) = $0.439 total.

4. **Auto-approved Task 5 human-verify checkpoint** (yolo + auto mode): All 7 spot-checks passed programmatically:
   - Tier 4/5 rollup show non-trivial scores (T4 0.791/0.263/0.238; T5 0.768/0.688/0.252 — NOT all-NaN; Phase 1+2 fixes stuck)
   - Tier 1/2/3 within prior-run ranges
   - Embedder-by-tier table 5 correct rows: T1/T3/T4/T5 = openai/text-embedding-3-small + openrouter; T2 = gemini-embedding-001 + google-managed
   - Tier 4 graph stats 28597 nodes (>>2886 smoke leftover threshold)
   - Tier 5 first context starts with `[paper_id=2005.11401]` (Phase 1 D-FORMAT confirmed)
   - Tier 4 first 3 records have n_ctx > 0
   - Verifier transcript ends with `OVERALL VERDICT: PASS`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adjusted Embedder-by-tier verifier regex to match actual emit pattern**
- **Found during:** Task 3 (Step 2 sweep verification)
- **Issue:** Plan's verifier used `grep -E '^## Embedder by tier'` but `compare.py:278` emits `**Embedder by tier:**` (bold markdown, not heading). RAW-LOCK forbids changing compare.py. Without the fix, Step 2 verification would have failed despite the table being correctly emitted.
- **Fix:** Changed verifier regex to `^\*\*Embedder by tier:\*\*` (matches actual emit). Reflects reality, not a wishful spec.
- **Files modified:** None (the change was in the inline verifier shell block, captured in 07-03-VERIFY.txt).
- **Verification:** Verifier transcript shows "Embedder-by-tier table: present (matched **Embedder by tier:** pattern)".

**2. [Rule 3 - Blocking] Auto-retried pipeline.py after transient Gemini 429 RESOURCE_EXHAUSTED**
- **Found during:** Task 3 (Step 2 first attempt)
- **Issue:** First `pipeline.py` invocation crashed at Tier 2 question 25/30 with Gemini 429 on embedding API. Stack trace surfaced via `google.genai.errors.ClientError`. Transient cloud-quota error, not a code defect.
- **Fix:** Reran `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache <Step 1 cache> --yes` without making any commits. HEAD stayed at `75f6f1b`, so sweep_sha would be identical. Second attempt succeeded fully.
- **Files modified:** None (pipeline.py is RAW-LOCKED; rerun is the prescribed recovery per plan's "Failure Modes" section).
- **Verification:** All 5 tier captures completed at sweep_ts `2026-05-07T10:59:10Z`; SHA equality across all 5 (verifier SC-2 PASS).

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes preserved RAW-LOCK and the single-SHA invariant. No scope creep. Original plan's verifier regex was a one-line documentation typo; pipeline retry is a documented recovery in plan's Failure Modes.

## Issues Encountered

1. **Pre-existing test-bleed cost JSON pollution.** Post-sweep pytest run (Task 6 Part B) wrote 5 empty placeholder cost JSONs at `tier-N-eval-20260507T120542Z.json` due to `shared.cost_tracker.DEFAULT_COSTS_DIR` being a hardcoded module-level path. This is a pre-existing v1.1 hardening item documented in Plan 02-04 SUMMARY and Plan 05-02 SUMMARY (`tracker.persist()` lacks a `dest_dir` arg). The pollution does NOT affect Plan 07-03's verdict — verifier SC-5 trivially passes ($0 ≤ $3.00) — but for honest provenance the SUMMARY computes spend from actual sweep-window cost JSONs.

2. **LiteLLM token-parser gap on RAGAS judge cost.** All 5 judge cost JSONs report $0 input/output tokens despite RAGAS clearly running and producing valid metrics. Pre-existing v1.1 hardening item documented in Plan 02-04 SUMMARY. Defensive `.totals.usd // 0` in verifier handles correctly. Estimated true judge spend (capture + judge mass-balanced): ~$0.45 — but verifier and SUMMARY honestly report what the cost JSONs say (zero), with this caveat noted.

3. **Tier 4 multimodal embedder cost in eval_capture.py was higher than estimated.** Step 1 (Tier 4 30q live capture via `eval_capture.py`) reported $0.287 vs the plan's $0.045 estimate. The discrepancy is the VLM-enhanced retrieval path (each multimodal-NNN question fans out into image-validation + fallback queries with 73 entities + 211 relations + 13 chunks per query), not a bug. Total Plan 07-03 spend ($0.439) is still well under the $3.00 ceiling and even under the $0.84 plan estimate. Plan 07-RESEARCH.md cost estimates can be tightened in v1.1.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Plan 07-03 wall time | ~75 min (incl. one transient retry) |
| Plan 07-03 spend (honest) | $0.439 (vs $3.00 ceiling) |
| Sweep wall time (Step 2 successful run) | ~64 min (10:59:10Z capture start → 12:02:55Z compare write) |
| Per-tier capture wall (avg) | ~60s (T1) to ~5 min (T4 multimodal) |
| Per-tier RAGAS judge wall (avg) | ~10-12 min |
| Tier 4 NaN reduction | 30/30 → 2/30 (93%) |
| Tier 5 NaN reduction | 30/30 → 0/30 (100% — perfect) |
| Forward-contract diff (9 files) | 0 bytes |
| Offline pytest baseline | 116 passed, 1 deselected (no regressions vs Plan 07-02) |
| Phase 7 cumulative spend | $25.32 (07-01 $0.00 + 07-02 $24.85 + 07-03 $0.44 incl. eval_capture) |

## User Setup Required

None — no external service configuration required. All sweep state is in gitignored `evaluation/results/` artifacts.

## Phase 8 Readiness Statement

**CAP-01 closed; Phase 7 ready for `/gsd-verify-phase`. Phase 8 (multi-judge spot-check) may proceed.**

The 5 fresh per-tier QueryLogs (`evaluation/results/queries/tier-{1..5}-2026-05-07T10_59_10Z.json`) all carrying sweep_sha=`75f6f1b` are the load-bearing input for Phase 8's multi-judge spot-check (run `/gsd-verifier` on Phase 7 first; on `verdict=passed` the orchestrator updates ROADMAP). Phase 9 (frozen handoff doc) will pure-copy from `evaluation/results/comparison.md` — verified via the 5-row rollup + 15-row per-class + 5-row Embedder-by-tier table (Phase 6 D-Q1 plumbing).

## Self-Check: PASSED

- 07-03-SUMMARY.md present
- 07-03-VERIFY.txt present (5 SC-N: PASS lines + OVERALL VERDICT: PASS)
- comparison.md regenerated
- 5/5 tier-N queries JSONs at sweep_ts=2026-05-07T10_59_10Z exist with 30 records each
- 5/5 tier-N metrics JSONs at sweep_ts=2026-05-07T10_59_10Z exist
- Forward-contract guard: 0 bytes diff on 9 RAW-LOCKED source files
- Offline pytest: 116 passed (no regressions vs Plan 07-02 baseline)

---
*Phase: 07-full-5-tier-rerun*
*Completed: 2026-05-07*
