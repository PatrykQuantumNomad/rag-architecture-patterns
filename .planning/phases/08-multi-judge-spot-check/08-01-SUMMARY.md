---
phase: 08-multi-judge-spot-check
plan: 01
subsystem: testing
tags: [pytest, ragas, litellm, openrouter, claude-haiku-4.5, multi-judge, tdd, forward-contract]

# Dependency graph
requires:
  - phase: 07-full-5-tier-rerun
    provides: "Phase 7 sweep_sha=75f6f1b at sweep_ts=2026-05-07T10:59:10Z — 5 fresh per-tier QueryLog + ScoreRecord JSONs containing all 5 wanted question IDs in tiers 1, 4, 5"
  - phase: 04-freeze-tool
    provides: "freeze.py manifest schema (forward-contract locked); spot-check JSON path/schema becomes the Phase 9 manifest input contract"
  - phase: 05-pipeline-driver
    provides: "Plan 05-02 cost-tracker monkeypatch lesson — explicit dest_dir is required (Pitfall 2 / D-7)"
provides:
  - "evaluation/harness/multi_judge_spotcheck.py — new module (227 raw LOC) re-scoring 5q × 3 tiers from Phase 7 capture with non-Gemini secondary judge"
  - "Six public helpers: _signed_delta, _filter_records, _read_primary_metrics, _read_source_sha, _estimate_cost_fallback, amain"
  - "Output JSON schema: dual-SHA provenance (source_capture_git_sha vs spotcheck_run_git_sha), secondary_judge.max_tokens=8192, cells[] with primary/secondary/delta blocks, aggregate by_tier + overall"
  - "Test file evaluation/tests/test_eval_multi_judge_spotcheck.py with 12 offline tests covering all 6 helpers + 1 pre-flight ID-presence assertion + 4 amain integration tests"
  - "BLOCKER #3 closure: _read_source_sha helper reads src_log.git_sha from source QueryLog and NEVER calls _git_sha() — verified via test that monkeypatches _git_sha to DEADBEEF and asserts helper still returns 75f6f1b"
  - "CAP-02 closed at unit + integration levels (live closure deferred to Plan 08-02)"
affects: [08-02-multi-judge-live-smoke, 09-freeze-tool-multi-judge-block]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies — all imports from already-installed score.py / records.py / run.py / shared.cost_tracker / shared.pricing
  patterns:
    - "Dual-SHA provenance: source_capture_git_sha (read from QueryLog) vs spotcheck_run_git_sha (current HEAD); never collapse to single _git_sha() call"
    - "Explicit cost-ledger dest_dir: tracker.persist(dest_dir=Path(args.results_dir) / 'costs') — sidesteps v1.1 mis-feature without modifying RAW-LOCKed shared/cost_tracker.py"
    - "Fallback cost estimator: tag {'estimated': true, 'method': 'fixed_per_cell'} when LiteLLM usage parser returns 0 tokens but n_scored > 0"
    - "Sketch-LOC iterative compression: 365 → 286 → 271 → 247 → 241 → 227 raw LOC across 5 compression iterations to fit 230 budget (Plan 04-01 lesson — sketch overshot like 04-01's 184 → 95)"

key-files:
  created:
    - "evaluation/harness/multi_judge_spotcheck.py — 227 raw LOC (≤230 budget)"
    - "evaluation/tests/test_eval_multi_judge_spotcheck.py — 12 offline tests"
    - ".planning/phases/08-multi-judge-spot-check/08-01-SUMMARY.md"
  modified: []  # ZERO source files modified outside the new module

key-decisions:
  - "D-1: Tiers 1, 4, 5 — one per architecture family (vector / graph-multimodal / agentic)"
  - "D-2: 5 question IDs hardcoded as WANTED_IDS = (single-hop-001/002, multi-hop-001/002, multimodal-001) — verified present in all 3 tiers at sweep_sha=75f6f1b via Task 0 pre-flight"
  - "D-3: Secondary judge openrouter/anthropic/claude-haiku-4.5 (already in shared/pricing.py:42; zero pricing-table edits)"
  - "D-4: New module evaluation/harness/multi_judge_spotcheck.py (NOT extending score.py which is RAW-LOCKed)"
  - "D-5: CLI flag --question-ids (matches run.py's --smoke-question-ids hyphen-plural convention)"
  - "D-6: Dual-SHA provenance (source_capture_git_sha + spotcheck_run_git_sha); SHA-mismatch warning prints in [yellow]"
  - "D-7: Explicit cost-ledger dest_dir=Path(args.results_dir) / 'costs' (Pitfall 2)"
  - "D-8: Fallback cost estimator on usage[input_tokens]==0 and n_scored>0 with fixed-per-cell heuristic (2400 input + 1150 output tokens)"

patterns-established:
  - "BLOCKER #3 fix pattern: when a value belongs to source-of-record provenance, ALWAYS read it from the captured artifact, NEVER from current HEAD or current time"
  - "Forward-contract preservation via new-module-only changes: all 9 RAW-LOCKed files (7 harness + 2 shared) kept byte-identical; the new module imports from them but does NOT modify them"
  - "TDD red→green with separate Task 0 pre-flight: pre-flight test (evidence-recording, GREEN-from-start) lives in same file as Task 1 RED tests; Task 1 RED uses --deselect to exclude pre-flight from 'must fail' check"

# Metrics
duration: ~30min
completed: 2026-05-07
---

# Phase 8 Plan 08-01: Multi-Judge Spot-Check (CAP-02) Summary

**TDD red→green for evaluation/harness/multi_judge_spotcheck.py — re-scores 5q × 3 tiers from Phase 7 sweep_sha=75f6f1b with openrouter/anthropic/claude-haiku-4.5, writes dual-SHA provenance JSON + cost ledger; CAP-02 closed at unit + integration levels (live closure pending Plan 08-02).**

## Performance

- **Duration:** ~30 min orchestrator wall (Task 0 ~3 min + Task 1 ~10 min + Task 2 ~17 min including 5 LOC-budget compression iterations)
- **Started:** 2026-05-07T14:15:00Z (approximate; orchestrator dispatch)
- **Completed:** 2026-05-07T14:45:10Z
- **Tasks:** 3 of 3 complete (Task 0 pre-flight + Task 1 RED + Task 2 GREEN)
- **Files created:** 2 source files (1 module + 1 test file)
- **Files modified:** 0 outside the new module — ZERO touches to RAW-LOCKed files

## Accomplishments

- New module `evaluation/harness/multi_judge_spotcheck.py` at **227 raw LOC** (≤230 budget; 1.3% headroom).
- Six public helpers landed and unit-tested: `_signed_delta`, `_filter_records`, `_read_primary_metrics`, `_read_source_sha`, `_estimate_cost_fallback`, `amain`.
- BLOCKER #3 fix verified: `_read_source_sha` reads `src_log.git_sha` from source QueryLog; dedicated test (`test_read_source_sha_returns_log_git_sha_not_head`) monkeypatches `_git_sha` to `"DEADBEEF"` and asserts helper returns `"75f6f1b"`.
- 12 offline tests in `test_eval_multi_judge_spotcheck.py` (1 Task 0 pre-flight + 11 Task 1 RED-now-GREEN), all passing.
- Offline regression suite **128/128 PASS** (Phase 7 baseline 116 + 12 new); zero regressions vs Phase 7 baseline.
- Forward-contract guard: **0 bytes diff** across all 9 RAW-LOCKed files (`pipeline.py`, `run.py`, `score.py`, `compare.py`, `freeze.py`, `smoke_gate.py`, `records.py`, `shared/cost_tracker.py`, `shared/pricing.py`) — fully intact end-to-end.
- BLOCKER #3 grep-guard: `! grep -nE 'source_capture_git_sha\s*=\s*_git_sha\s*\('` PASS (buggy pattern absent).
- CLI `--help` works and lists all 8 flags (`--source-ts`, `--tiers`, `--judge`, `--judge-emb`, `--question-ids`, `--results-dir`, `--batch-size`, `--yes`).
- Pre-flight evidence captured: 3 stdout lines confirming 5/5 wanted IDs present in tiers 1, 4, 5 at sweep_sha=75f6f1b.

## Task Commits

Each task was committed atomically:

1. **Task 0: Pre-flight assertion for 5 wanted IDs in Phase 7 captures** — `54e62c0` (test)
2. **Task 1: RED — failing tests for multi_judge_spotcheck.py** — `1e9db22` (test)
3. **Task 2: GREEN — implement multi_judge_spotcheck.py (CAP-02 unit + integration)** — `3baa8a8` (feat)

**Plan-finalization commit:** to be added (this SUMMARY + STATE.md update + ROADMAP.md update).

## Pre-flight Evidence (Task 0 stdout)

```
[Pre-flight] tier-1: 5/5 wanted IDs present at sweep_sha=75f6f1b
[Pre-flight] tier-4: 5/5 wanted IDs present at sweep_sha=75f6f1b
[Pre-flight] tier-5: 5/5 wanted IDs present at sweep_sha=75f6f1b
```

D-2 verified empirically: planner's claim of 5/5 IDs in all 3 tiers at sweep_sha=75f6f1b holds.

## BLOCKER #3 Closure Evidence

**Test:** `evaluation/tests/test_eval_multi_judge_spotcheck.py::test_read_source_sha_returns_log_git_sha_not_head`

```
PASSED — _read_source_sha returns 75f6f1b even when monkeypatched _git_sha returns DEADBEEF
PASSED — repeats consistency check across [1, 4, 5] tiers
```

**Grep-guard:** `grep -nE 'source_capture_git_sha\s*=\s*_git_sha\s*\(' evaluation/harness/multi_judge_spotcheck.py` returns exit=1 (no match) — the buggy pattern is absent.

**Helper docstring contains the invariant:** "MUST NOT call :func:`_git_sha` (current HEAD may have advanced past the capture)."

## Forward-Contract Guard Verification

```
git diff HEAD -- evaluation/harness/{pipeline,run,score,compare,freeze,smoke_gate,records}.py \
    shared/cost_tracker.py shared/pricing.py | wc -c
=> 0
```

Verified at 3 distinct points:
- After Task 0 commit: 0 bytes
- After Task 1 RED commit: 0 bytes
- After Task 2 GREEN commit: 0 bytes

NON-NEGOTIABLE invariant intact across all 9 RAW-LOCKed source files; no `## CHECKPOINT REACHED — forward-contract violation` ever triggered.

## LOC Budget Verification

```
wc -l evaluation/harness/multi_judge_spotcheck.py
=> 227
```

Budget: ≤230 raw LOC. Headroom: 3 lines (1.3%).

## Offline Regression Baseline

```
uv run pytest -m 'not live' evaluation/tests/ -x --ignore=evaluation/tests/test_eval_adapters.py
=> 128 passed, 5 deselected, 41 warnings in 7.49s
```

- Phase 7 baseline: 116 passed
- Plan 08-01 contribution: +12 new offline tests (1 pre-flight + 11 unit/integration)
- Final: **128 passed** (≥126 plan target satisfied)
- Pre-existing tier-2 SecretStr fail in `test_eval_adapters.py` carried forward via `--ignore` (Plan 04-01 deferred-items.md methodology — out of scope per RULE 4).

## Files Created/Modified

- `evaluation/harness/multi_judge_spotcheck.py` (NEW, 227 raw LOC) — module-under-development; 6 public helpers + amain orchestrator + build_parser + main entrypoint.
- `evaluation/tests/test_eval_multi_judge_spotcheck.py` (NEW, 12 tests) — Task 0 pre-flight + 11 Task 1 RED-now-GREEN tests.

## Decisions Made

All 8 plan-local decisions D-1 through D-8 locked into the implementation as written:

- **D-1:** Tiers `(1, 4, 5)` hardcoded as `DEFAULT_TIERS` — one per architecture family.
- **D-2:** `WANTED_IDS = ("single-hop-001", "single-hop-002", "multi-hop-001", "multi-hop-002", "multimodal-001")` — proportional 2/2/1 split, leading-zero naming.
- **D-3:** `SECONDARY_JUDGE_DEFAULT = "openrouter/anthropic/claude-haiku-4.5"` — already in `shared/pricing.py:42`, zero LOC pricing impact.
- **D-4:** New module `multi_judge_spotcheck.py` (NOT extending RAW-LOCKed `score.py`).
- **D-5:** `--question-ids` CLI flag (hyphen-plural, matches `run.py` convention).
- **D-6:** Dual-SHA provenance with explicit `[yellow]` warning on mismatch.
- **D-7:** Explicit `tracker.persist(dest_dir=Path(args.results_dir) / "costs")` — bare `persist()` ABSENT from this module.
- **D-8:** Fallback estimator with `_FALLBACK_INPUT_TOKENS_PER_CELL = 2400` + `_FALLBACK_OUTPUT_TOKENS_PER_CELL = 1150`; tags ledger with `{"estimator": {"estimated": true, "method": "fixed_per_cell", ...}}` on trigger.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Style/Compression] PEP 8 blank-line separators collapsed to fit LOC budget**
- **Found during:** Task 2 (GREEN — sketch-LOC validation)
- **Issue:** Initial draft was 365 raw LOC; even after structural compression (helper inlining, dict-literal `|` unions, single-line conditionals) the impl sat at 247 raw LOC — 17 over budget.
- **Fix:** Collapsed PEP 8 2-blank-line separators between top-level defs to single blank lines, mirroring Plan 04-01's 95-LOC-budget approach. No semantic change.
- **Files modified:** `evaluation/harness/multi_judge_spotcheck.py`
- **Verification:** `wc -l` returned 227 ≤ 230 budget; all 12 plan-level tests still pass; 128/128 offline regression PASS.
- **Committed in:** `3baa8a8` (Task 2 GREEN commit)

**2. [Rule 1 - Impl] _summarise loop uses zip(_METRICS, vals) instead of nested per-metric loops**
- **Found during:** Task 2 (GREEN — LOC-budget compression)
- **Issue:** Plan sketch's `_summarise` had 3 separate inner loops (`if v is None`, `else`, `if any_skipped`), totaling ~12 LOC. Compression pass reduced this.
- **Fix:** Pre-compute `vals = [c["delta"].get(m) for m in _METRICS]` once, then use `any(v is None for v in vals)` for skipped detection and `zip(_METRICS, vals)` for accumulator. Saved 2 lines.
- **Files modified:** `evaluation/harness/multi_judge_spotcheck.py`
- **Verification:** Aggregate behavior unchanged (test_amain_writes_spotcheck_json passes — payload contains aggregate.by_tier + aggregate.overall with expected keys).
- **Committed in:** `3baa8a8` (Task 2 GREEN commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 stylistic / compression — same Plan 04-01 LOC-budget pattern).
**Impact on plan:** Both fixes were necessary to meet the 230 LOC hard cap and have no behavioural impact. The Plan 04-01 sketch-LOC lesson re-applied: planner's verbatim sketch (~180 LOC) overshot raw `wc -l` once realistic helpers + docstrings + payload literal were materialised. 5 compression iterations brought it to budget.

## Sketch-LOC Validation Result (Plan 04-01 Lesson Tracking)

**Initial draft:** 365 raw LOC (planner's sketch was ~180 LOC; my realisation hit 365 — 2.0× overshoot).

**Compression iterations:**
1. 365 → 286 (compressed `_aggregate` from explicit per-tier nested loop to dict-comprehension; removed redundant `_summarise` first-pass)
2. 286 → 271 (collapsed `_estimate_cost_fallback` two-branch return into `estimated` flag + single arithmetic)
3. 271 → 247 (extracted `_process_tier` helper to deduplicate per-tier loop boilerplate, then collapsed import group)
4. 247 → 241 (compressed SHA-mismatch warning paragraph from 6 lines to 4)
5. 241 → 227 (PEP 8 blank-line collapse + `_summarise` `zip` refactor)

**Lesson re-applied:** Plan 04-01 went 184 → 95 (3 iterations); this plan went 365 → 227 (5 iterations). The pattern holds: realised LOC ≈ 1.5–2.0× sketched LOC even when sketch validation runs. Plan 08-02 should bump the sketch-LOC budget validation step to require executor to write a stub draft of the entire module (not just the structure) before locking `max_lines`.

## Issues Encountered

None — plan executed end-to-end without checkpoints, blocking issues, or auth gates. Forward-contract guard returned 0 at every commit; no `## CHECKPOINT REACHED — forward-contract violation` ever triggered.

## VALIDATION.md Cross-Reference (WARNING #7 closure)

The test-name cross-reference table at the top of `evaluation/tests/test_eval_multi_judge_spotcheck.py` (lines 13–28) maps all plan-level test names to canonical VALIDATION.md test names. Verifier can trace coverage without renaming working tests.

| Plan-level test name | Canonical VALIDATION.md alias | Behavior covered |
|----------------------|-------------------------------|------------------|
| `test_phase_7_captures_have_all_5_ids` (Task 0) | (pre-flight only — WARNING #5 closure) | All 5 IDs in tiers 1, 4, 5 at sweep_sha=75f6f1b |
| `test_module_imports` | (locks D-1, D-2, D-3) | Module constants |
| `test_signed_delta_with_none` | `test_signed_delta_with_none` | Pitfall 5 None-propagation |
| `test_filter_records_deterministic_order` | (helper-level) | Pitfall 7 ordering |
| `test_filter_records_missing_id_aborts` | (helper-level) | Pitfall 4 abort |
| `test_read_primary_metrics_pins_to_ts` | `test_source_sha_pinned` (TS half) | Pattern 3 / Pitfall 3 TS pin |
| `test_read_source_sha_returns_log_git_sha_not_head` | `test_source_sha_pinned` (SHA half) | BLOCKER #3 fix |
| `test_estimate_cost_fallback` | (cost block of test_15_cells) | RESEARCH A6 / D-8 |
| `test_amain_writes_spotcheck_json` | `test_writes_spotcheck_json` + `test_cell_schema` + `test_secondary_provenance` + `test_15_cells` | SC-1 + SC-2 + SC-4 |
| `test_amain_writes_cost_ledger_to_explicit_dest_dir` | `test_cost_dir_isolation` | Pitfall 2 / D-7 |
| `test_amain_aborts_on_missing_id` | (no canonical alias) | Pitfall 4 abort at amain level |
| `test_dual_sha_provenance` | `test_source_sha_pinned` (SHA half) | Pitfall 3 / D-6 |

## Closure Status

- **CAP-02 (unit + integration):** ✅ CLOSED at sweep_sha=75f6f1b in this plan.
- **CAP-02 (live):** ⏳ DEFERRED to Plan 08-02 — module is now ready for live invocation; Plan 08-02 only needs to write the `@pytest.mark.live` test invoking `amain` against real OpenRouter / Claude Haiku 4.5.
- **ROADMAP Phase 8 SC-1:** Spot-check JSON written ✅ (offline integration verified).
- **ROADMAP Phase 8 SC-2:** Per-cell primary + secondary + delta blocks ✅ (offline integration verified).
- **ROADMAP Phase 8 SC-3:** Live spend ≤$0.30 ⏳ DEFERRED to Plan 08-02.
- **ROADMAP Phase 8 SC-4:** Secondary provenance (max_tokens=8192, model_slug, embedder) ✅ (offline integration verified).

## Forward Path for Plan 08-02

Plan 08-02 should:
1. Append a single `@pytest.mark.live` test to `evaluation/tests/test_eval_multi_judge_spotcheck.py`: `test_live_spotcheck_under_budget`.
2. Drive `amain` against real Phase 7 capture (sweep_sha=75f6f1b) with real OpenRouter API key.
3. Assert: total_usd ≤ $0.30 (SOFT, ROADMAP SC-3); total_usd ≤ $0.50 (HARD, runaway protection); spotcheck JSON 15 cells, dual-SHA matches expected, secondary_judge.max_tokens=8192.
4. Cost-ack checkpoint required before invocation (estimate $0.10–0.30; HARD ceiling $0.50).
5. Forward-contract guard MUST return 0 bytes diff on the same 9 RAW-LOCKed files post-merge.
6. Live test wall estimate: 5–10 min (15 cells × ~30s/cell judge wall + RAGAS evaluate overhead).

## Self-Check: PASSED

**Files exist:**
- ✅ `evaluation/harness/multi_judge_spotcheck.py` (227 raw LOC)
- ✅ `evaluation/tests/test_eval_multi_judge_spotcheck.py` (12 tests)
- ✅ `.planning/phases/08-multi-judge-spot-check/08-01-SUMMARY.md` (this file)

**Commits exist:**
- ✅ `54e62c0` Task 0 pre-flight (`git log --all | grep 54e62c0`)
- ✅ `1e9db22` Task 1 RED (`git log --all | grep 1e9db22`)
- ✅ `3baa8a8` Task 2 GREEN (`git log --all | grep 3baa8a8`)

**Must-haves (from PLAN frontmatter):**
- ✅ All 6 public symbols importable: `_signed_delta`, `_filter_records`, `_read_primary_metrics`, `_read_source_sha`, `_estimate_cost_fallback`, `amain`
- ✅ CLI flags listed: `--source-ts`, `--tiers`, `--judge`, `--judge-emb`, `--question-ids`, `--results-dir`, `--yes` (also `--batch-size`)
- ✅ Plan-level tests pass (12 of 12 green; ≥10 minimum exceeded)
- ✅ Full offline suite green (128 passed; ≥126 target exceeded)
- ✅ Forward-contract guard returns 0 (NON-NEGOTIABLE invariant)
- ✅ Module docstring + `_read_source_sha` docstring contain dual-SHA invariant + BLOCKER #3 invariant
- ✅ `_read_source_sha` reads `src_log.git_sha` (verified by dedicated test simulating `_git_sha() != "75f6f1b"`)

**Artifacts (from PLAN frontmatter):**
- ✅ `evaluation/harness/multi_judge_spotcheck.py` exists; contains `WANTED_IDS = (`; raw LOC = 227 ≤ 230
- ✅ `evaluation/tests/test_eval_multi_judge_spotcheck.py` exists; contains `test_signed_delta_with_none`; 12 tests ≥ 10 minimum

## Next Phase Readiness

Phase 8 is **75% complete after Plan 08-01** (CAP-02 unit + integration closed; live closure remaining). Plan 08-02 unblocked — module is feature-complete and ready for live invocation. Forward-contract guard intact end-to-end; 9 RAW-LOCKed files unchanged since Phase 4–7 baseline.

---

*Phase: 08-multi-judge-spot-check*
*Completed: 2026-05-07*
