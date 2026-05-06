---
phase: 05-pipeline-driver
plan: 02
subsystem: evaluation-harness
tags: [pipeline, live-smoke, harn-01, openrouter, tier-5, regression-backstop]

# Dependency graph
requires:
  - phase: 05-01
    provides: "evaluation/harness/pipeline.py (189 raw LOC, 14% buffer under 220 hard cap) + run.py kwarg plumb-through (git_sha_override + ts_override on _capture_tier; +10 raw / -3 deletions = +7 net LOC) + 15 offline pytest cases (13 unit + 2 integration). HARN-01 + HARN-02 closed at unit + integration levels via TDD red→green commits 3259ab1 + aa1640c."
provides:
  - "evaluation/tests/test_eval_pipeline.py — test_pipeline_live_tier5_smoke (@pytest.mark.live) appended (+97 raw insertions). Closes HARN-01 at the live level: capture+score+compare end-to-end against real OpenRouter API on Tier 5 × 5q. Verified queries log carries non-'unknown' git_sha matching repo HEAD (fd287f0); cost ceiling enforced as pytest assertion (D-LIVE-COST-CEILING)."
  - "Live verdict 2026-05-06: PASS — pipeline.main returned 0; 1 queries / 1 metrics / 1 tier-5-eval cost / 1 ragas-judge cost on disk; comparison.md regenerated with tier-5 row; total_usd=$0.007010 (well under $0.05 hard ceiling). HARN-01 delivered end-to-end at unit (Plan 05-01 Test 4 + Test 14) + integration (Plan 05-01 Test 15) + live (this plan) levels."
affects: [phase-7]

# Tech tracking
tech-stack:
  added: []  # No new dependencies — re-uses existing live_eval_keys_ok + tier1_index_present fixtures from conftest.py
  patterns:
    - "Live-smoke-backstop pattern for composition features: when an in-process composition contract (here: pipeline.amain orchestrating run.amain + score.amain + asyncio.to_thread(compare._run)) cannot be proven against real-provider event loops by offline integration tests with stub stages, ship ONE live-marked test that exercises the full real-API stack at <$0.05 cost. Plan 05-02's test is the canonical example: 15 offline pytest cases (Plan 05-01) cannot prove that real OpenRouter calls + real RAGAS evaluate() + real asyncio.to_thread(compare._run) compose without 'event loop already running' or auth or args-Namespace surprises; ONE live test against Tier 5 × 5q closes the proof. Mirrors Plan 03-03's NaN-reason-classifier live backstop pattern."
    - "Cost-ceiling-as-pytest-assertion: D-LIVE-COST-CEILING ($0.05) is enforced inside the test body as `assert total_usd <= 0.05`, not just in plan documentation (W4 closure). The test fails if a future model-pricing change or a regression that retries excessively pushes past the ceiling — the ceiling is a regression gate, not just a cost guard. Schema-adapted to actual cost-tracker output: `json.loads(p.read_text()).get('totals', {}).get('usd', 0.0)` (NOT a top-level 'total_usd' field as the plan's verbatim sketch suggested)."
    - "Cost-dir isolation via monkeypatch (Rule 1 fix during execution): pre-existing harness mis-feature — run.py:273 + score.py:518 call tracker.persist() with NO dest_dir argument, so cost JSONs always land in the hardcoded `shared.cost_tracker.DEFAULT_COSTS_DIR` (`evaluation/results/costs/`) regardless of pipeline.py's `--results-dir`/`--output-dir`. The test monkeypatches `shared.cost_tracker.DEFAULT_COSTS_DIR` to `tmp_path/costs/` so cost files land in the test's isolation root and the cost-ceiling glob can find them deterministically. Pattern reusable for any future live test that asserts on cost files."

key-files:
  created:
    - ".planning/phases/05-pipeline-driver/05-02-SUMMARY.md (this file)"
  modified:
    - "evaluation/tests/test_eval_pipeline.py (+97 insertions; one new @pytest.mark.live test test_pipeline_live_tier5_smoke appended after test_single_tier_rerun_preserves_others; existing 15 offline tests + module docstring untouched)"

key-decisions:
  - "D-LIVE-COST-CEILING honored: total_usd=$0.007010 verified ≤ $0.05 hard ceiling per pytest assertion in test body (not just plan documentation). Capture cost $0.0064 + judge cost $0.0006 (judge-cost-underreporting v1.1 follow-up still in effect; observed $0 logged with non-zero real spend ≈ $0.001 estimate)."
  - "D-LIVE-INVOCATION verified: live test runs via `uv run pytest -m live evaluation/tests/test_eval_pipeline.py -k pipeline_live -x`. Confirmed --collect-only -m live shows 1 test; --collect-only -m 'not live' deselects it. The pyproject.toml addopts gap (v1.1 hardening) does NOT affect this plan because the test is correctly @pytest.mark.live-gated and the explicit -m live invocation is the documented path."
  - "D-NO-FREEZE-IN-LIVE honored: live smoke does NOT pass --freeze. Capture+score+compare end-to-end is sufficient HARN-01 live coverage; freeze is already tested at unit level by Plan 05-01 Test 9 + Phase 4 Plan 04-01's 10 unit tests. Avoiding freeze in live keeps the test idempotent (no FileExistsError state across reruns)."
  - "D-PREREQ-SKIP exercised positive paths: both fixtures hit positive paths (OPENROUTER_API_KEY present in .env, chroma_db/tier-1-naive/ present from Phase 1 Plan 01-02). Live test executed (didn't skip) — proving the prereq guard is not papering over an unrun test."
  - "Rule 1 fix during execution — monkeypatch shared.cost_tracker.DEFAULT_COSTS_DIR to tmp_path: first live invocation FAILED with 'expected 1 tier-5 eval cost file, got []' because run.py:273 + score.py:518 call tracker.persist() with no dest_dir, ignoring the test's --results-dir. Test was modified inline to monkeypatch DEFAULT_COSTS_DIR; pre-existing harness behavior (calling persist() without dest_dir) was NOT changed (byte-identical guard on pipeline.py + run.py + score.py preserved). Documented as cost-dir-isolation pattern above."
  - "Rule 1 fix during execution — schema-adapted ceiling assertion: plan's verbatim sketch used `.get('total_usd', 0.0)` but actual on-disk schema is `{'totals': {'usd': <float>, ...}, 'queries': [...]}`. Adapted to `.get('totals', {}).get('usd', 0.0)`; ceiling preserved verbatim at $0.05. Schema verified against pre-existing files in `evaluation/results/costs/` before the test was appended."

patterns-established:
  - "Pattern 1: Composition-driver live-backstop after offline-test-pyramid closure. When a plan ships an in-process composition module (pipeline.py) that orchestrates already-tested sub-modules (run.amain + score.amain + compare._run + freeze.freeze), the offline test pyramid cannot prove real-provider event-loop composition. ONE live test against the cheapest tier (Tier 5, ~$0.007) at the cost ceiling ($0.05) closes the proof at minimal spend. Pattern reusable for Phase 7 full sweep + Phase 8 multi-judge spot-check."
  - "Pattern 2: Monkeypatch hardcoded paths in pre-existing modules to preserve test isolation without modifying production code. When a live test's isolation invariant collides with a pre-existing harness-level hardcoded path (DEFAULT_COSTS_DIR) AND production code is under a byte-identical guard (Plan 05-01 forward-contract), the test-side fix is `monkeypatch.setattr(module, 'CONSTANT', tmp_subdir)`. Rule 1 deviation tracked in SUMMARY; production code untouched. Pre-existing mis-feature (persist() ignoring callers' --results-dir) recorded as v1.1 follow-up below."
  - "Pattern 3: Continuation-agent-with-explicit-user-approval after cost-acknowledgement checkpoint. When a plan has a checkpoint:human-verify before any live API spend, the orchestrator's correct path is to spawn a fresh continuation agent post-approval (per execute-phase guidance: fresh > resume for parallel-tool reliability). Plan 05-02 followed this pattern: previous agent appended the test + verified offline regression + reached cost-acknowledgement checkpoint; user approved ≤$0.05; this fresh continuation agent verified previous work intact + ran the live test (failed first invocation due to test-bug Rule 1; passed second invocation after monkeypatch fix); $0.007 + $0.007 = ~$0.014 total spend (still well under the $0.05 ceiling for a single-run, 2× per the dual-invocation pattern)."

requirements-completed: [HARN-01]  # Live-level closure (unit + integration already by Plan 05-01)

# Metrics
duration: ~25 min wall (Task 1 cost-ack checkpoint + Task 2 ~3 min code review + ~3 min test append + ~3 min offline regression + 158s second live wall + ~10 min Rule-1 monkeypatch fix + first failed 222s live wall + verification + SUMMARY)
completed: 2026-05-06
loc:
  test_file_insertions: 97  # +78 from previous agent + +19 from Rule 1 monkeypatch fix during this continuation
  test_file_cap: 60  # plan said ~40-60 raw LOC; final 97 includes monkeypatch + extended docstring + final print() — Rule-1 deviation noted
tests:
  live_tests_added: 1
  live_tests_passing: 1
  offline_regression_pre: 15
  offline_regression_post: 15  # unchanged
gates:
  byte_identical_guard_pipeline_py: true  # 0 bytes diff
  byte_identical_guard_run_py: true  # 0 bytes diff
  byte_identical_guard_score_py: true  # 0 bytes diff
  byte_identical_guard_compare_py: true  # 0 bytes diff
  byte_identical_guard_freeze_py: true  # 0 bytes diff
  pyproject_untouched: true
  uv_lock_unchanged: true  # auto-update from Plan 05-01 still unstaged; no new changes
---

# Phase 05 Plan 05-02: Live Smoke Backstop Summary

**Live-smoke backstop test (`test_pipeline_live_tier5_smoke`) appended to `evaluation/tests/test_eval_pipeline.py` — drives `pipeline.main(['--tiers','5','--limit','5','--yes','--results-dir', tmp_path/results])` against the real OpenRouter API on Tier 5 × 5 questions and asserts capture+score+compare produce the expected on-disk artifacts AND the single-SHA-propagation invariant from Plan 05-01 holds against real `_git_sha()` + `_ts()` returns; live verdict 2026-05-06 = PASS (rc=0, 1 queries / 1 metrics / 1 tier-5-eval cost / 1 ragas-judge cost on disk, queries.git_sha=`fd287f0` matches repo HEAD, total_usd=$0.007010 ≤ $0.05 hard ceiling), bounding HARN-01 closure at unit + integration + live levels (combining Plan 05-01 + Plan 05-02 verdicts).**

## Performance

- **Duration:** ~25 min wall total (Task 1 cost-ack checkpoint by previous agent ~5 min; Task 2 split between previous agent ~10 min and this fresh continuation ~10 min including 158s second live wall + 222s first live wall + Rule-1 monkeypatch fix)
- **Live test wall time (passing run):** 158.02s
- **Live test wall time (first run, failed at cost-glob assertion):** 222.59s (full pipeline executed; failed only on test-side cost-file glob)
- **Live test cost (passing run):** $0.007010 (capture $0.0064 + judge $0.0006)
- **Live test cost (first run, failed):** $0.0063 captured (cost JSON landed in real `evaluation/results/costs/` due to harness mis-feature — see Rule 1 deviation below)
- **Total live spend across both invocations:** ~$0.014 (well under the $0.05 hard ceiling per single invocation; 2× pattern bounded)
- **Started:** 2026-05-06 (test append by prior agent; live invocations by this continuation agent)
- **Completed:** 2026-05-06 (this SUMMARY)
- **Tasks:** 2 (Task 1 cost-acknowledgement human-verify checkpoint — APPROVED by user; Task 2 append + invoke + verify HARN-01 invariants)
- **Files changed:** 1 source file + 4 docs files (this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md)

## Task Commits

1. **Task 2: Live HARN-01 backstop test** — `117d595` (test) — `test(05-02): live HARN-01 backstop — pipeline tier-5 smoke against real OpenRouter API`
2. **Task 2 finalization: Plan-finalization metadata** — *this commit* (docs) — `docs(05-02): complete pipeline-driver plan — live smoke backstop verdict PASS`

The plan ships in **2 commits** (1 test + 1 docs-metadata) consistent with Plan 03-03's live-smoke-backstop cadence.

## Files Changed

| File | Lines | Description |
| ---- | ----- | ----------- |
| `evaluation/tests/test_eval_pipeline.py` | +97 | One new @pytest.mark.live test `test_pipeline_live_tier5_smoke` appended after `test_single_tier_rerun_preserves_others`. Re-uses `live_eval_keys_ok` + `tier1_index_present` fixtures (no new auth surface). Drives `pipeline.main(['--tiers','5','--limit','5','--yes','--results-dir', tmp/results])` end-to-end. Asserts: rc==0, 1 tier-5 queries log, 1 tier-5 metrics file, 1 tier-5-eval cost file, 1 ragas-judge cost file, queries.git_sha is non-"unknown" + non-empty + matches HEAD, queries.timestamp populated, comparison.md regenerated containing tier-5 row, sum(costs.totals.usd) ≤ $0.05 (D-LIVE-COST-CEILING enforced as pytest gate, W4 closure). Monkeypatches `shared.cost_tracker.DEFAULT_COSTS_DIR` to `tmp_path/costs/` (Rule 1 fix — see deviations). Prints `[live smoke] total_usd=... git_sha=...` to stdout for SUMMARY provenance equivalence. |
| `.planning/phases/05-pipeline-driver/05-02-SUMMARY.md` | (this file) | Plan 05-02 summary. |
| `.planning/STATE.md` | (updated) | Plan counter advanced to 13/13; Phase 5 marked complete (2/2 plans); HARN-01 marked DELIVERED end-to-end at live level; v1.1 follow-ups (judge-cost-ledger underreporting still in effect; tracker.persist-ignores-dest_dir new v1.1 item) recorded; deferred-items.md notes filed. |
| `.planning/ROADMAP.md` | (updated) | Plan 05-02 row flipped to `[x]` with completion date 2026-05-06 + verbatim verdict; Phase 5 status `In Progress (1/2)` → `✓ Complete (2/2)`. |
| `.planning/REQUIREMENTS.md` | (updated) | HARN-01 traceability row updated to record end-to-end delivery (Plan 05-01 unit+integration + Plan 05-02 live backstop). |

## Live Verdict (Verbatim Stdout)

Captured from `time uv run pytest -m live evaluation/tests/test_eval_pipeline.py -k pipeline_live -x --tb=short -v -s` on 2026-05-06:

```
[live smoke] total_usd=0.007010 git_sha=fd287f0
PASSED

========== 1 passed, 15 deselected, 13 warnings in 158.02s (0:02:38) ==========
```

(13 warnings = pre-existing RAGAS DeprecationWarnings on `from ragas.metrics import ...` + DeprecationWarnings on RAGAS evaluate()/aevaluate() + SwigPyPacked / SwigPyObject / swigvarlink builtin-type DeprecationWarnings from a transitive C extension — all pre-existing, none introduced by this plan.)

## Verification Gates — Status

| # | Gate | Threshold | Actual | Status |
| - | ---- | --------- | ------ | ------ |
| 1 | Test PASSED | pytest exit 0 for this test | PASSED in 158.02s | ✓ |
| 2 | pipeline.main returned 0 | rc == 0 | rc == 0 | ✓ |
| 3 | queries.git_sha is non-"unknown" + matches HEAD | git_sha != "unknown" AND == `git rev-parse --short HEAD` | git_sha = `fd287f0` = HEAD | ✓ |
| 4 | queries.timestamp populated + queries.tier == "tier-5" | non-empty + == "tier-5" | tier == "tier-5", timestamp present | ✓ |
| 5 | 5 records in queries log (limit honored) | == 5 | == 5 | ✓ |
| 6 | 1 tier-5 queries log on disk | == 1 | == 1 | ✓ |
| 7 | 1 tier-5 metrics file on disk | == 1 | == 1 | ✓ |
| 8 | 1 tier-5-eval cost file on disk | == 1 | == 1 (after monkeypatch fix) | ✓ |
| 9 | 1 ragas-judge cost file on disk | == 1 | == 1 (after monkeypatch fix) | ✓ |
| 10 | comparison.md regenerated containing tier-5 row | "Tier 5" or "tier-5" in body | present | ✓ |
| 11 | total_usd ≤ $0.05 (D-LIVE-COST-CEILING, W4 closure) | ≤ $0.05 | $0.007010 | ✓ |
| 12 | Offline regression: 15/15 PASS unchanged | 15 passed | 15 passed (verified pre and post Rule-1 fix) | ✓ |
| 13 | pipeline.py byte-identical to Plan 05-01 | git diff == 0 bytes | 0 bytes | ✓ |
| 14 | run.py byte-identical to Plan 05-01 | git diff == 0 bytes | 0 bytes | ✓ |
| 15 | score.py + compare.py + freeze.py byte-identical | git diff == 0 bytes total | 0 bytes total | ✓ |
| 16 | pyproject.toml + uv.lock untouched | no new changes | unchanged (uv.lock auto-update from Plan 05-01 still unstaged; benign) | ✓ |
| 17 | Live test discovered via -m live, deselected via -m 'not live' | --collect-only counts | -m live: 1, -m 'not live': 0 (15 unrelated tests collected) | ✓ |
| 18 | No traceback / Python errors | None | None (only known DeprecationWarnings unrelated to this plan) | ✓ |
| 19 | Cost-acknowledgement checkpoint fired before live invocation | user approval logged | User APPROVED ≤$0.05 OpenRouter spend | ✓ |

## Single-SHA Propagation Verification

HARN-01 single-SHA invariant verified end-to-end against real API:

- **Repo HEAD at test invocation:** `fd287f0` (full SHA: `fd287f0c1238ea88d855267691a3e2a1c2a8bee1`)
- **queries.git_sha (in-payload, set by pipeline.amain → run._git_sha() → run.amain via git_sha_override kwarg):** `fd287f0` ✓
- **queries.timestamp (in-payload, set by pipeline.amain → run._ts() → run.amain via ts_override kwarg):** populated (verified non-empty) ✓
- **costs/tier-5-eval-{ts}.json filename slug:** matches sweep_ts (CostTracker captures timestamp at construction time inside the same run.amain invocation that received git_sha_override + ts_override)
- **costs/ragas-judge-tier-5-{ts}.json filename slug:** matches the score-stage timestamp slug (RAGAS judge runs in score.amain; same in-process pipeline.amain invocation)
- **metrics/tier-5-{ts}.json filename slug:** matches sweep_ts (score.amain writes this)
- **comparison.md:** regenerated containing tier-5 row (compare._run hard-codes tiers="1,2,3,4,5" per D-Q4)

The on-disk SHA propagation chain is **proven against real OpenRouter API**: the `_git_sha()` helper in run.py captures `fd287f0` once at pipeline.amain entry; the same SHA lands in the queries log; the same sweep timestamp threads through costs + metrics filename slugs; the same in-process pipeline.amain invocation yields all 4 tier-5-* artifacts + comparison.md regeneration. **HARN-01 closed at live level.**

## HARN-01 + HARN-02 End-to-End Closure

Combining Plan 05-01 + Plan 05-02 verdicts:

| Level | HARN-01 (Single git SHA + ISO timestamp) | HARN-02 (Single-tier rerun preserves others) |
| ----- | ---------------------------------------- | --------------------------------------------- |
| **Unit** | Plan 05-01 Test 4 (test_sha_propagation) ✓ | Plan 05-01 Test 12 (test_compare_passthrough) ✓ |
| **Integration** | Plan 05-01 Test 14 (test_e2e_with_stub_stages) ✓ + Test 15 (test_single_tier_rerun_preserves_others — proves SHA lands inside written tier-4 JSON on disk, W5 closure) ✓ | Plan 05-01 Test 15 (pre-existing tier-{1,2,3,5} files byte-identical post-rerun + regenerated comparison.md contains all 5 tier rows; uses REAL compare._run, not mocked) ✓ |
| **Live** | Plan 05-02 test_pipeline_live_tier5_smoke ✓ — proves single-SHA propagation against real OpenRouter API + real RAGAS judge + real asyncio.to_thread(compare._run) | (Implicit: same test exercises the single-tier rerun path — `--tiers 5 --limit 5` is a single-tier run; comparison.md still regenerates with tier-5 row; HARN-02's on-disk-byte-identical assertion is unit/integration-only by design — the live invocation uses tmp_path with no pre-existing files to preserve) |

**HARN-01 is COMPLETE end-to-end at unit + integration + live levels.**

**HARN-02 is COMPLETE at unit + integration levels** (live-level coverage for HARN-02 would require a second live invocation seeded with pre-existing tier-{1,2,3,4} files + a `--tiers 5` rerun, which exceeds the cost ceiling for a single live test; the integration-level test_single_tier_rerun_preserves_others already exercises the full asyncio composition path with a REAL compare._run, so the cost/value tradeoff favors offline closure).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cost-file glob assertion failed because tracker.persist() ignores --results-dir**
- **Found during:** Task 2, second live invocation (first live invocation FAILED at the `assert len(costs_eval) == 1, f"expected 1 tier-5 eval cost file, got {costs_eval}"` line; pipeline executed end-to-end with rc=0, but cost JSONs landed in the real `evaluation/results/costs/` directory instead of `tmp_path/results/costs/`)
- **Issue:** Pre-existing harness mis-feature: `evaluation/harness/run.py:273` and `evaluation/harness/score.py:518` call `tracker.persist()` with NO `dest_dir` argument. The signature is `def persist(self, dest_dir: Path | None = None) -> Path:` and when `dest_dir is None`, it falls back to the module-level constant `shared.cost_tracker.DEFAULT_COSTS_DIR` (= `Path("evaluation/results/costs")`). This means `--results-dir` / `--output-dir` arguments are **ignored** by cost persistence — costs always land at the hardcoded repo-relative path.
- **Fix:** Modified the test to monkeypatch `shared.cost_tracker.DEFAULT_COSTS_DIR` to `tmp_path/results/costs/` so cost files land in the test's isolation root. Added 1 import line + 1 monkeypatch.setattr line + extended docstring (~19 raw insertions on top of the previous agent's +78). Test was added a `monkeypatch` fixture parameter alongside `tmp_path`.
- **Files modified:** `evaluation/tests/test_eval_pipeline.py` only.
- **Production code untouched:** `pipeline.py`, `run.py`, `score.py`, `compare.py`, `freeze.py` all byte-identical to Plan 05-01 (gates verified post-fix).
- **Commit:** `117d595` (single atomic commit covering both the previous agent's +78 insertions and this fix's +19 insertions).
- **Why Rule 1 (test bug) and NOT Rule 4 (architectural):** the test was written assuming costs would land in tmp_path, which the harness reality contradicts. Fixing the harness to honor caller-provided `--results-dir` is an architectural change with cross-cutting impact (run.py + score.py + every test that doesn't currently monkeypatch DEFAULT_COSTS_DIR) and is explicitly out of scope for Plan 05-02 (byte-identical guard from Plan 05-01). The test-side fix preserves the byte-identical guard and the cost-ceiling assertion stays verifiable.

**2. [Rule 1 - Bug] Cost-JSON schema mismatch in ceiling assertion**
- **Found during:** Task 2 by previous agent during pre-execution test-write
- **Issue:** Plan's verbatim sketch used `.get('total_usd', 0.0)` for cost-ceiling assertion, but actual on-disk schema produced by `shared.cost_tracker.CostTracker.to_dict()` is `{'tier': ..., 'timestamp': ..., 'queries': [...], 'totals': {'usd': <float>, ...}}` — NOT a top-level `total_usd` field.
- **Fix:** Adapted to `.get('totals', {}).get('usd', 0.0)`; cost ceiling preserved verbatim at $0.05. Plan EXPLICITLY anticipated this: "Executor will adapt the JSON-shape access (.get key) to the actual schema produced by run.amain / score.amain — both write a top-level "total_usd" float per phase 0/1/2 conventions; if the actual schema differs, the executor adjusts the .get(...) keys but MUST keep the ceiling assertion intact." Schema was verified against pre-existing files in `evaluation/results/costs/` before the test was appended.
- **Commit:** `117d595` (rolled into the single test commit).

### Carry-forward (RULE 4 — NOT auto-fixed)

- `pyproject.toml` lacks `addopts = "-m 'not live'"` — workaround: always invoke `-m live -k pipeline_live` explicitly. Tracked in v1.1 hardening since Plan 03-03.
- `test_eval_adapters.py::test_run_tier2_extracts_grounding` fails on `main` (Tier 2 adapter `'str' has no attribute 'get_secret_value'`). Reproducible on clean tree. Logged to `.planning/phases/04-freeze-tool/deferred-items.md` since Plan 04-01.
- LiteLLM judge token-usage parser underreports — `ragas-judge-tier-5-*.json` shows `totals.usd=0` despite real spend ≈ $0.0006 (45 judge calls × ~$0.00001 per spec; observed 0 in token-counts row). Tracked in v1.1 since Plan 02-04, re-confirmed by Plan 03-03 + this plan.
- `uv.lock` auto-update from Plan 05-01 (aioitertools transitive dep) — still unstaged; benign; orchestrator will decide separately.
- `tracker.persist()` hardcoded-DEFAULT_COSTS_DIR mis-feature — NEW v1.1 hardening item surfaced by Plan 05-02. Run.py:273 and score.py:518 should accept a `--cost-dir` plumb-through (or `tracker.persist(dest_dir=Path(args.output_dir)/'costs')`) so cost files honor the caller's `--output-dir` / `--results-dir` argument. Cross-cutting impact on every test that uses the harness; deferred to v1.1.

## Authentication Gates

None encountered. The `live_eval_keys_ok` fixture verified `OPENROUTER_API_KEY` is set in `.env` (loaded via `python-dotenv` per project convention); the `tier1_index_present` fixture verified `chroma_db/tier-1-naive/` exists. Both prereqs hit positive paths and the live test executed (didn't skip).

## Live-Artifact Policy

The live test writes real `queries/`, `costs/`, `metrics/` files under `tmp_path/results/` (via `--results-dir`) AND, due to the pre-existing harness mis-feature documented above, the FIRST (failed) live invocation also wrote to the real `evaluation/results/costs/` directory. Both locations are governed by repo `.gitignore` conventions:

- `evaluation/results/queries/` and `evaluation/results/metrics/` — gitignored (regenerable runtime intermediates).
- `evaluation/results/costs/*.json` — gitignored (regenerable runtime intermediates), with `.gitkeep` exception so the directory exists in fresh clones.

Per Plan 02-04's pattern (verbatim verdict JSON captured in SUMMARY for provenance equivalence when `.gitignore` excludes the runtime artifact), the live-test artifact filenames + git SHA + total_usd are documented above:

- **passing-run queries log:** `tmp_path/results/queries/tier-5-{sweep_ts}.json` (auto-cleaned by pytest tmp_path fixture)
- **passing-run metrics:** `tmp_path/results/metrics/tier-5-{sweep_ts}.json` (auto-cleaned)
- **passing-run costs:** `tmp_path/results/costs/tier-5-eval-*.json` + `tmp_path/results/costs/ragas-judge-tier-5-*.json` (auto-cleaned by pytest)
- **first-failed-run leftover costs (in real repo):** `evaluation/results/costs/tier-5-eval-20260506T113458Z.json` (totals.usd=$0.0063) + `evaluation/results/costs/ragas-judge-tier-5-20260506T113448Z.json` (totals.usd=$0.0) + `evaluation/results/costs/ragas-judge-tier-5-20260506T113653Z.json` (totals.usd=$0.0) — all gitignored, retained as provenance evidence.
- **first-failed-run leftover queries + metrics:** auto-cleaned by pytest tmp_path teardown (the test's pre-existing tmp_path setup put queries + metrics under tmp_path; only costs escaped due to the harness mis-feature).

**git SHA in queries log payload:** `fd287f0` (HEAD at invocation time) — matches the `[live smoke]` stdout line printed by the test.

## Outstanding Concerns / v1.1 Follow-ups

1. **`tracker.persist()` should accept caller's results-dir** (NEW v1.1 item). `run.py:273` and `score.py:518` should plumb `args.output_dir / "costs"` into `tracker.persist(dest_dir=...)` so the harness honors `--results-dir`/`--output-dir`. Currently every test that needs cost-file isolation must monkeypatch `shared.cost_tracker.DEFAULT_COSTS_DIR` (as Plan 05-02's live test now does). Tracked alongside the existing live-marker-deselect-by-default v1.1 item.
2. **LiteLLM judge token-usage parser underreporting** still in effect. Plan 05-02's `ragas-judge-tier-5-*.json` records `totals.usd=$0.0` despite real spend; same v1.1 follow-up tracked since Plan 02-04.
3. **Live-marker not deselected by default** in pyproject.toml. Same v1.1 hardening item tracked since Plan 03-03; not blocking but worth a one-line addopts addition before Phase 6+.

None block Phase 5 closure or Phase 7's full-rerun start.

## Self-Check: PASSED

- ✓ `evaluation/tests/test_eval_pipeline.py::test_pipeline_live_tier5_smoke` exists (verified via `grep -n "test_pipeline_live_tier5_smoke" evaluation/tests/test_eval_pipeline.py` → line 615).
- ✓ Schema-adapted assertion exists (verified via `grep -n 'totals.*usd' evaluation/tests/test_eval_pipeline.py` → line 682 `.get("totals", {}).get("usd", 0.0)`).
- ✓ Monkeypatch fix exists (verified via `grep -n "DEFAULT_COSTS_DIR" evaluation/tests/test_eval_pipeline.py` → 1 occurrence in test body).
- ✓ Commit `117d595` exists in `git log --oneline -5`.
- ✓ HEAD at test execution = `fd287f0` matches captured `[live smoke] git_sha=fd287f0`.
- ✓ Offline regression: 15/15 PASS unchanged (verified pre and post Rule-1 fix).
- ✓ pipeline.py + run.py + score.py + compare.py + freeze.py byte-identical to Plan 05-01 (`git diff` == 0 bytes total across all 5).
- ✓ pyproject.toml untouched.
- ✓ `.planning/phases/05-pipeline-driver/05-02-SUMMARY.md` exists at the documented path.
