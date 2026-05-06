---
phase: 05-pipeline-driver
plan: 01
subsystem: evaluation-harness
tags: [pipeline, composition, harn-01, harn-02, tdd]
requires:
  - HARN-03
  - HARN-04
  - "Plan 04-01 freeze() forward-contract: freeze(version, force, results_dir, source) -> Path"
provides:
  - HARN-01
  - HARN-02
  - "evaluation/harness/pipeline.py — in-process composition driver"
  - "run.py kwarg plumb-through (git_sha_override / ts_override) for sweep-level identity propagation"
affects:
  - "evaluation/harness/run.py — backwards-compat additive kwargs on _capture_tier + amain"
  - "evaluation/tests/test_eval_run.py — fake_capture_tier signatures gain **kw to absorb new kwargs (deviation Rule 3)"
tech-stack:
  added: []
  patterns:
    - "In-process composition (no subprocess/os.system/os.popen) via direct function calls"
    - "asyncio.to_thread wrapping for sync stages (compare._run, freeze.freeze) called from async pipeline.amain"
    - "Single asyncio.run boundary at top of main() — Pitfall 5 of run.py docstring"
    - "argparse.Namespace synthesis per stage (run / score / compare) with always-yes=True to suppress inner prompts"
    - "Sweep-level git SHA + ISO timestamp captured ONCE at pipeline.amain entry; threaded into run.amain via getattr-defaulted kwargs"
key-files:
  created:
    - evaluation/harness/pipeline.py
    - evaluation/tests/test_eval_pipeline.py
    - .planning/phases/05-pipeline-driver/05-01-SUMMARY.md
  modified:
    - evaluation/harness/run.py
    - evaluation/tests/test_eval_run.py
decisions:
  - "D-Q1 (locked): targeted ~6-LOC additive change to run.py — _capture_tier accepts optional git_sha_override + ts_override kwargs; amain plumbs them via getattr(args, ...). Final actual diff: +10 raw insertions / -3 deletions (= cap)."
  - "D-Q2 (locked): defer --no-capture / --no-score / --no-compare flags to v1.1; pipeline.py implements only the 12 critical-path flags."
  - "D-Q3 (locked): do NOT expose --force on pipeline.py for freeze; user invokes `python -m evaluation.harness.freeze --force` directly when overwriting."
  - "D-Q4 (locked): compare stage operates on ALL 5 tiers regardless of --tiers (single-tier reruns produce a complete comparison.md)."
  - "D-LOC (locked): pipeline.py hard cap = 220 raw LOC. Final actual: 189 raw LOC (14% buffer; no compression iterations needed — sketch-validation per Plan 04-01 lesson worked first try)."
metrics:
  duration_minutes: ~12
  completed_date: 2026-05-06
  loc:
    pipeline_py_raw: 189
    pipeline_py_cap: 220
    run_py_diff_insertions: 10
    run_py_diff_cap: 10
  tests:
    pipeline_tests_added: 15
    pipeline_tests_passing: 15
    regression_baseline_pre: 88
    regression_baseline_post: 103
---

# Phase 5 Plan 01: Pipeline Driver Summary

Ship `evaluation/harness/pipeline.py` (189 raw LOC, ≤220 hard cap) — an in-process composition layer that runs capture → score → compare → freeze as one command with a single sweep-level git SHA + ISO timestamp captured at the top, single cost-surprise prompt, and HARN-02 single-tier-rerun semantics that fall out for free from existing `_latest()` mtime resolution. Closes HARN-01 + HARN-02 at unit + integration levels.

## What Shipped

### `evaluation/harness/pipeline.py` (NEW, 189 raw LOC)

- **Public API:** `build_parser()`, `async amain(args, console) -> int`, `main(argv) -> int` + private `_pipeline_cost_estimate`, `_build_run_args`, `_build_score_args`, `_build_compare_args`.
- **Composition:** `await run.amain(...)`, `await score.amain(...)`, `await asyncio.to_thread(compare._run, ...)`, `await asyncio.to_thread(freeze.freeze, ...)`. Single `asyncio.run` boundary in `main()`.
- **Sweep identity (HARN-01):** `sweep_sha = run._git_sha()` + `sweep_ts = run._ts()` captured at `amain` entry; banner via `console.rule(...)`; threaded into `run.amain` via the new `git_sha_override` + `ts_override` kwargs.
- **Cost-surprise consolidation (Success Criterion 4):** pipeline.py prompts ONCE if `not args.yes`; threads `args.yes=True` into both `run_args` and `score_args` Namespaces to suppress the inner prompts in `run.amain` (`_cost_surprise`) and `score.amain` (inline prompt).
- **Compare always-all-5 (D-Q4):** `_build_compare_args` hard-codes `tiers="1,2,3,4,5"` so single-tier reruns still produce a complete comparison.md row set.
- **Freeze refusal handling:** `try: await asyncio.to_thread(freeze.freeze, ...) except (FileExistsError, FileNotFoundError, RuntimeError) as e: print red ‘Freeze refused’ and return 2`.
- **12 argparse flags:** `--tiers` (default `1,2,3,4,5`), `--limit`, `--smoke-question-ids`, `--tier-4-from-cache`, `--results-dir` (default `evaluation/results`), `--judge-model` (default `score.JUDGE_LLM_SLUG_DEFAULT`), `--judge-emb` (default `score.JUDGE_EMB_SLUG_DEFAULT`), `--batch-size` (default 10), `--mode` (choices `naive|local|global|hybrid|mix`, default `hybrid`), `--tier1-k` (default 5), `--freeze` (default None — gates freeze stage), `--yes`.

### `evaluation/harness/run.py` (MODIFIED, +10 raw insertions / -3 deletions = +7 net LOC)

- `_capture_tier()` signature: keyword-only `git_sha_override: str | None = None`, `ts_override: str | None = None`. Body: `timestamp = ts_override or _ts()`, `git_sha = git_sha_override or _git_sha()`. Defaults preserve byte-identical behavior for existing direct-CLI callers (verified by `test_eval_run.py` 7/7 PASS).
- `amain()` (before tier loop): `sha_o = getattr(args, "git_sha_override", None)`, `ts_o = getattr(args, "ts_override", None)`. Tier loop call updated to forward the kwargs.
- Docstring augmented with one Plan 05-01 reference line.

### `evaluation/tests/test_eval_pipeline.py` (NEW, 15 offline tests)

| #  | Test                                              | Closes                       |
| -- | ------------------------------------------------- | ---------------------------- |
| 1  | `test_build_parser_help_exits_zero`               | argparse surface             |
| 2  | `test_build_parser_defaults`                      | locked argparse defaults     |
| 3  | `test_pipeline_cost_estimate`                     | cost-estimator math          |
| 4  | `test_sha_propagation`                            | **HARN-01 unit**             |
| 5  | `test_full_sweep_order`                           | HARN-01 sequence             |
| 6  | `test_single_cost_prompt`                         | SC4 consolidation            |
| 7  | `test_cost_prompt_abort`                          | SC4 abort path               |
| 8  | `test_stage_exit_propagation`                     | failure semantics            |
| 9  | `test_freeze_passthrough`                         | Phase 4 forward contract     |
| 10 | `test_freeze_refusal_handling`                    | freeze FileExistsError → 2   |
| 11 | `test_yes_flag_consolidation`                     | --yes propagation            |
| 12 | `test_tier_4_from_cache_passthrough`              | **HARN-02 unit**             |
| 13 | `test_no_subprocess_calls`                        | **SC3 invariant**            |
| 14 | `test_e2e_with_stub_stages`                       | HARN-01 integration          |
| 15 | `test_single_tier_rerun_preserves_others`         | **HARN-02 integration + W5** |

Test 15 (HARN-02 + WARNING W5 closure) builds a tmp `results_dir` with pre-existing `tier-{1,2,3,5}` queries/costs/metrics files (mtime backdated 1 day), monkeypatches `run.amain` + `score.amain` to write fresh tier-4 files containing the propagated `args.git_sha_override`, leaves `compare._run` REAL (not mocked), and asserts:
- (a) pipeline returns 0
- (b) every pre-existing file is byte-identical (`Path.read_bytes() == before[path]`)
- (c) regenerated `comparison.md` contains all 5 tier rows
- (e) `json.loads(new_tier4.read_text())["git_sha"] == sweep_sha` — proving the propagated SHA actually lands inside the written JSON on disk, not just on `args.Namespace` (closes WARNING W5 at integration level).

## Verification Results

### Functional gates
- All 15 pipeline tests **PASS** (`uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live'` → `15 passed in 2.77s`).
- Regression baseline **PASS**: 103 tests pass (Phase 4 baseline 88 + Plan 05-01's 15 = 103). Excluded per project convention: `test_eval_adapters.py` (pre-existing Tier 2 fail) + `test_eval_smoke_live.py` (live).

### CLI quality gate (6 sub-checks)
| # | Check                                                                         | Result                                                                |
| - | ----------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| a | `python -m evaluation.harness.pipeline --help` exits 0, lists all 12 flags    | **PASS** — usage + all flags rendered                                 |
| b | `python -m evaluation.harness.pipeline --tiers 99 --yes` exits 2              | **PASS** — banner + "Unsupported tier(s): [99]" + halt-at-capture msg |
| c | `from evaluation.harness.pipeline import amain, build_parser, main` callable | **PASS** — `True True True`                                           |
| d | `wc -l evaluation/harness/pipeline.py` ≤ 220                                  | **PASS** — 189 LOC (14% buffer)                                       |
| e | `git diff` of score.py / compare.py / freeze.py byte-count == 0               | **PASS** — 0 bytes diff (byte-identical guard intact)                 |
| f | `git diff --stat evaluation/harness/run.py` insertions ≤ 10                   | **PASS** — exactly 10 insertions / 3 deletions = +7 net               |

### No-subprocess invariant (Success Criterion 3)
- Test 13 (`test_no_subprocess_calls`) reads `pipeline.py` source as text and asserts `"subprocess" not in text and "os.system" not in text and "os.popen" not in text`.
- Shell-level guard `! grep -E 'subprocess|os\.system|os\.popen' evaluation/harness/pipeline.py` returns exit 0.
- pipeline.py composes via in-process function calls only.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking issue caused by current task] run.py kwarg added breaks two existing test_eval_run.py tests**

- **Found during:** Task 2 GREEN, after run.py edit landed.
- **Issue:** The plan's run.py kwarg plumb-through (D-Q1) calls `await _capture_tier(tier, qa, args, console, git_sha_override=sha_o, ts_override=ts_o)` from `amain`. But two existing tests in `test_eval_run.py` (`test_run_tier1_loop_persists_query_log`, `test_run_smoke_ids_filters_correctly`) use `monkeypatch.setattr(run_mod, "_capture_tier", fake_capture_tier)` with a 4-arg `(tier, qa, args, console)` signature that does NOT accept the new kwargs. Result: `TypeError: fake_capture_tier() got an unexpected keyword argument 'git_sha_override'`.
- **Fix:** Added `**kw` to both `fake_capture_tier` signatures in `test_eval_run.py` so they absorb the new kwargs harmlessly. The new `_capture_tier` signature is fully backwards-compatible at the runtime call site (defaults preserve old behavior); the test breakage was purely a stub-signature mismatch.
- **Files modified:** `evaluation/tests/test_eval_run.py` (2 stub functions).
- **Commit:** `aa1640c` (rolled into the GREEN commit; test_eval_run.py was strictly necessary for the GREEN verify gate to pass).
- **Rule justification:** Rule 3 — blocking issue *directly caused by* the current task's changes. The test file is not in `files_modified` per the plan, but the plan's own verification step ("Run `uv run pytest evaluation/tests/test_eval_run.py -m 'not live' -x` first") cannot pass without this fix. Scope is bounded: 2 lines, additive `**kw` only, no behavioral change to either test.

**2. [Rule 1 — Bug: docstring contained forbidden token] pipeline.py module docstring tripped Test 13's no-subprocess invariant**

- **Found during:** Task 2 GREEN, first pipeline-test run.
- **Issue:** Initial docstring read `"In-process composition (no subprocesses): ..."` — the plural `subprocesses` contains the substring `subprocess`, which Test 13 grep-matches as a forbidden token. The plan EXPLICITLY warns that Test 13 "checks raw substring containment, NOT just imports — pipeline.py must NEVER reference these tokens at all (no subprocess fallback paths, no commented-out `os.system` calls, no debug `subprocess.run` left behind)."
- **Fix:** Reworded the docstring sentence to `"In-process composition (one Python process; no shell-out): ..."` — same meaning, no forbidden tokens.
- **Files modified:** `evaluation/harness/pipeline.py` (1 line, docstring only).
- **Commit:** `aa1640c` (rolled into the GREEN commit).
- **Rule justification:** Rule 1 — fix needed for current task to pass its own RED test. Took ~30 seconds.

### No other deviations

- Sketch LOC validation per Plan 04-01 lesson worked first try: 189 raw LOC vs 220 hard cap — 14% buffer, zero compression iterations needed.
- Phase 4 forward-contract `freeze(version, force, results_dir, source) -> Path` honored verbatim; freeze.py byte-identical (0 bytes diff).
- Test 15 HARN-02 integration had to provide tier-4 cost file (not just queries) so `compare.aggregate_tier(4, ...)` would return a non-None row enabling tier-4 in the comparison markdown — minor fixture refinement, not a plan deviation.

## Risk Acceptance / Carry-Forward

### From Plan 03-03 / Plan 04-01 (still open, not addressed by Plan 05-01):

1. **Pre-existing `test_eval_adapters.py::test_run_tier2_extracts_grounding` failure on main (Tier 2 adapter)** — RULE 4 territory; NOT in `files_modified`; logged in Plan 04-01's `deferred-items.md`. Phase 5 baseline preserved by ignoring this file (per project convention).
2. **Pytest live-marker deselect-by-default not configured in pyproject.toml** — RULE 4; v1.1 hardening item carried since Plan 03-03. Plan 05-01 honored the workaround by always invoking `pytest -m 'not live'` explicitly.
3. **LiteLLM judge token-usage parser underreports** — RULE 4; v1.1 follow-up tracked since Plan 02-04. Out of scope for Plan 05-01 (no judge invocation in offline tests).

### New for Plan 05-01:

4. **`run._capture_tier` adapter-exception swallow** — when an adapter raises mid-question, `_capture_tier` returns None and `run.amain` prints yellow + continues to next tier (run.py:316-318). Pipeline.py inherits this: a partial-capture failure flows through to score stage's `_latest()` which may resolve to a stale prior file, masking the failure. Documented in Pitfall 2 of `05-RESEARCH.md`. Out of scope for v1.0; document and move on.
5. **Tier-4-from-cache silent-ignore** — if user passes `--tier-4-from-cache <path>` to pipeline.py BUT does not include tier 4 in `--tiers`, the flag is silently threaded into `run_args` but `run.amain` never reaches the tier-4 branch. Pipeline.py does NOT warn (could add a yellow notice in v1.1 if user testing demands it). Documented as Pitfall 8 of `05-RESEARCH.md`.

## Phase 5 Status

- **Plan 05-01 (Wave 1):** ✓ COMPLETE — HARN-01 + HARN-02 closed at unit + integration levels.
- **Plan 05-02 (Wave 2):** unblocked — live smoke backstop pending (~$0.05 OpenRouter spend, Tier 5 × 5q × full pipeline).

Phase 5 ships HARN-01 and HARN-02 at code-and-test level; the live-API closure of HARN-01 ("real OpenRouter spend through the full chain") is the explicit Plan 05-02 deliverable. Phase 5 unblocks Phase 7's full 5-tier rerun: `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` is the single command of record.

## Commits

- `3259ab1` — `test(05-01): RED — 15 pipeline pytest cases (all failing on missing module)`
- `aa1640c` — `feat(05-01): GREEN — pipeline.py + run.py kwarg plumb-through`

## Self-Check: PASSED

- Files created: `evaluation/harness/pipeline.py` (FOUND), `evaluation/tests/test_eval_pipeline.py` (FOUND).
- Files modified: `evaluation/harness/run.py` (FOUND, diff verified), `evaluation/tests/test_eval_run.py` (FOUND, diff verified).
- Commits 3259ab1 + aa1640c verified in `git log --oneline -5`.
- All 15 pipeline tests + 7 run.py tests + 103-test regression baseline GREEN.
- Byte-identical guards (score.py / compare.py / freeze.py) intact (0 bytes diff).
