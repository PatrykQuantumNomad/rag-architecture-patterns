# Phase 5: Pipeline Driver — Validation Map

**Phase:** 05-pipeline-driver
**Created:** 2026-05-05
**Status:** Active

This document formalizes the test-requirement map for Phase 5. Every requirement
covered by this phase has at least one test (offline AND/OR live) that proves
the requirement is satisfied. Source of truth: `05-RESEARCH.md` § Validation
Architecture (lines 422-450) plus the test list in `05-01-PLAN.md` Task 1
`<behavior>` block.

---

## Phase Requirement Coverage

This phase covers two requirement IDs from `.planning/REQUIREMENTS.md`:

| Req ID | Scope in This Phase | Out-of-Scope (Other Phases) |
|--------|--------------------|-----------------------------|
| HARN-01 | `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5` runs capture → score → rollup → freeze in one command, with a SINGLE git SHA + ISO timestamp captured ONCE at the top of `pipeline.amain` (via `run._git_sha()` / `run._ts()`) and propagated into every per-tier output JSON via the new additive `git_sha_override` / `ts_override` kwargs on `run._capture_tier()`. Cost-surprise prompt fires EXACTLY ONCE per pipeline invocation (D-COST-EST: pipeline.py prompt fires; inner prompts on `run.amain` and `score.amain` are suppressed by threading `args.yes=True` into both synthesized Namespaces). Composition is in-process via `await run.amain(...)`, `await score.amain(...)`, `await asyncio.to_thread(compare._run, ...)`, `await asyncio.to_thread(freeze.freeze, ...)` — Success Criterion 3 (no subprocess / `os.system` / `os.popen` calls; pipeline.py composes via in-process function calls only). 12 argparse flags locked. Live backstop runs Tier 5 × 5q against real OpenRouter API at hard cost ceiling $0.05/run (D-LIVE-COST-CEILING; ~$0.02 expected). | TIER-01..05 / METH-01..02 (other ROADMAP requirement families). HARN-03 / HARN-04 / HARN-05 (other harness concerns; HARN-05 closed by Phase 3). Phase 6 embedder-provenance capture (orthogonal to pipeline.py per RESEARCH.md Pitfall 12). Phase 7 actual full 5-tier rerun (Phase 5 unblocks it; the rerun itself happens later). |
| HARN-02 | Single-tier rerun (e.g. `pipeline --tiers 4 --tier-4-from-cache <path>`) preserves other tiers' captures byte-identically on disk; the comparison stage re-renders ALL 5 tier rows in `comparison.md` regardless of `--tiers` (D-Q4: compare always operates on all 5 tiers). Backbone is `compare._latest()` + `score._latest_query_log()` mtime-DESC tier-isolation (verified by source reading; ZERO edits to score.py / compare.py). Validated via offline integration test 14 in `test_eval_pipeline.py`: pre-existing tier-{1,2,3,5} fixtures (older mtime via `os.utime`); pipeline rerun writes new tier-4; assertion (a) pre-existing files byte-identical, (b) new comparison.md contains all 5 tier rows, (c) tier-4 row reflects fresh data, (d) tier-{1,2,3,5} rows reflect their pre-existing files. | A live HARN-02 verification is intentionally NOT executed at this phase (the offline integration is the proof; the live smoke focuses on HARN-01 single-SHA propagation; HARN-02 at live level would require a 2-stage live invocation costing ~$0.10+ for marginal additional confidence). |

Requirements NOT covered by this phase (deferred or owned elsewhere):

- HARN-03 / HARN-04 (other harness fields) — out of scope; not on Phase 5 critical path.
- HARN-05 (NaN reason instrumentation) — already closed by Phase 3.
- METH-01 (bootstrap CI) / METH-02 (multi-judge spot-check) — Phase 8+.
- TIER-01..05 — phases 1, 2, 7 (rerun phases).

---

## Requirement → Test Map

| Req ID | Behavior Under Test | Test Type | Test Path | Plan Owning |
|--------|--------------------|-----------|-----------|-------------|
| HARN-01 (argparse `--help` exits 0) | `build_parser().parse_args(["--help"])` raises `SystemExit(0)`; help text lists all 12 flags | unit | `evaluation/tests/test_eval_pipeline.py::test_build_parser_help_exits_zero` | 05-01 (Test 1) |
| HARN-01 (argparse defaults locked) | Defaults match locked argparse surface in `05-01-PLAN.md` Task 2 step (j): `tiers="1,2,3,4,5"`, `limit=None`, `freeze=None`, `yes=False`, `mode="hybrid"`, `tier1_k=5`, `batch_size=10`, etc. | unit | `evaluation/tests/test_eval_pipeline.py::test_build_parser_defaults` | 05-01 (Test 2) |
| HARN-01 (cost-estimator math) | `_pipeline_cost_estimate([1,2,3], 10) == (capture, judge, total)` with `judge == pytest.approx(0.003 * 10 * 3)`; `total == capture + judge` | unit | `evaluation/tests/test_eval_pipeline.py::test_pipeline_cost_estimate` | 05-01 (Test 3) |
| HARN-01 (single SHA propagation to run.amain) | Synthesized `run_args.git_sha_override == sweep_sha`, `run_args.ts_override == sweep_ts` after pipeline.amain runs (with `_git_sha`/`_ts` monkeypatched to deterministic values) | unit | `evaluation/tests/test_eval_pipeline.py::test_sha_propagation` | 05-01 (Test 4) |
| HARN-01 (sweep order: capture → score → compare → freeze) | Recorded call order across mocked stages == `["run.amain", "score.amain", "compare._run", "freeze.freeze"]` | integration | `evaluation/tests/test_eval_pipeline.py::test_full_sweep_order` | 05-01 (Test 5) |
| HARN-01 (single cost-surprise prompt) | `input` called EXACTLY ONCE per invocation; `captured_run_args.yes is True` and `captured_score_args.yes is True` (suppresses inner prompts) | unit | `evaluation/tests/test_eval_pipeline.py::test_single_cost_prompt` | 05-01 (Test 6) |
| HARN-01 (cost-surprise abort path) | `input → "n"` causes pipeline to return 1; no stage is invoked | unit | `evaluation/tests/test_eval_pipeline.py::test_cost_prompt_abort` | 05-01 (Test 7) |
| HARN-01 (stage exit propagation) | `run.amain` returns 2 → pipeline returns 2; `score.amain` / `compare._run` / `freeze.freeze` NEVER called | unit | `evaluation/tests/test_eval_pipeline.py::test_stage_exit_propagation` | 05-01 (Test 8) |
| HARN-01 (freeze passthrough — locked Phase 4 contract) | `freeze.freeze` called EXACTLY ONCE with `version="v1.0"`, `force=False`, `results_dir=Path(...)`, `source=None` | unit | `evaluation/tests/test_eval_pipeline.py::test_freeze_passthrough` | 05-01 (Test 9) |
| HARN-01 (freeze refusal handling) | `freeze.freeze` raises `FileExistsError` → pipeline returns 2 (other stages succeeded; partial state survives) | unit | `evaluation/tests/test_eval_pipeline.py::test_freeze_refusal_handling` | 05-01 (Test 10) |
| HARN-01 (`--yes` flag consolidation) | `--yes` suppresses pipeline prompt AND threads `yes=True` into run_args + score_args | unit | `evaluation/tests/test_eval_pipeline.py::test_yes_flag_consolidation` | 05-01 (Test 11) |
| HARN-02 (tier-4-from-cache passthrough) | Synthesized `run_args.tier_4_from_cache == "/path/to/cache.json"` | unit | `evaluation/tests/test_eval_pipeline.py::test_tier_4_from_cache_passthrough` | 05-01 (Test 12) |
| HARN-01 (Success Criterion 3: no subprocess / in-process composition only) | `evaluation/harness/pipeline.py` source contains ZERO occurrences of `subprocess`, `os.system`, `os.popen` (composition is in-process function calls only) | unit | `evaluation/tests/test_eval_pipeline.py::test_no_subprocess_calls` | 05-01 (Test 13 — added per checker WARNING W3) |
| HARN-01 (E2E with stub stages) | All 4 stage marker files written; pipeline returns 0 for `--tiers 1 --freeze v1.0 --yes` | integration | `evaluation/tests/test_eval_pipeline.py::test_e2e_with_stub_stages` | 05-01 (Test 14) |
| HARN-02 (single-tier rerun preserves others; SHA lands in written JSON) | (a) pre-existing tier-{1,2,3,5} files byte-identical post-rerun; (b) new `comparison.md` contains all 5 tier rows; (c) tier-4 row reflects fresh data; (d) the freshly-written tier-4 JSON contains the propagated `git_sha == sweep_sha` value (proves SHA ends up on disk, not just on `args.Namespace` per checker WARNING W5) | integration | `evaluation/tests/test_eval_pipeline.py::test_single_tier_rerun_preserves_others` | 05-01 (Test 15 — extended per checker WARNING W5) |
| HARN-01 (live backstop: real OpenRouter API end-to-end) | `pipeline.main(['--tiers','5','--limit','5','--yes','--results-dir', tmp])` returns 0; queries log carries non-`"unknown"` git SHA matching `run._git_sha()`; comparison.md contains tier-5 row; `total_usd` summed across cost JSONs ≤ $0.05 (asserted in pytest body per checker WARNING W4) | live (`@pytest.mark.live`) | `evaluation/tests/test_eval_pipeline.py::test_pipeline_live_tier5_smoke` | 05-02 (Task 2) |

---

## Coverage Self-Check

Every Wave 0 gap from `05-RESEARCH.md § Wave 0 Gaps` (lines 447-450) has at
least one row in the table above. Cross-reference:

- [x] `evaluation/tests/test_eval_pipeline.py` — covers HARN-01 + HARN-02 + freeze-passthrough + cost-prompt + stage-gating across 15 offline tests (10 unit + 2 integration + 1 in-process-composition guard + 2 reserved live slots) + 1 live test owned by Plan 05-02.
- [x] No new framework deps — pytest 8.4 already pinned in `pyproject.toml [dependency-groups].test`.
- [x] HARN-01 covered at all three levels: unit (Tests 1-13), integration (Tests 14-15), live (Plan 05-02 Task 2).
- [x] HARN-02 covered at integration level (Test 15 — pre-existing-file byte-identicality + comparison.md all-5-rows + SHA-propagation-to-disk per WARNING W5).
- [x] Success Criterion 3 (no subprocess) covered at unit level (Test 13 per WARNING W3) AND grep verification gate in `05-01-PLAN.md` Task 2 `<verify>`.
- [x] Live cost ceiling ($0.05) enforced as pytest assertion in `test_pipeline_live_tier5_smoke` body (per WARNING W4) — not just plan documentation.

No requirement in this phase is uncovered.

---

## Sampling Rate / Nyquist Coverage

The Nyquist principle here is "sample at least twice the rate of the
behavior you want to detect". For Phase 5 that means: every observable
truth in `must_haves.truths` is covered by a verification command (offline
unit / offline integration / live smoke), AND every locked decision (D-Q1
through D-Q4, D-LOC, D-COST-EST, D-LIVE-*) is bound to a verification
gate either as (a) a pytest assertion, (b) a `wc -l` / `grep` shell
guard in the plan's `<verify>` block, or (c) a human-verify checkpoint
acknowledging cost.

| Stage | Command | Estimated Wall Time | API Cost |
|-------|---------|---------------------|----------|
| Per task commit (Plan 05-01 RED) | `uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live' --collect-only` | ≤ 3 seconds | $0 |
| Per task commit (Plan 05-01 GREEN) | `uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live' -x` | ≤ 15 seconds (15 offline tests) | $0 |
| Per wave merge | `uv run pytest evaluation/tests/ --ignore=evaluation/tests/test_eval_adapters.py --ignore=evaluation/tests/test_eval_smoke_live.py -m 'not live' -x` | ≤ 60 seconds | $0 |
| Phase gate (before `/gsd-verify-work`) | Full offline suite green PLUS `uv run pytest -m live evaluation/tests/test_eval_pipeline.py::test_pipeline_live_tier5_smoke -x -v` | ≤ 60 seconds (offline) + ≤ 120 seconds (live) | ≤ $0.05 per live run (D-LIVE-COST-CEILING; asserted as pytest gate per WARNING W4; ~$0.02 expected) |

---

## Per-Test Cost Ceiling for Live Tests

| Test | Hard Cost Ceiling | Expected Cost | Enforcement |
|------|-------------------|---------------|-------------|
| `test_pipeline_live_tier5_smoke` | **$0.05** (D-LIVE-COST-CEILING) | ~$0.02 (capture ~$0.005-0.01 via `run.COST_PER_Q[5]` × 5q + judge ~$0.015 via `0.003 × 5q × 1tier`) | (a) `pytest` body asserts `sum(json.loads(p.read_text()).get("total_usd", 0.0) for p in (*costs_eval, *costs_judge)) <= 0.05` (pytest gate per WARNING W4); (b) `--limit 5` + Tier 5 (cheapest agentic tier) bounds upper bound; (c) human-verify checkpoint (Plan 05-02 Task 1) acknowledges cost BEFORE invocation; (d) `live_eval_keys_ok` + `tier1_index_present` fixtures skip cleanly if prereqs missing — zero-cost skip path. |

No other live tests are introduced in Phase 5. The total live-cost budget
for the full Phase 5 verification cycle is therefore **$0.05** (one
invocation of `test_pipeline_live_tier5_smoke`).

---

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4+ (per `pyproject.toml` `[dependency-groups].test`: `pytest>=8.4,<9`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`; markers: `live: tests that hit real APIs and incur cost` (registered) |
| Quick run command | `uv run pytest evaluation/tests/test_eval_pipeline.py -m 'not live' -x` |
| Full unit suite | `uv run pytest evaluation/tests/ --ignore=evaluation/tests/test_eval_adapters.py --ignore=evaluation/tests/test_eval_smoke_live.py -m 'not live' -x` (excludes pre-existing tier-2 adapter fail per documented ignore list) |
| Live smoke | `uv run pytest -m live evaluation/tests/test_eval_pipeline.py::test_pipeline_live_tier5_smoke -x -v` |
| New framework deps | None — `pytest`, `rich.console`, and the four already-shipped harness modules are all already in use. |

**v1.1 hardening note:** `pyproject.toml` lacks `addopts = "-m 'not live'"`,
so a bare `pytest evaluation/tests/test_eval_pipeline.py` runs the live
test silently. Phase 5 mitigates by (a) the `@pytest.mark.live` gate so
`-m 'not live'` (the regression baseline pattern) skips it, and (b) Plan
05-02's documented invocation explicitly uses `-m live`. The underlying
gap is a STATE.md hardening item, not a Phase 5 deliverable.

---

## Linkage to RESEARCH.md and PLAN.md

This validation map is derived from `05-RESEARCH.md § Validation Architecture`
(lines 422-450) and the test list in `05-01-PLAN.md` Task 1 `<behavior>`
block. After checker iteration 1 (this revision):

- WARNING W3 (no-subprocess invariant) added Test 13 — covered above.
- WARNING W4 (live cost ceiling as pytest assertion) added the cost-summing
  assertion to `test_pipeline_live_tier5_smoke` — covered above.
- WARNING W5 (SHA lands in written JSON) extended Test 15 (formerly
  Test 14) to assert the propagated `git_sha` lands inside the
  freshly-written tier-4 JSON, not just on `args.Namespace`.
- WARNING W6 (test count consistency) — canonical count is **15 offline
  tests** (10 unit + 2 integration + 1 in-process-composition guard +
  2 reserved live slots) plus **1 live test** (Plan 05-02). All references
  in `05-01-PLAN.md` and `05-02-PLAN.md` updated to match.

Any future edit to RESEARCH.md's validation section or to either plan's
test list MUST also update this file (or vice versa); they are paired
contracts.
