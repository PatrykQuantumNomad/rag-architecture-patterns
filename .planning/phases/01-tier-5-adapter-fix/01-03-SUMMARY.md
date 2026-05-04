---
phase: 01-tier-5-adapter-fix
plan: 03
subsystem: testing
tags: [pyproject, optional-dependencies, debug-tier5, pydantic, openinference, phoenix, opentelemetry, fallback-runbook, diagnostics]

# Dependency graph
requires:
  - phase: 01-tier-5-adapter-fix
    provides: Tier 5 adapter walk + RAG_DEBUG_TIER5_TRACING toggle (Plan 01-01)
provides:
  - "[debug-tier5] opt-in extra in pyproject.toml (OpenInference 1.4.2 + arize-phoenix >=6,<8 + opentelemetry-sdk >=1.30,<2 + opentelemetry-exporter-otlp-proto-http >=1.30,<2)"
  - "evaluation/harness/diagnostics.py — FallbackLog + FallbackAttempt + SpanObservation + SmokeQuestionResult Pydantic models, open_fallback_log, write_fallback_log writer"
  - ".planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md — operator checklist encoding D-06 instrument-first ordering"
affects: [phase-7-full-rerun, phase-9-frozen-doc, tier-5-fallback]

# Tech tracking
tech-stack:
  added:
    - "[debug-tier5] extra (opt-in only — does NOT mutate default install)"
  patterns:
    - "Pydantic v2 BaseModel + model_dump_json(indent=2) persistence (matches records.py:69)"
    - "Single source of truth for SHA/TS helpers (diagnostics.py imports from run.py — no redefinition)"
    - "Literal-typed disposition fields for machine-checkable provenance (AttemptKind, SmokeOutcome, final_disposition)"
    - "Opt-in optional-dependencies for debug stacks — runtime modules contain ZERO references to debug deps (D-09)"

key-files:
  created:
    - "evaluation/harness/diagnostics.py"
    - "evaluation/tests/test_eval_diagnostics.py"
    - ".planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md"
  modified:
    - "pyproject.toml"

key-decisions:
  - "FallbackLog persistence uses model_dump_json(indent=2) — matches records.py persistence convention exactly"
  - "diagnostics.py imports _git_sha / _ts / _ts_for_filename / _REPO_ROOT from run.py rather than redefining (single source of truth for SHA/TS conventions)"
  - "SmokeQuestionResult.final_output_truncated enforces max_length=400 at the model level — truncation is the caller's responsibility, the model enforces the contract"
  - "Runbook encodes D-06 ordering verbatim: instrument first, mutate second — explicit anti-pattern listing prevents skipping Step 1"

patterns-established:
  - "Opt-in [debug-tier5] pattern: diagnostic tooling lives behind an extras flag; default install is byte-identical without it (template for any future debug/profiling stacks)"
  - "Provenance log shape: phase + git_sha + opened_at + closed_at + captured_versions + attempts[] — frozen-doc-grade artifact for any phase that may degrade-ship"

# Metrics
duration: 4min
completed: 2026-05-04
---

# Phase 1 Plan 03: Fallback scaffolding (debug-tier5 extra + diagnostics module + runbook) Summary

**Opt-in `[debug-tier5]` extras + Pydantic-typed FallbackLog writer + D-06 instrument-first runbook — provenance scaffolding ready for the Plan 01-02 smoke checkpoint to invoke on FAIL/INCONCLUSIVE.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-04T14:47:27Z
- **Completed:** 2026-05-04T14:51:01Z
- **Tasks:** 2
- **Files created:** 3
- **Files modified:** 1

## Accomplishments

- `[debug-tier5]` opt-in extra in `pyproject.toml` with all 4 required pins (OpenInference 1.4.2, arize-phoenix >=6,<8, opentelemetry-sdk >=1.30,<2, opentelemetry-exporter-otlp-proto-http >=1.30,<2). Default install is byte-identical without it.
- `evaluation/harness/diagnostics.py` — `FallbackLog` Pydantic model (phase, git_sha, opened_at, closed_at, final_disposition, captured_versions, attempts[]), `FallbackAttempt` with `AttemptKind` Literal (`instrument`/`simplify_schema`/`switch_model_slug`/`bump_openai_agents`), `SpanObservation` for Phoenix-derived triage, `SmokeQuestionResult` with max_length=400 truncation contract, `open_fallback_log()` constructor, `write_fallback_log()` writer producing `evaluation/results/diagnostics/tier-5-fallback-{TS}.json`.
- `evaluation/tests/test_eval_diagnostics.py` — 4 unit tests, all green (round-trip, max_length contract, captured_versions snapshot, open_fallback_log defaults).
- `.planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md` — operator-facing runbook encoding D-06 ordering verbatim. Steps: open log -> instrument first -> simplify schema -> switch model slug -> bump openai-agents (LAST RESORT, requires user authorization) -> close log. Includes cost budget (≤$0.50 total) and explicit anti-patterns.

## Task Commits

1. **Task 1 RED: failing tests for diagnostics** — `4be0f0a` (test)
2. **Task 1 GREEN: [debug-tier5] extra + diagnostics module** — `3e8c601` (feat)
3. **Task 2: fallback runbook** — `99e02b0` (docs)

_Note: Task 1 followed TDD (RED -> GREEN). No REFACTOR commit needed — implementation passed tests on first run with the literal source from the plan._

## Files Created/Modified

- `pyproject.toml` — added `[debug-tier5]` optional-dependencies group (10 lines, after `tier-5`, before `evaluation`)
- `evaluation/harness/diagnostics.py` — NEW, 120 LOC, FallbackLog Pydantic models + writer
- `evaluation/tests/test_eval_diagnostics.py` — NEW, 143 LOC, 4 unit tests (all green)
- `.planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md` — NEW, 172 LOC operator runbook

## Decisions Made

- **Persistence pattern reuse:** `write_fallback_log` uses `path.write_text(log.model_dump_json(indent=2))` — exactly the shape used at `records.py:69` for `QueryLog`. No new persistence convention introduced.
- **Helper reuse:** `diagnostics.py` imports `_git_sha`, `_ts`, `_ts_for_filename`, `_REPO_ROOT` from `evaluation.harness.run` rather than redefining. Single source of truth for SHA + ISO 8601 Z conventions; if `run.py` changes the timestamp format later, diagnostics tracks it automatically.
- **Truncation contract at model level:** `SmokeQuestionResult.final_output_truncated = Field(max_length=400)` rejects oversized strings via `ValidationError`. The caller (runbook Step 1 / Step 2) must truncate before instantiation; this prevents accidental fallback logs ballooning to MB-sized JSON files.
- **Runbook anti-patterns explicit:** The runbook lists 4 anti-patterns by name (skip Step 1, modify agent prompt, install debug-tier5 by default, bump openai-agents without authorization). Operators can paste-cite the anti-pattern in case of process disputes.

## Deviations from Plan

### Process Note: Parallel-execution commit collision

**1. [Process - Plan-01-02 file leakage] Commit `3e8c601` (feat 01-03 GREEN) inadvertently included Plan 01-02 files**
- **Found during:** Task 1 GREEN commit (post-stage diff inspection)
- **Issue:** Plan 01-02 (running in parallel per the plan's PARALLEL note) had pre-staged its work-in-progress files (`evaluation/tests/test_eval_run.py` modifications and a new `evaluation/tests/test_eval_smoke_gate.py`) in the git index before this executor ran `git add pyproject.toml evaluation/harness/diagnostics.py`. Git's index is repo-global, not per-plan, so the `git commit` swept up the already-staged Plan 01-02 files alongside Plan 01-03's intended files.
- **Fix:** None applied. Reverting would require coordination with the parallel Plan 01-02 executor and risk losing its work. The files are committed correctly with valid content; only the commit message is slightly inaccurate (commit `3e8c601` is labeled `feat(01-03)` but contains Plan 01-02 test files).
- **Files inadvertently swept in:** `evaluation/tests/test_eval_run.py` (96 lines added), `evaluation/tests/test_eval_smoke_gate.py` (304 lines, NEW).
- **Impact on Plan 01-02:** Plan 01-02's executor will see those files as already committed. It should observe a clean working tree for those paths and may need to author a placeholder docs/no-op commit, or accept the existing commit hash as the authoritative landing point and reference it from its own SUMMARY.md.
- **Lesson:** When two plans run in parallel against a single repo (no worktree isolation), the git index is a shared resource. Future parallel-execution plans should either (a) use git worktrees per executor, or (b) use `git stash` / `git restore --staged` to scope the index per commit. This was a known risk — the plan's PARALLEL note flagged commit-message collisions but did not flag index sharing.
- **Committed in:** 3e8c601 (Task 1 GREEN commit — content correct, label slightly broad)

---

**Total deviations:** 1 process note (no code-level deviation)
**Impact on plan:** All Plan 01-03 deliverables landed correctly. Plan 01-02 files were inadvertently included in commit 3e8c601 due to shared git index across parallel executors; no content was lost or corrupted. Process lesson captured for future parallel-execution plans.

## Issues Encountered

None — RED tests failed as expected (ImportError on missing module), GREEN passed on first run with the literal source from the plan, runbook grep verifications all returned ≥1 on first write.

## Verification Results

All 5 success-criteria items confirmed green:

1. **Goal-backward truth #1** — `[debug-tier5]` extra is opt-in (`tomllib` parse confirms presence + 4 pins; default `pyproject.toml.dependencies` is unchanged at `[]`).
2. **Goal-backward truth #2** — `write_fallback_log` produces `evaluation/results/diagnostics/tier-5-fallback-{TS}.json` (verified via `test_fallback_log_roundtrip` writing to `tmp_path` and asserting `path.exists()` + filename pattern).
3. **Goal-backward truth #3** — `model_dump_json(indent=2)` ↔ `model_validate_json` round-trip green (test passes; reloaded log preserves git_sha, attempts[*].kind, attempts[*].smoke_outcome).
4. **Goal-backward truth #4** — Runbook exists at `.planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md` with all required keywords (Instrument first ×2, write_fallback_log ×3, simplify_schema ×1, switch_model_slug ×1, bump_openai_agents ×2, RAG_DEBUG_TIER5_TRACING ×2, debug-tier5 ×6).
5. **Goal-backward truth #5** — D-09 leakage check: `tier_5.py`, `tier-5-agentic/agent.py`, `tier-5-agentic/tools.py` each return `0` matches for `openinference|arize-phoenix|opentelemetry`. Default install is import-time-clean.

## Threat Model Compliance

| Threat ID | Mitigation Plan Status |
|-----------|------------------------|
| T-03-01 (T) — debug-tier5 widens default install | **Mitigated.** `[debug-tier5]` is a standalone optional-dependencies group; no other extras reference it. Verified via `tomllib` parse. |
| T-03-02 (I) — fallback log persists provenance | **Accepted.** Same provenance set as `evaluation/results/queries/`; no new exposure surface. |
| T-03-03 (D) — fallback runbook DoS via runaway smoke runs | **Mitigated.** Runbook "Cost budget" section caps at $0.50 total; `cost_usd` per attempt makes running total auditable. |
| T-03-04 (E) — autonomous bump of openai-agents | **Mitigated.** Runbook Step 2c marks `bump_openai_agents` as LAST RESORT requiring user authorization. The `AttemptKind = "bump_openai_agents"` Literal in `diagnostics.py` is the documented marker. |

## User Setup Required

None — `[debug-tier5]` extras are opt-in; default `uv sync` is unchanged. Operators only invoke `uv sync --extra debug-tier5` if/when the runbook is exercised.

## Next Phase Readiness

- **Plan 01-02 smoke checkpoint:** has all the scaffolding it needs to invoke the runbook on FAIL/INCONCLUSIVE — `evaluation.harness.diagnostics` is importable, the runbook is at the documented path, and the `[debug-tier5]` extras flag is ready for `uv sync`.
- **Was the runbook exercised?** **Unknown at write-time.** This plan ran in parallel with Plan 01-02; Plan 01-02's smoke gate result is not yet known to this executor. If Plan 01-02 reports PASS, this scaffolding remains dormant for Phase 7; if FAIL/INCONCLUSIVE, an operator works the runbook and Plan 01-02's SUMMARY links to the resulting `tier-5-fallback-{TS}.json`.
- **Phase 7 reads this:** to know whether Tier 5 ships clean (smoke PASS, no diagnostics JSON) or with the partial-fix caveat (DEGRADED_SHIP with diagnostics JSON cited in the frozen doc).

## Self-Check: PASSED

- [x] `evaluation/harness/diagnostics.py` exists and imports cleanly under default install
- [x] `evaluation/tests/test_eval_diagnostics.py` exists with 4 passing tests
- [x] `.planning/phases/01-tier-5-adapter-fix/01-fallback-runbook.md` exists with all required keywords
- [x] `pyproject.toml` `[debug-tier5]` extra contains all 4 documented pins
- [x] Commits `4be0f0a`, `3e8c601`, `99e02b0` exist in git log
- [x] D-09 verified: zero debug-stack references in `tier_5.py` / `agent.py` / `tools.py`

---

*Phase: 01-tier-5-adapter-fix*
*Plan: 03*
*Completed: 2026-05-04*
