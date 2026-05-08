# Phase 1: Tier 5 Adapter Fix — Validation Map

**Phase:** 01-tier-5-adapter-fix
**Created:** 2026-05-04
**Status:** Active

This document formalizes the test-requirement map for Phase 1. Every requirement
covered by this phase has at least one test that proves the requirement is
satisfied. Source of truth: `01-RESEARCH.md` § Validation Architecture (lines
803-845). Locked decisions: `01-CONTEXT.md` D-01 through D-09.

---

## Phase Requirement Coverage

This phase covers two requirement IDs from `.planning/REQUIREMENTS.md`:

| Req ID | Scope in This Phase | Out-of-Scope (Other Phases) |
|--------|--------------------|-----------------------------|
| TIER-01 | Tier 5 adapter walk replaces hard-coded `retrieved_contexts=[]`; tracing toggle becomes env-var-conditional. Pitfalls 8 + 9 of 130-RESEARCH preserved. | Tier 4 graphml regen (Phase 2), full 5-tier rerun (Phase 7). |
| TIER-03 (Tier-5 portion) | `--smoke-question-ids` CLI flag on `evaluation/harness/run.py`; `evaluation/harness/smoke_gate.py` classifier + gate evaluator; live smoke test mirroring `test_eval_smoke_tier1_full_pipeline`. | Tier 4 smoke harness reuse (Phase 2 reuses the same `DEFAULT_SMOKE_IDS` constant + the same `smoke_gate.py` module). |

Requirements NOT covered by this phase (deferred):

- TIER-02, TIER-04, TIER-05 — other tiers' fixes / reruns; not in Phase 1 scope.
- METH-01 (bootstrap CI) — deferred to v1.1 per CONTEXT.md § Deferred Ideas.
- METH-02 (multi-judge spot-check) — Phase 8 owns the 5×3 multi-judge re-score.

---

## Requirement → Test Map

| Req ID | Behavior Under Test | Test Type | Test Path | Plan Owning |
|--------|--------------------|-----------|-----------|-------------|
| TIER-01 | `run_tier5` projects `ToolCallOutputItem.output` into provenance-prefixed `retrieved_contexts` (positive case + dedup) | unit (mocked Runner) | `evaluation/tests/test_eval_adapters.py::test_run_tier5_extracts_tool_outputs` | 01-01 (NEW) |
| TIER-01 (preservation) | `MaxTurnsExceeded` still surfaces `error="max_turns_exceeded"` and `retrieved_contexts=[]` (Pitfall 8 of 130-RESEARCH) | unit | `evaluation/tests/test_eval_adapters.py::test_run_tier5_max_turns_exceeded` | 01-01 (PRESERVED — already at line 178; D-07 invariant) |
| TIER-01 (Pitfall 9 of 130-RESEARCH) | Zero-tool-call run returns `retrieved_contexts=[]` honestly (no synthesis from `final_output`) | unit | `evaluation/tests/test_eval_adapters.py::test_run_tier5_zero_tool_calls_returns_honest_empty` | 01-01 (NEW) |
| TIER-01 (tool-error handling) | Tool error payloads (`{"error": "..."}` from `lookup_paper_metadata`) are skipped, not appended | unit | `evaluation/tests/test_eval_adapters.py::test_run_tier5_skips_tool_error_payloads` | 01-01 (NEW) |
| TIER-01 (tracing toggle) | `set_tracing_disabled(disabled=...)` reads `RAG_DEBUG_TIER5_TRACING` env var; default disabled when var unset | unit | `tier-5-agentic/tests/test_agent.py::test_tracing_disabled_default` | 01-01 (NEW) |
| TIER-01 (tracing toggle, opt-in) | Setting `RAG_DEBUG_TIER5_TRACING=1` flips `disabled=False` | unit | `tier-5-agentic/tests/test_agent.py::test_tracing_enabled_when_env_set` | 01-01 (NEW) |
| TIER-01 (live) | End-to-end: real Runner.run, real ChromaDB, real OpenRouter; assert `len(rec.retrieved_contexts) > 0` for at least one of the 5 smoke questions | integration / live | `evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier5_full_pipeline` | 01-02 (NEW; mirror of `test_eval_smoke_tier1_full_pipeline`) |
| TIER-03 (Tier-5 filter) | `--smoke-question-ids` CLI flag filters `golden_qa.json` correctly; preserves user-given order | unit | `evaluation/tests/test_eval_run.py::test_run_smoke_ids_filters_correctly` | 01-02 (NEW) |
| TIER-03 (Tier-5 error path) | Unknown question IDs cause `amain` to return 2 with a friendly red-printed error (no silent partial run) | unit | `evaluation/tests/test_eval_run.py::test_run_smoke_ids_unknown_returns_2` | 01-02 (NEW) |
| TIER-03 (gate classifier) | `classify_row(rec, score)` returns one of `populated`, `empty_no_tool_calls`, `agent_truncated` per priority order | unit | `evaluation/tests/test_eval_smoke_gate.py::test_classify_populated`, `::test_classify_empty_no_tool_calls`, `::test_classify_agent_truncated` | 01-02 (NEW) |
| TIER-03 (gate PASS) | 3 populated + 1 truncated + 1 empty → `verdict="PASS"` (canonical Pitfall 5 of 132-RESEARCH case: exclude truncated and empty from denominator) | unit | `evaluation/tests/test_eval_smoke_gate.py::test_gate_pass_with_measurable_exclusions` | 01-02 (NEW) |
| TIER-03 (gate FAIL low ratio) | Populated/measurable ratio < 0.8 → `verdict="FAIL"` with reason in message | unit | `evaluation/tests/test_eval_smoke_gate.py::test_gate_fail_low_ratio` | 01-02 (NEW) |
| TIER-03 (gate FAIL NaN metric) | Populated row with `faithfulness=None` (judge failure) → `verdict="FAIL"` mentioning "NaN metrics on populated rows" | unit | `evaluation/tests/test_eval_smoke_gate.py::test_gate_fail_nan_metric_on_populated` | 01-02 (NEW) |
| TIER-03 (gate INCONCLUSIVE) | Measurable denominator < 3 → `verdict="INCONCLUSIVE"` mentioning "re-run with different IDs" | unit | `evaluation/tests/test_eval_smoke_gate.py::test_gate_inconclusive_too_few_measurable` | 01-02 (NEW) |
| TIER-01 (fallback scaffolding) | `evaluation.harness.diagnostics.write_fallback_log` produces a Pydantic-typed JSON at `evaluation/results/diagnostics/tier-5-fallback-{TS}.json` (round-trip via `model_validate_json`) | unit | `evaluation/tests/test_eval_diagnostics.py::test_fallback_log_roundtrip` | 01-03 (NEW) |
| TIER-01 (fallback scaffolding) | `SmokeQuestionResult.final_output_truncated` rejects strings > 400 chars (truncation is caller responsibility) | unit | `evaluation/tests/test_eval_diagnostics.py::test_fallback_log_truncates_final_output` | 01-03 (NEW) |
| TIER-01 (fallback scaffolding) | `_captured_versions()` returns 5-key dict with all expected library names; `openai-agents` value is non-empty string | unit | `evaluation/tests/test_eval_diagnostics.py::test_captured_versions_snapshot` | 01-03 (NEW) |
| TIER-01 (fallback scaffolding) | `open_fallback_log()` initializes required fields (git_sha, opened_at ISO 8601 UTC, attempts=[], final_disposition=None, captured_versions has 5 keys) | unit | `evaluation/tests/test_eval_diagnostics.py::test_open_fallback_log_initializes_required_fields` | 01-03 (NEW) |

---

## Coverage Self-Check

Every Wave 0 gap from `01-RESEARCH.md § Wave 0 Gaps` (lines 836-845) has at least
one row in the table above. Cross-reference:

- [x] `test_run_tier5_extracts_tool_outputs` → row 1 (TIER-01 positive case + dedup).
- [x] `test_run_tier5_zero_tool_calls_returns_honest_empty` → row 3 (Pitfall 9).
- [x] `test_run_tier5_happy_path` (UPDATE) → row 3 covers the same path; the existing happy_path is preserved as a duplicate sentinel per Plan 01-01 Task 1 behavior.
- [x] `test_run_smoke_ids_filters_correctly` → row 8 (TIER-03 filter).
- [x] `test_run_smoke_ids_unknown_returns_2` → row 9 (TIER-03 error path).
- [x] `test_eval_smoke_gate.py` (NEW FILE) → rows 10-14 (gate logic, exclusions, measurable denominator, ratio, NaN handling).
- [x] `test_eval_smoke_tier5_full_pipeline` → row 7 (live integration).
- [x] `test_tracing_disabled_default` → row 5 (tracing toggle env var default).

No requirement in this phase is uncovered.

---

## Sampling Rate

| Stage | Command | Estimated Wall Time | API Cost |
|-------|---------|---------------------|----------|
| Per task commit | `pytest evaluation/tests/test_eval_adapters.py -k tier5 evaluation/tests/test_eval_run.py evaluation/tests/test_eval_smoke_gate.py evaluation/tests/test_eval_diagnostics.py tier-5-agentic/tests/test_agent.py -x` | ≤ 5 seconds | $0 (no API calls) |
| Per wave merge | `pytest -x` (full unit suite, excludes `-m live`) | ≤ 30 seconds | $0 |
| Phase gate (before `/gsd-verify-work`) | `pytest -x` PLUS `pytest -m live evaluation/tests/test_eval_smoke_live.py -x` | ≤ 3 minutes | ≤ $0.05 per run (Pitfall 7 of 132-RESEARCH guard) |

---

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4+ (per `pyproject.toml:74` `pytest>=8.4,<9`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (line 80-81); marker registered: `live: tests that hit real APIs and incur cost` |
| Quick run command | `pytest evaluation/tests/test_eval_adapters.py -k tier5 -x` |
| Full suite command | `pytest -x` (excludes live; no `-m live`); `pytest -m live` for live smoke |
| New framework deps | None — pytest 8.4 already pinned and in use. |

---

## Linkage to RESEARCH.md

This validation map is derived from `01-RESEARCH.md § Validation Architecture`
(lines 803-845, specifically the "Phase Requirements → Test Map" table at
lines 818-827 and the "Wave 0 Gaps" checklist at lines 836-845). Any future
edit to RESEARCH.md's validation section MUST also update this file (or vice
versa); they are paired contracts.
