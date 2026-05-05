# Phase 3: NaN Reason Instrumentation — Validation Map

**Phase:** 03-nan-reason-instrumentation
**Created:** 2026-05-05
**Status:** Active

This document formalizes the test-requirement map for Phase 3. Every requirement
covered by this phase has at least one test that proves the requirement is
satisfied. Source of truth: `03-RESEARCH.md` § Validation Architecture (lines
477-512). No `03-CONTEXT.md` exists — design decisions are within Claude's
discretion bounded by ROADMAP success criteria + HARN-05 requirement text.

---

## Phase Requirement Coverage

This phase covers one requirement ID from `.planning/REQUIREMENTS.md`:

| Req ID | Scope in This Phase | Out-of-Scope (Other Phases) |
|--------|--------------------|-----------------------------|
| HARN-05 | Per-row `nan_reason` distinguishes RAGAS-internal failures (`empty_statements`, `json_parse_failure`, `llm_did_not_finish`) IN ADDITION TO the existing pre-call reasons (`empty_contexts`, `agent_truncated`, `tier4_unavailable`, `cached_miss`). New `NaNReasonTracer` (langchain `BaseCallbackHandler` subclass) captures per-row exceptions during `evaluate()`; new `_classify_post_evaluate_nan` maps (row, metric, value, tracer) → reason string with documented precedence (faithfulness > answer_relevancy > context_precision per RESEARCH.md Pitfall 5). `score_query_log` is wired to instantiate the tracer, pass it via `callbacks=[tracer]`, and call the classifier in BOTH the dataframe branch and the `result.scores` fallback branch. `compare.py` receives ZERO edits (RESEARCH.md Architectural Responsibility Map: aggregator already iterates whatever string appears in `nan_reason`). One live smoke backstop asserts zero `unknown_nan` rows against fresh Gemini judge output. | TIER-01 / TIER-02 / TIER-03 / TIER-04 / TIER-05 (rerun phases). HARN-01 through HARN-04 (other harness work). METH-01 / METH-02 (CI / multi-judge). Backfill of pre-2026-05-04 metrics files (NO backfill — RESEARCH.md Open Question 2 / A4: existing `empty_contexts` labels are already truthful; Phase 7 rerun produces fresh richer data). |

Requirements NOT covered by this phase (deferred):

- TIER-01 through TIER-05 — tier-specific fixes / reruns; not in Phase 3 scope.
- HARN-01 through HARN-04 — other harness concerns (capture/cost/aggregation/comparison output) already shipped or scoped elsewhere.
- METH-01 (bootstrap CI) — deferred to v1.1 per project ROADMAP.
- METH-02 (multi-judge spot-check) — Phase 8 owns the 5×3 multi-judge re-score.

---

## Requirement → Test Map

| Req ID | Behavior Under Test | Test Type | Test Path | Plan Owning |
|--------|--------------------|-----------|-----------|-------------|
| HARN-05 (regression — pre-call reasons) | `_short_circuit_nan` continues to emit existing 4 reasons (`empty_contexts`, `agent_truncated`, `tier4_unavailable`, `cached_miss`); behavior byte-identical pre/post Phase 3 | unit | `evaluation/tests/test_eval_score.py::test_short_circuit_empty_contexts`, `::test_short_circuit_agent_truncated`, `::test_short_circuit_tier4_unavailable_with_contexts`, `::test_short_circuit_cached_miss` | 03-01 (PRESERVED — already in file ≥ Plan 02-04) |
| HARN-05 (tracer captures per-row exception) | `NaNReasonTracer.on_chain_error` records `(row_idx, metric_name) → exception_type_name` after `on_chain_start` builds the chain hierarchy via `parent_run_id` walk | unit | `evaluation/tests/test_eval_score.py::test_nan_reason_tracer_captures_metric_exception` (+ 5 sibling tests A-F per Plan 03-01 behavior block) | 03-01 (NEW) |
| HARN-05 (tracer chain-name suffix stripping) | METRIC chain `name="faithfulness-0"` is parsed to `metric_name="faithfulness"` via `.rsplit("-", 1)[0]` (Pitfall 4 of RESEARCH.md) | unit | `evaluation/tests/test_eval_score.py::test_nan_reason_tracer_captures_metric_exception` (subsumed) | 03-01 (NEW) |
| HARN-05 (tracer idempotence) | Firing `on_chain_error` twice on the same `(row, metric)` keeps the FIRST exception type (parent re-raise does not overwrite leaf) | unit | `evaluation/tests/test_eval_score.py::test_nan_reason_tracer_idempotent_on_repeat_error` (Test F) | 03-01 (NEW) |
| HARN-05 (classifier: RagasOutputParserException → json_parse_failure) | `_classify_post_evaluate_nan(row_idx, metric_name, None, tracer_with_RagasOutputParserException)` returns `"json_parse_failure"` | unit | `evaluation/tests/test_eval_score.py::test_classify_json_parse_failure` (Test H) | 03-01 (NEW) |
| HARN-05 (classifier: LLMDidNotFinishException → llm_did_not_finish) | `_classify_post_evaluate_nan(..., tracer_with_LLMDidNotFinishException)` returns `"llm_did_not_finish"` | unit | `evaluation/tests/test_eval_score.py::test_classify_llm_did_not_finish` (Test I) | 03-01 (NEW) |
| HARN-05 (classifier: faithfulness=NaN, no captured exception → empty_statements) | `_classify_post_evaluate_nan(0, "faithfulness", None, empty_tracer)` returns `"empty_statements"` (Pitfall 2 of RESEARCH.md — silent NaN path) | unit | `evaluation/tests/test_eval_score.py::test_classify_empty_statements` (Test J) | 03-01 (NEW) |
| HARN-05 (classifier: answer_relevancy=NaN, no captured exception → empty_questions) | `_classify_post_evaluate_nan(0, "answer_relevancy", None, empty_tracer)` returns `"empty_questions"` (defensive bucket per RESEARCH.md Open Question 4 / A1) | unit | `evaluation/tests/test_eval_score.py::test_classify_empty_questions` (Test K) | 03-01 (NEW) |
| HARN-05 (classifier: context_precision=NaN, no captured exception → invalid_verdicts) | `_classify_post_evaluate_nan(0, "context_precision", None, empty_tracer)` returns `"invalid_verdicts"` | unit | `evaluation/tests/test_eval_score.py::test_classify_invalid_verdicts` (Test L) | 03-01 (NEW) |
| HARN-05 (classifier: unknown metric, no rule → unknown_nan + WARNING) | `_classify_post_evaluate_nan(0, "mystery_metric", None, empty_tracer)` returns `"unknown_nan"` AND emits `logging.warning` (Pitfall 7 of RESEARCH.md — defensive default; no silent NaN reaches disk) | unit (+ caplog) | `evaluation/tests/test_eval_score.py::test_classify_unknown_nan_emits_warning` (Test M) | 03-01 (NEW) |
| HARN-05 (classifier: non-NaN value → None) | `_classify_post_evaluate_nan(0, "faithfulness", 0.85, any_tracer)` returns `None` (no NaN, no reason) | unit | `evaluation/tests/test_eval_score.py::test_classify_non_nan_returns_none` (Test G) | 03-01 (NEW) |
| HARN-05 (classifier: captured-but-unmapped type falls through to semantic) | tracer captured `"TimeoutError"` (unknown to classifier), faithfulness value=None → returns `"empty_statements"` (semantic fallback wins over unmapped capture) | unit | `evaluation/tests/test_eval_score.py::test_classify_unmapped_exception_falls_through_to_semantic` (Test N) | 03-01 (NEW) |
| HARN-05 (smoke import) | `from evaluation.harness.score import NaNReasonTracer, _classify_post_evaluate_nan` succeeds (asserts both new symbols are top-level exports) | unit | `evaluation/tests/test_eval_score.py::test_nan_reason_tracer_imports` | 03-01 (NEW) |
| HARN-05 (integration: stub LLM → json_parse_failure end-to-end) | `score_query_log` with `_MalformedJsonLLM` stub produces `ScoreRecord(nan_reason="json_parse_failure", faithfulness=None)` for a 1-record `QueryLog` with non-empty contexts (no pre-call short-circuit) | integration (stubbed, no live API) | `evaluation/tests/test_eval_score.py::test_score_query_log_distinguishes_json_parse_failure` | 03-02 (NEW) |
| HARN-05 (integration: stub LLM → empty_statements end-to-end) | `score_query_log` with `_EmptyStatementsLLM` stub (returns `{"statements": []}`) produces `ScoreRecord(nan_reason="empty_statements", faithfulness=None)` — exercises the silent NaN path through real RAGAS evaluate() | integration (stubbed, no live API) | `evaluation/tests/test_eval_score.py::test_score_query_log_distinguishes_empty_statements` | 03-02 (NEW) |
| HARN-05 (integration: clean stub LLM → nan_reason=None) | `score_query_log` with `_CleanLLM` stub (returns valid faithfulness JSON with statement+verdict, valid AR/CP responses) produces `ScoreRecord(nan_reason=None)` — exercises the all-metrics-non-NaN path | integration (stubbed, no live API) | `evaluation/tests/test_eval_score.py::test_score_query_log_nan_reason_none_when_clean` | 03-02 (NEW — added per checker WARNING 1 / Option A) |
| HARN-05 (regression: short-circuit precedence preserved) | A `QueryLog` with `retrieved_contexts=[]` still yields `ScoreRecord(nan_reason="empty_contexts")` even when a stub judge LLM is configured (pre-evaluate short-circuit fires before evaluate() is called) | integration (stubbed) | `evaluation/tests/test_eval_score.py::test_score_query_log_preserves_short_circuit_reasons` | 03-02 (NEW) |
| HARN-05 (compare.py rollup is reason-agnostic) | `aggregate_tier(tier, results_dir)` buckets all 4 reasons (NEW: `json_parse_failure`, `empty_statements`; OLD: `empty_contexts`; clean: `None`) into `nan_breakdown` correctly without any compare.py modification (load-bearing proof for RESEARCH.md Architectural Responsibility Map) | unit | `evaluation/tests/test_eval_compare.py::test_aggregate_tier_with_new_reasons` | 03-02 (NEW) |
| HARN-05 (compare.py emit_markdown renders new reasons) | `emit_markdown(...)` over the same seeded results emits a NaN-breakdown footer line containing `1 json_parse_failure` and `1 empty_statements` (proves the rendering path at compare.py:268-279 also flows new reasons through) | unit | `evaluation/tests/test_eval_compare.py::test_aggregate_tier_with_new_reasons` (extended assertion per checker WARNING 2) | 03-02 (NEW — added per checker WARNING 2) |
| HARN-05 (live backstop: no unknown_nan in real Gemini output) | Live RAGAS evaluate() over the existing Tier 5 smoke capture (5 questions) produces ZERO rows with `nan_reason="unknown_nan"` — proves the classifier covers every NaN path RAGAS 0.4.3 actually exercises against real judge LLM output | smoke (live, `-m live` opt-in only) | `evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_nan_reasons` | 03-03 (NEW; live-marked — deselected by default) |

---

## Coverage Self-Check

Every Wave 0 gap from `03-RESEARCH.md § Wave 0 Gaps` (lines 504-512) has at least
one row in the table above. Cross-reference:

- [x] `test_nan_reason_tracer_captures_per_row` → tracer rows (Plan 03-01 Tests A-F).
- [x] `test_classify_json_parse_failure` → classifier row (Plan 03-01 Test H).
- [x] `test_classify_empty_statements` → classifier row (Plan 03-01 Test J).
- [x] `test_classify_unknown_nan_warns` → unknown_nan row (Plan 03-01 Test M, uses `caplog`).
- [x] `test_score_query_log_distinguishes_reasons` → integration rows (Plan 03-02 — three sibling tests covering json_parse_failure, empty_statements, AND the clean-path nan_reason=None case added per checker WARNING 1).
- [x] `test_aggregate_tier_with_new_reasons` → compare.py rollup row + extended emit_markdown assertion (Plan 03-02 — single test exercises BOTH the aggregator AND the markdown renderer per checker WARNING 2).
- [x] `test_short_circuit_unchanged` → regression row (covered by existing `test_short_circuit_*` in test_eval_score.py — preserved, no new test needed).
- [x] Live smoke `unknown_nan == 0` assertion → live backstop row (Plan 03-03 `test_eval_smoke_nan_reasons`).
- [x] `test_emit_markdown_with_new_reasons` (Wave 0 gap line 496) → covered as the extended assertion inside `test_aggregate_tier_with_new_reasons` (single test exercises both paths; checker WARNING 2 closure).

No requirement in this phase is uncovered.

---

## Sampling Rate

| Stage | Command | Estimated Wall Time | API Cost |
|-------|---------|---------------------|----------|
| Per task commit | `python -m pytest evaluation/tests/test_eval_score.py -k "tracer or classify" -x` | ≤ 3 seconds | $0 (no API calls) |
| Per wave merge (Wave 1 → Wave 2) | `python -m pytest evaluation/tests/test_eval_score.py evaluation/tests/test_eval_compare.py -x` | ≤ 10 seconds | $0 |
| Phase gate (before `/gsd-verify-work`) | `python -m pytest evaluation/tests/ -x` PLUS `python -m pytest -m live evaluation/tests/test_eval_smoke_live.py -k smoke_nan_reasons -x` | ≤ 60 seconds (unit) + ≤ 90 seconds (live) | ≤ $0.02 per live run (Pitfall 7 of RESEARCH.md cost guard; well under the $0.05/run ceiling) |

---

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4+ (per `pyproject.toml` `pytest>=8.4,<9`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`; marker registered: `live: tests that hit real APIs and incur cost` |
| Quick run command | `python -m pytest evaluation/tests/test_eval_score.py -x` |
| Full unit suite | `python -m pytest evaluation/tests/ -x` (excludes live; no `-m live`) |
| Live smoke | `python -m pytest -m live evaluation/tests/test_eval_smoke_live.py -k smoke_nan_reasons -x -s` |
| New framework deps | None — `pytest`, `langchain-core` (transitive via `ragas`), and `ragas` 0.4.3 are all already pinned and in use. |

---

## Linkage to RESEARCH.md

This validation map is derived from `03-RESEARCH.md § Validation Architecture`
(lines 477-512, specifically the "Phase Requirements → Test Map" table at
lines 488-497 and the "Wave 0 Gaps" checklist at lines 504-512). Any future
edit to RESEARCH.md's validation section MUST also update this file (or vice
versa); they are paired contracts.

The test-to-plan ownership table above mirrors the `Plan Owning` column added
during planner revision (iteration 1) so that the plan checker can cross-check
each Wave 0 gap against the plan that materializes it.
