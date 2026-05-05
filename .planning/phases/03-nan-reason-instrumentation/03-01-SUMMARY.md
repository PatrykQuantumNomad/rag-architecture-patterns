---
phase: 03-nan-reason-instrumentation
plan: 01
subsystem: evaluation
tags: [ragas, langchain-callbacks, nan-classification, tdd, evaluation-harness, harn-05, gemini-2.5-flash, openrouter]

# Dependency graph
requires:
  - phase: 02-04
    provides: "JUDGE_MAX_TOKENS=8192 wired into score._build_judge — Tier 4 + Tier 5 smoke gates PASS. Plan 03-01 reuses .venv (langchain_core + ragas.callbacks) and the unmodified _short_circuit_nan baseline as the regression surface."
provides:
  - "evaluation/harness/score.py — NaNReasonTracer(BaseCallbackHandler) class capturing per-row exception types via on_chain_start dispatch on ChainType (ROW/METRIC/RAGAS_PROMPT) + on_chain_error idempotent record, AND _classify_post_evaluate_nan(row_idx, metric_name, metric_value, tracer) -> Optional[str] pure helper mapping (cell, captured-error) to one of: None, 'json_parse_failure', 'llm_did_not_finish', 'empty_statements', 'empty_questions', 'invalid_verdicts', 'unknown_nan'"
  - "evaluation/tests/test_eval_score.py — 15 new tests (1 smoke import + 6 NaNReasonTracer + 8 _classify_post_evaluate_nan) appended after test_build_judge_max_tokens_is_named_constant; full file now 28 tests (was 13)"
affects: [phase-03-02, phase-03-03, phase-7]

# Tech tracking
tech-stack:
  added: []  # langchain_core + ragas.callbacks already transitive deps in .venv (verified pre-commit)
  patterns:
    - "langchain BaseCallbackHandler subclass-as-tracer: subclass langchain_core.callbacks.BaseCallbackHandler, override on_chain_start (build {run_id → {type, row_index, metric_name}} dict) + on_chain_error (record type(error).__name__ in (row_idx, metric) keyed dict). The tracer is installed at evaluate() callbacks=[...] (Plan 03-02) and read AFTER evaluate() returns (Plan 03-02). Pure data structure, no side effects beyond the dict mutations."
    - "ChainType-dispatch in on_chain_start: ROW captures row_index from metadata; METRIC inherits row_index from parent ROW + extracts metric_name from serialized['name'] by stripping '-{i}' suffix (Pitfall 4 — RAGAS names metric runs '{metric}-{row_index}'); RAGAS_PROMPT inherits both from parent METRIC via parent_run_id walk in self._chains. Three-level hierarchy mirrors RAGAS evaluation.py:244 + executor.submit name= conventions."
    - "Idempotence-by-first-set: tracer.errors[(row, metric)] only set if key absent. Preserves leaf-most exception type when parent re-raises propagate the same error up the chain (e.g. RAGAS_PROMPT raises RagasOutputParserException, METRIC catches+re-raises, ROW catches+re-raises — all three on_chain_error firings preserve the original RagasOutputParserException type rather than overwriting with the parent re-raise)."
    - "Pure post-evaluate classifier with precedence ordering: _classify_post_evaluate_nan reads tracer.errors AFTER evaluate(). Precedence is captured-exception-type FIRST (specific), then per-metric semantic NaN paths (general), then 'unknown_nan' + WARNING log (defensive — Pitfall 7 of 03-RESEARCH.md says never silently drop NaN). Captured-but-unknown-type (e.g. 'TimeoutError') falls through to per-metric semantic mapping rather than 'unknown_nan' so a known-metric NaN with a novel exception type still gets a meaningful reason string."
    - "TDD red→green for two coupled units in one plan: Tracer + Classifier are tightly coupled (classifier reads tracer's errors dict) but both pure → land in a single plan with one RED commit (15 failing tests) + one GREEN commit (109 lines of impl). Plan 03-02 wires them into score_query_log; Plan 03-01 ships them as standalone testable units."

key-files:
  created:
    - ".planning/phases/03-nan-reason-instrumentation/03-01-SUMMARY.md (this file)"
  modified:
    - "evaluation/harness/score.py (+109 / -0; pure additions; no protected function bodies modified — _short_circuit_nan, _to_float_or_none, _build_judge, _strip_openrouter_prefix, _ts, score_query_log, _persist_metrics, amain, build_parser, main are all byte-identical post-commit)"
    - "evaluation/tests/test_eval_score.py (+253 / -0; 15 new tests appended after test_build_judge_max_tokens_is_named_constant; 13 pre-existing tests untouched)"

key-decisions:
  - "Stage 1 of HARN-05 (Plan 03-01) ships the two new units in isolation; Stage 2 (Plan 03-02) wires NaNReasonTracer into evaluate(callbacks=[tracer]) inside score_query_log AND post-processes each row's metrics through _classify_post_evaluate_nan. Splitting the two stages keeps Plan 03-01 fast (~5 min wall, no live LLM call needed for any test) and reduces blast radius — Plan 03-01's RED→GREEN cycle is fully offline."
  - "ChainType.ROW / METRIC / RAGAS_PROMPT enum values are used directly (not .value) in metadata.get('type') comparisons. The enum is a str-Enum so `ChainType.ROW == 'row'` evaluates True, but the canonical pattern in 03-RESEARCH.md (and ragas/evaluation.py:244) is to compare against the enum member. Tests construct metadata={'type': ChainType.ROW, ...} with the enum member rather than the raw 'row' string for forward-compat with future RAGAS releases that may switch to a non-str-Enum implementation."
  - "Idempotence semantics chose 'first wins' (leaf-most error preserved) rather than 'last wins' (root-most error preserved). RAGAS_PROMPT is the leaf and that's where RagasOutputParserException originates; METRIC and ROW catch-and-re-raise the same exception, but the leaf is the most specific signal for HARN-05 reason classification. If a future RAGAS version starts wrapping leaf exceptions in higher-level types (e.g. METRIC catches RagasOutputParserException and re-raises as RagasMetricException), this decision needs revisiting — Plan 03-02 integration tests should surface that scenario."
  - "_classify_post_evaluate_nan returns 'unknown_nan' + emits WARNING log for cells the precedence ladder doesn't recognize (unknown metric, or cells reaching the bottom of the ladder). The WARNING is critical because it surfaces the gap in run logs — Phase 7's full-run will exercise the function over 30 questions × 5 tiers and any 'unknown_nan' bucket count >0 in comparison.md is a signal to extend the classifier. Tests assert the warning fires (Test M uses caplog fixture)."
  - "score.py byte-identical preservation of all 9 pre-existing top-level functions verified via `git diff` showing only +109 lines (insertions: import + class + function), 0 lines removed. score_query_log is unchanged → Plan 03-02 modifies it; Plan 03-01 does not. This isolation is what makes Plan 03-01's regression risk near-zero."
  - "No deviations applied. Plan executed exactly as written. langchain_core + ragas.callbacks imports verified working in .venv before any code change (the only pre-implementation check beyond reading the plan)."

patterns-established:
  - "Pattern 1: subclass-as-tracer for RAGAS callback chain — install at evaluate(callbacks=[tracer]) (Plan 03-02), read tracer.errors dict AFTER evaluate() returns. Generalizes beyond NaN reason: any future per-row diagnostic that needs to recover state lost by Executor.wrap_callable_with_index can subclass BaseCallbackHandler and follow the same on_chain_start dispatch + on_chain_error capture pattern."
  - "Pattern 2: TDD for callback-chain code via direct method invocation — tests instantiate NaNReasonTracer() and call tracer.on_chain_start(...) / on_chain_error(...) directly with handcrafted args (uuid.uuid4() run_ids, plain dict metadata) rather than spinning up a full RAGAS evaluate() loop. Keeps unit tests <1s each and offline; Plan 03-02 will add an end-to-end integration test with a stub LLM that exercises the real callback chain."
  - "Pattern 3: pure-helper signature for post-evaluate classification — _classify_post_evaluate_nan(row_idx, metric_name, metric_value, tracer) takes the cell coordinates + captured state + the pre-built tracer, returns Optional[str]. No I/O beyond a single logging.warning call on the unknown_nan path. Trivially testable, trivially mockable; tests for it instantiate a fresh NaNReasonTracer() (or one with errors[(row, metric)] preset) and assert the returned string."

# Metrics
duration: ~5min wall (Task 1 RED ~2min including 15-test append + RED verification, Task 2 GREEN ~3min including imports + class + classifier + verification gates 4-7 + final pytest)
completed: 2026-05-05
---

# Phase 03 Plan 03-01: NaNReasonTracer + _classify_post_evaluate_nan Summary

**One-line:** Added `NaNReasonTracer(BaseCallbackHandler)` and `_classify_post_evaluate_nan` pure helper to `evaluation/harness/score.py` via TDD red→green so Plan 03-02 can wire them into `score_query_log` and HARN-05 can ship structured `nan_reason` strings (json_parse_failure / llm_did_not_finish / empty_statements / empty_questions / invalid_verdicts / unknown_nan) instead of opaque NaN buckets in `comparison.md`.

## Performance

- **Duration:** ~5 min wall (Task 1 ~2 min code + RED verification, Task 2 ~3 min impl + verification gates + final pytest)
- **Started:** 2026-05-05T17:46:00Z (approx — RED commit landed)
- **Completed:** 2026-05-05T17:50:53Z (this SUMMARY)
- **Tasks:** 2 (Task 1 RED unit tests, Task 2 GREEN implementation)
- **Files created:** 1 (this SUMMARY.md)
- **Files modified:** 2 (score.py +109, test_eval_score.py +253)
- **Test count delta:** 13 → 28 (+15 = 1 smoke import + 6 NaNReasonTracer + 8 _classify_post_evaluate_nan)
- **LOC added to score.py:** 109 (50 NaNReasonTracer class + 30 _classify_post_evaluate_nan + ~5 imports/constant + ~24 docstrings/blank lines)

## Accomplishments

- **`NaNReasonTracer(BaseCallbackHandler)`** at `evaluation/harness/score.py` (lines 171–225, between unchanged `_short_circuit_nan` and unchanged `_to_float_or_none`):
  - `__init__`: builds two empty dicts — `self._chains: dict[str, dict]` (run_id-keyed chain context) and `self.errors: dict[tuple[int, str], str]` ((row_idx, metric) → exception type name).
  - `on_chain_start(serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)`: dispatches on `metadata.get('type')`. ROW → store `row_index` from metadata. METRIC + parent_run_id → inherit `row_index` from parent in `self._chains`, extract `metric_name` from `serialized['name']` by `rsplit('-', 1)[0]` (strips `-{i}` suffix per Pitfall 4 of 03-RESEARCH.md). RAGAS_PROMPT + parent_run_id → inherit both `row_index` and `metric_name` from parent METRIC. All paths store the entry under `str(run_id)`.
  - `on_chain_error(error, *, run_id, **kwargs)`: looks up chain by `str(run_id)`. If `row_idx` and `metric_name` both present, records `type(error).__name__` at `self.errors[(row_idx, metric_name)]` with idempotence (only set if key absent). Silently no-ops when `row_idx` or `metric_name` is missing — that's the EVALUATION-level / ROW-level case, which doesn't map to a (row, metric) NaN cell.
- **`_classify_post_evaluate_nan(row_idx, metric_name, metric_value, tracer) -> Optional[str]`** at `evaluation/harness/score.py` (lines 228–268). Precedence ladder (most specific to least):
  1. `metric_value is not None` → `None` (defensive: non-NaN never gets a reason).
  2. `tracer.errors.get((row_idx, metric_name)) == 'RagasOutputParserException'` → `'json_parse_failure'`.
  3. captured `'LLMDidNotFinishException'` → `'llm_did_not_finish'`.
  4. `metric_name == 'faithfulness'` (no/unmapped capture) → `'empty_statements'` (`ragas/metrics/_faithfulness.py:210-211`: `Faithfulness._ascore` returns `np.nan` when `statements==[]`).
  5. `metric_name == 'answer_relevancy'` → `'empty_questions'` (`ragas/metrics/_answer_relevance.py:120-124`: `calculate_score` returns `np.nan` when all `gen_questions==''`).
  6. `metric_name == 'context_precision'` → `'invalid_verdicts'` (`ragas/metrics/_context_precision.py:116`: `_calculate_average_precision` returns `np.nan`).
  7. Unknown metric → `'unknown_nan'` + `_NAN_REASON_LOG.warning(...)` (Pitfall 7 of 03-RESEARCH.md — surface in run logs, never silently drop NaN).
- **3 module-level imports added** at the top of `score.py` (after the existing import block):
  - `import logging` (stdlib).
  - `from langchain_core.callbacks import BaseCallbackHandler` (transitive dep, already in .venv).
  - `from ragas.callbacks import ChainType` (already in .venv via `ragas==0.4.3`).
- **`_NAN_REASON_LOG = logging.getLogger(__name__)`** module-level logger (line 31 region) — used only by `_classify_post_evaluate_nan` on the `'unknown_nan'` path.
- **15 new unit tests in `evaluation/tests/test_eval_score.py`** (appended after `test_build_judge_max_tokens_is_named_constant`):
  - `test_nan_reason_smoke_import` — both names importable from `evaluation.harness.score`.
  - **NaNReasonTracer (6 tests, A–F):** `test_nan_reason_tracer_row_only_no_metric_no_capture`, `test_nan_reason_tracer_captures_metric_exception_parser`, `test_nan_reason_tracer_captures_metric_exception_llm_did_not_finish`, `test_nan_reason_tracer_ragas_prompt_inherits_via_parent_walk`, `test_nan_reason_tracer_two_metrics_same_row`, `test_nan_reason_tracer_idempotence_first_wins`.
  - **_classify_post_evaluate_nan (8 tests, G–N):** `test_classify_non_nan_returns_none`, `test_classify_captured_ragas_output_parser_exception`, `test_classify_captured_llm_did_not_finish_exception`, `test_classify_empty_tracer_faithfulness_empty_statements`, `test_classify_empty_tracer_answer_relevancy_empty_questions`, `test_classify_empty_tracer_context_precision_invalid_verdicts`, `test_classify_unknown_nan_emits_warning` (uses caplog), `test_classify_captured_unknown_exception_falls_to_semantic`.
- **All 28 tests pass after GREEN** (13 existing regression + 15 new). Total wall time `pytest evaluation/tests/test_eval_score.py -x`: 3.27s.

## Task Commits

Each task committed atomically (TDD red→green per project convention):

1. **Task 1 RED — failing tests for NaNReasonTracer + _classify_post_evaluate_nan** — `e97e864` (`test(03-01)`) — appends 15 tests to `evaluation/tests/test_eval_score.py`. RED state verified via `pytest -k "tracer or classify or smoke_import"` → 15 failed (all `ImportError: cannot import name 'NaNReasonTracer' / '_classify_post_evaluate_nan'`), 13 deselected (existing regression intact).
2. **Task 2 GREEN — implement NaNReasonTracer + _classify_post_evaluate_nan** — `bc80825` (`feat(03-01)`) — adds 109 pure-addition lines to `evaluation/harness/score.py`: 3 imports + 1 module logger + `class NaNReasonTracer(BaseCallbackHandler)` (50 LOC) + `def _classify_post_evaluate_nan(...) -> Optional[str]` (30 LOC). Placed BETWEEN `_short_circuit_nan` (unchanged) and `_to_float_or_none` (unchanged). All 28 tests pass; verification gates 4–7 all PASS.
3. **Plan-metadata commit (this SUMMARY + STATE.md + ROADMAP.md + REQUIREMENTS.md)** — _(forthcoming, captures provenance for both atomic commits and the test-count delta)._

## Verification Gate Sweep

All 7 gates from `<verification>` pass:

1. ✅ RED commit exists: `git log --oneline | grep "test(03-01): RED"` → `e97e864 test(03-01): RED — failing tests for NaNReasonTracer + _classify_post_evaluate_nan`.
2. ✅ GREEN commit exists: `git log --oneline | grep "feat(03-01): GREEN"` → `bc80825 feat(03-01): GREEN — implement NaNReasonTracer + _classify_post_evaluate_nan`.
3. ✅ All test_eval_score.py tests pass: `pytest evaluation/tests/test_eval_score.py -x` exits 0 with `28 passed, 6 warnings in 3.27s`.
4. ✅ NaNReasonTracer is BaseCallbackHandler subclass: `python -c "from evaluation.harness.score import NaNReasonTracer; from langchain_core.callbacks import BaseCallbackHandler; assert issubclass(NaNReasonTracer, BaseCallbackHandler)"` exits 0.
5. ✅ _classify_post_evaluate_nan signature: `inspect.signature(...).parameters` returns `['row_idx', 'metric_name', 'metric_value', 'tracer']` exactly.
6. ✅ _short_circuit_nan unchanged: `pytest test_short_circuit_empty_contexts test_short_circuit_agent_truncated test_short_circuit_tier4_unavailable_with_contexts test_short_circuit_cached_miss -x` → 4/4 pass.
7. ✅ score_query_log untouched: `git diff evaluation/harness/score.py | grep "def score_query_log"` returns nothing (no signature change in any direction).

## Deviations from Plan

**None.** Plan executed exactly as written.

The pre-implementation `langchain_core` + `ragas.callbacks` import verification (one extra `python -c "..."` invocation against `.venv/bin/python`) was a sanity check, not a deviation — RESEARCH.md already documented these as transitive deps, and the verification confirmed it before any code was written.

The 6 pre-existing `DeprecationWarning`s emitted by `pytest` come from `score.py:202` (lazy `from ragas.metrics import faithfulness, answer_relevancy, context_precision`) and predate Plan 03-01 (visible in the Plan 02-04 final pytest run). RULE 4 says pre-existing warnings in unrelated paths are out of scope; tracked for future RAGAS migration work, not Plan 03-01.

## Open Questions / Follow-ups for Plan 03-02

- Wire `NaNReasonTracer` into `score_query_log`'s `evaluate(callbacks=[tracer], ...)` call so RAGAS streams chain events through the tracer during the per-tier scoring loop.
- Post-process each row's metrics dict through `_classify_post_evaluate_nan(orig_idx, metric_name, metric_value, tracer)` and write the resulting string to `ScoreRecord.nan_reason` for cells that are NaN (None after `_to_float_or_none`).
- Add an integration test with a stub LLM that intentionally raises `RagasOutputParserException` for one row × one metric, exercises the full evaluate() → tracer → classifier path, and asserts the resulting `ScoreRecord.nan_reason == 'json_parse_failure'`. Confirms the leaf-most-wins idempotence works with the real RAGAS callback chain (Plan 03-01 unit tests use direct method invocation; Plan 03-02 will exercise the real chain).
- Verify in Plan 03-02 that `_short_circuit_nan` precedence is preserved: pre-call NaN reasons (`empty_contexts`, `agent_truncated`, `tier4_unavailable`, `cached_miss`) MUST still take precedence over post-evaluate reasons (`empty_statements`, `json_parse_failure`, etc.) because the short-circuit fires BEFORE evaluate() runs.

## Self-Check: PASSED

- ✅ FOUND: `evaluation/harness/score.py` (modified, +109 LOC).
- ✅ FOUND: `evaluation/tests/test_eval_score.py` (modified, +253 LOC).
- ✅ FOUND: `.planning/phases/03-nan-reason-instrumentation/03-01-SUMMARY.md` (this file).
- ✅ FOUND commit: `e97e864 test(03-01): RED — failing tests for NaNReasonTracer + _classify_post_evaluate_nan`.
- ✅ FOUND commit: `bc80825 feat(03-01): GREEN — implement NaNReasonTracer + _classify_post_evaluate_nan`.

## TDD Gate Compliance

- ✅ RED gate: `test(03-01): RED ...` commit `e97e864` precedes implementation. RED state verified by failing pytest run with 15 ImportError failures BEFORE the GREEN commit.
- ✅ GREEN gate: `feat(03-01): GREEN ...` commit `bc80825` lands AFTER `e97e864` and brings the full test file to 28/28 passing.
- ⏭️ REFACTOR gate: Not executed — Plan 03-01's GREEN implementation was already minimal and well-organized (class + function placed in the canonical NaN-helper region, all docstrings co-located with code, no duplicate logic to extract). Plan 03-02 may refactor when wiring into score_query_log.
