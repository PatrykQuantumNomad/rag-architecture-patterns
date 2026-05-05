# Deferred Items — Phase 03 Execution

Items discovered during Phase 03 plan execution that are out of scope per
RULE 4 / SCOPE BOUNDARY (only auto-fix issues directly caused by the
current task's changes; pre-existing failures in unrelated files are
out of scope).

## Plan 03-02 (2026-05-05)

### Pre-existing test failures (unrelated to score.py / compare.py / nan_reason)

1. **`evaluation/tests/test_eval_adapters.py::test_run_tier2_extracts_grounding`**
   - **Failure:** `AttributeError: 'str' object has no attribute 'get_secret_value'`
     at `evaluation/harness/adapters/tier_2.py:89`
   - **Cause:** Tier 2 adapter expects `settings.gemini_api_key` to be a
     `SecretStr` (with `.get_secret_value()`) but is receiving a plain `str`.
     Pydantic v2 settings handling change OR the test fixture overrides the
     field with a plain string.
   - **Verified pre-existing:** `git stash && pytest ...` reproduces the
     same failure on the pre-Plan-03-02 baseline.
   - **Owner / next action:** Tier 2 adapter maintenance (likely Phase 6
     embedder provenance work or earlier).

2. **`evaluation/tests/test_eval_smoke_live.py::test_eval_smoke_tier1_full_pipeline`**
   - **Failure:** `AssertionError: Smoke score was NaN with reason: empty_statements`
   - **Cause:** Live RAGAS run produces a `empty_statements` NaN on the
     Tier 1 5-question smoke; the test was written before HARN-05 wiring
     existed and asserts `sr.nan_reason is None`.
   - **Verified pre-existing:** `git stash && pytest ...` reproduces on
     pre-Plan-03-02 baseline. The Plan 03-02 wiring CORRECTLY classifies
     the NaN as `empty_statements` (i.e., the new instrumentation is
     working as designed). The test's tight assertion (`is None`) is the
     thing that needs to relax — that's exactly what Plan 03-03's live
     smoke backstop is for ("unknown_nan==0" instead of "nan_reason is None").
   - **Owner / next action:** Plan 03-03 will replace the strict assertion
     with the unknown_nan==0 backstop; this failure surfaces the gap that
     Plan 03-03 is designed to close.

Neither failure is caused by Plan 03-02's changes; both are out of scope
per RULE 4 and surface to be addressed in their respective owning plans.
