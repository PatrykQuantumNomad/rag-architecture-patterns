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

## Plan 03-03 (2026-05-05)

### Process deviation routed through user-approved checkpoint:decision

**Live test deselect-by-default gap in pyproject.toml** (NEW v1.1 hardening item):

- **Issue surfaced when:** A previous agent (Plan 03-03 Task 1 verification) ran
  `pytest evaluation/tests/test_eval_smoke_live.py -v` (no `-m` filter, no `-s`)
  to verify the new `test_eval_smoke_nan_reasons` test was deselected by default.
  The pytest invocation's `-v` plus collection logic actually executed the new
  live test against the real OpenRouter judge, consuming ~$0.005-0.02 of API
  budget. Stdout was swallowed because `-s` was missing.
- **Root cause:** `pyproject.toml` registers the `live` marker under
  `[tool.pytest.ini_options].markers` but does NOT include
  `addopts = "-m 'not live'"`. The Phase 1 Plan 01-02 + Phase 2 Plan 02-03 +
  Plan 03-03 plans all assumed live-deselect-by-default semantics that pytest
  does not actually enforce without explicit `addopts`.
- **User-approved recovery:** Orchestrator surfaced the swallowed-stdout
  deviation as a `checkpoint:decision`. User approved option 1 (re-run the
  live test ONCE with `-s` to capture verbatim provenance). Total live cost
  across both invocations bounded at ~$0.028 — well under the $0.05 cost guard.
  Live verdict captured: `n_total=5 n_unknown_nan=0 n_scored_post_short_circuit=5`.
- **v1.1 hardening recommendation:** Add `addopts = "-m 'not live'"` to
  `[tool.pytest.ini_options]` so a bare `pytest path/to/file.py -v` (no -m flag)
  cannot accidentally consume API budget. Update plan templates + CLAUDE.md to
  use `-m live -k <test>` explicitly when invoking live tests.
- **Out of scope per RULE 4:** Pyproject.toml is not in Plan 03-03's
  files_modified list and the change has cross-cutting impact on every other
  live test in the repo. Tracked in STATE.md as a v1.1 deferred item; this plan
  ships exactly as written.

### Re-confirmed: judge cost ledger underreports on LiteLLM completions

Plan 03-03 live verdict reproduced the Plan 02-04 observation that
`token_usage_parser=get_token_usage_for_openai` does not surface usage from
LiteLLM `ModelResponse` bodies for some calls — the verbatim stdout shows
`judge_input_tokens=0 / judge_output_tokens=0` despite real spend
(~$0.014 estimated by call count). Same v1.1 follow-up tracked in STATE.md
since Plan 02-04. Cost guard is bounded by question×metric×call count
regardless: 5 × 3 × ~3 ≈ 45 calls × ~$0.0003 ≈ $0.014, well under $0.05.
