# Deferred Items — Phase 01 (Tier 5 Adapter Fix)

Out-of-scope discoveries logged during plan execution. NOT fixed in this phase
because they are unrelated to the load-bearing 5-line `tier_5.py:125` fix.

## Pre-existing test failure: `test_run_tier2_extracts_grounding`

- **File:** `evaluation/harness/adapters/tier_2.py:89`
- **Error:** `AttributeError: 'str' object has no attribute 'get_secret_value'`
- **Root cause:** The test mocks `gemini_api_key` as a plain string, but the
  adapter calls `.get_secret_value()` (pydantic `SecretStr` accessor). Either
  the test mock is stale or the production type was changed without updating
  tests.
- **Discovered during:** Phase 01 Plan 01 GREEN-phase pytest run (2026-05-04).
- **Verified pre-existing:** `git stash && pytest …test_run_tier2_extracts_grounding`
  reproduces the failure on the unmodified working tree.
- **Disposition:** Out-of-scope for Phase 01 (Tier 5 only). Track for a Tier 2
  cleanup pass; the test mock should construct a real `SecretStr` (or the
  adapter should accept either) so the unit suite is fully green.
