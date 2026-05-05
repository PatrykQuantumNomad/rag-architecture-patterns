# Phase 04 Deferred Items

Out-of-scope discoveries during Phase 04 execution. These predate this plan and
must NOT be auto-fixed per RULE 4 (cross-cutting impact, not in
`files_modified`).

| Discovery | File | Symptom | Why Deferred |
|---|---|---|---|
| `test_eval_adapters.py::test_run_tier2_extracts_grounding` fails on `main` (pre-existing) | `evaluation/harness/adapters/tier_2.py:89` | `AttributeError: 'str' object has no attribute 'get_secret_value'` — `settings.gemini_api_key` is a plain str in the test fixture context, but the adapter calls `.get_secret_value()` on it. Reproduced on a clean tree (`git stash` then re-run) — present BEFORE Plan 04-01 added any files. | Adapter file is not in Plan 04-01's `files_modified` list. Fix requires either (a) updating `tier_2.py` to handle both `SecretStr` and bare-`str` settings, or (b) updating the test fixture to wrap the key in `SecretStr`. Either change is a Phase 1/Phase 2 concern (Tier 2 adapter), not a Freeze Tool concern. Track as a separate Phase 1 follow-up. |

## Phase 4 RED-gate baseline note

The "existing test suite must remain green" gate (success-criteria item) is
evaluated against the `evaluation/tests/` set EXCLUDING the pre-existing
adapter failure above:

```
.venv/bin/pytest evaluation/tests/ \
  --ignore=evaluation/tests/test_eval_smoke_live.py \
  --ignore=evaluation/tests/test_eval_adapters.py \
  -x
```

Result: 78 passed (clean baseline). Plan 04-01 must keep this 78-passed
baseline intact while adding 10 new passing tests for freeze.py.
