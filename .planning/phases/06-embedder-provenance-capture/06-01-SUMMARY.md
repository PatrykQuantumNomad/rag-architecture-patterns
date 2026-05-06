---
phase: 06-embedder-provenance-capture
plan: 01
subsystem: evaluation
tags: [pydantic, ragas, evaluation-harness, provenance, embedder, capture]

# Dependency graph
requires:
  - phase: 05-pipeline-driver
    provides: pipeline.py end-to-end composition (in-process; Phase 5 forward
      contract preserved byte-identical)
  - phase: 04-freeze-tool
    provides: freeze.py(version, force, results_dir, source) -> Path contract +
      per-tier manifest entry (extended here with embedder fields)
provides:
  - QueryLog.embedder + QueryLog.embedder_source (Optional[str]) — schema
    extension carried by every Phase 7 capture JSON
  - Per-tier EMBEDDER_SOURCE constants in 5 tier modules + EMBED_MODEL in
    tier-2-managed (single source of truth per tier)
  - run.py _capture_tier threads (embedder, source) per tier branch
  - tier-4-multimodal/scripts/eval_capture.py threads (embedder, source) at
    the live capture path (D-CAPTURE-ENTRYPOINTS — Tier 4 has TWO paths)
  - compare.py per-tier embedder line + dedicated "Embedder by tier"
    markdown table block (D-Q1; Phase 9 becomes pure copy-paste)
  - freeze.py per_tier manifest entries carry embedder + embedder_source
affects: [07-full-rerun, 08-judge-spot-check, 09-frozen-handoff-doc]

# Tech tracking
tech-stack:
  added: []  # Pure repo-internal instrumentation; no new dependencies
  patterns:
    - "Pydantic v2 Optional[T]=None for backwards-compatible schema extension
       (legacy JSONs load with None; D-BACKCOMPAT)"
    - "Single-source-of-truth constants per tier (EMBED_MODEL +
       EMBEDDER_SOURCE colocated with the tier module's embed code)"
    - "Static-source assertion as a fragility-free alternative to RAG
       stack mocking (Task 4 test pattern — recommended in 06-RESEARCH.md)"
    - "Derived-flag rendering: 'Managed' column computed in compare.py from
       embedder_source == 'google-managed'; no third schema field"

key-files:
  created:
    - .planning/phases/06-embedder-provenance-capture/06-01-SUMMARY.md
  modified:
    - evaluation/harness/records.py (84 -> 86 LOC, +2 fields on QueryLog)
    - evaluation/harness/run.py (391 -> 420 LOC, +5 imports + 5 src/source
      assigns + 2 QueryLog kwargs across all 5 tier branches)
    - evaluation/harness/compare.py (404 -> 425 LOC, aggregate_tier
      pass-through + emit_markdown footer + Embedder-by-tier table)
    - evaluation/harness/freeze.py (95 -> 97 LOC, +2 manifest entries)
    - tier-4-multimodal/scripts/eval_capture.py (236 -> 243 LOC,
      EMBEDDER_SOURCE import + 2 QueryLog kwargs)
    - tier-1-naive/embed_openai.py (80 -> 81 LOC)
    - tier-2-managed/main.py (347 -> 353 LOC, +EMBED_MODEL + EMBEDDER_SOURCE)
    - tier-3-graph/rag.py (173 -> 175 LOC, +EMBEDDER_SOURCE + __all__ entry)
    - tier-4-multimodal/rag.py (239 -> 241 LOC, +EMBEDDER_SOURCE + __all__)
    - tier-5-agentic/tools.py (171 -> 178 LOC, +EMBEDDER_SOURCE locally
      declared; EMBED_MODEL re-imported from Tier 1 — Pitfall 4)
    - evaluation/tests/test_eval_records.py (+82 LOC, 3 new tests)
    - evaluation/tests/test_eval_run.py (+153 LOC, 1 parametrized test ×
      5 sub-cases)
    - evaluation/tests/test_eval_tier4.py (+36 LOC, 1 new test)
    - evaluation/tests/test_eval_compare.py (+145 LOC, 3 new tests + 2
      private helpers)
    - evaluation/tests/test_eval_freeze.py (+59 LOC net, 1 new test +
      _build_fixture extension with with_embedder flag)

key-decisions:
  - "D-ROADMAP-OVERRIDE locked: Tier 5 records (openai/text-embedding-3-small,
     openrouter) — IDENTICAL to Tier 1. ROADMAP's 'OpenAI hosted vector-store
     + managed=true' wording is factually wrong (verified in 06-RESEARCH.md
     against tier-5-agentic/tools.py:47-50,90-101)."
  - "D-Q1: Embedder-by-tier disclosure table emitted in compare.py during
     Phase 6 (+~10 LOC), NOT deferred to Phase 9. Phase 9 becomes pure
     copy-paste from comparison.md."
  - "D-Q2: Tier 2's EMBED_MODEL + EMBEDDER_SOURCE added to tier-2-managed/
     main.py (sibling to INDEX_USD_PER_1M_TOKENS at line 91); no new
     tier-2-managed/embed.py module (Google File Search has no separate
     embed call to host)."
  - "D-Q3: Pydantic field name is `embedder` (single word), mirroring
     `judge_emb` in score.py / freeze.py manifest convention."
  - "D-BACKCOMPAT: Optional[str]=None default + defensive .get() in
     compare.py / freeze.py emit em-dash for legacy QueryLog JSONs (no
     migration tool — Phase 7 re-captures all 5 tiers)."
  - "D-CAPTURE-ENTRYPOINTS: BOTH evaluation/harness/run.py AND
     tier-4-multimodal/scripts/eval_capture.py are edited; each gets a
     dedicated regression test (Pitfall 1 of 06-RESEARCH.md)."
  - "D-NO-LIVE-SMOKE: Pure-offline TDD. Every embedder identifier is a
     static literal known at code-edit time. Phase 7's 5-tier rerun
     naturally exercises the new plumbing in production."
  - "D-LOC: All 10 files honor explicit max_lines budgets; raw `wc -l`
     verified at every commit (per Plan 04-01 / 05-01 lesson)."

patterns-established:
  - "Pattern: Per-tier provenance constants (EMBED_MODEL + EMBEDDER_SOURCE)
     colocated with the tier module's embed code. Reusable for any future
     per-tier facet capture (e.g., chunking strategy, retriever variant)."
  - "Pattern: Static-source assertion test (Task 4) when mocking a heavy
     library stack adds no signal beyond confirming a literal in source."
  - "Pattern: Derived-flag rendering (compare.py 'Managed' column from
     embedder_source) — keeps the schema minimal while surfacing semantic
     classifications at presentation time."

# Metrics
duration: 18min
completed: 2026-05-06
---

# Phase 6 Plan 06-01: Embedder Provenance Capture Summary

**Per-tier embedder model + provenance gateway recorded in every capture
JSON, surfaced in comparison.md (line + dedicated table), threaded through
freeze.py per_tier manifest, and locked at unit-test level — closes CAP-03
end-to-end at offline level. Phase 7's rerun is unblocked.**

## Performance

- **Duration:** ~18 min wall time (single executor session)
- **Started:** 2026-05-06 (HEAD 074e668 pre-execution)
- **Completed:** 2026-05-06 (HEAD 5d13879 post-GREEN, pre-final-docs commit)
- **Tasks:** 7 (6 implementation + 1 verification gate)
- **Source files modified:** 10 (10 budget-tracked) + 5 test files
- **Test count delta:** 103 -> 116 passing offline tests (+13: 2 records
  schema + 1 records constants + 5 run-parametrized + 1 tier-4 + 3 compare
  + 1 freeze)

## Accomplishments

- **CAP-03 closed at unit + integration levels** for both capture
  entrypoints (run.py + eval_capture.py).
- **D-ROADMAP-OVERRIDE locked**: Tier 5 records the SAME embedder+source as
  Tier 1 (openai/text-embedding-3-small / openrouter). The frozen-doc
  embedder-confound disclosure will be data-backed by Phase 7's captures —
  no narrative claim about a hosted vector store survives.
- **Phase 9 cost ceiling cut**: comparison.md now emits the
  "Embedder by tier" table directly; Phase 9's frozen doc is a pure
  copy-paste with zero new logic.
- **Forward contract preserved byte-identical**: score.py, pipeline.py,
  smoke_gate.py — `git diff` reports 0 bytes total across all three files.
- **All 10 LOC budgets honored** (validated via raw `wc -l` per commit;
  Plan 04-01 / 05-01 lesson worked first try, ZERO compression iterations).

## Task Commits

Each task was committed atomically — RED + GREEN as separate commits per
Plan 04-01 / 05-01 precedent (12 implementation commits + 1 docs commit
for SUMMARY/STATE/ROADMAP/REQUIREMENTS = 13 total):

1. **Task 1: QueryLog Optional embedder fields** — `a6c3155` (test:RED) +
   `2980d5f` (feat:GREEN)
2. **Task 2: Per-tier EMBEDDER_SOURCE constants** — `30d8e1c` (test:RED) +
   `36bfb9d` (feat:GREEN)
3. **Task 3: run.amain writes embedder per tier** — `740bfd2` (test:RED) +
   `bdc00d7` (feat:GREEN)
4. **Task 4: tier-4 eval_capture writes embedder fields** — `df5b2e9`
   (test:RED) + `5acdc07` (feat:GREEN)
5. **Task 5: compare embedder line + table** — `47d4fe3` (test:RED) +
   `0ed546d` (feat:GREEN)
6. **Task 6: freeze manifest carries embedder per tier** — `c571530`
   (test:RED) + `5d13879` (feat:GREEN)
7. **Task 7: full-suite gate** — verification-only; rolled into the final
   docs commit (this SUMMARY + STATE/ROADMAP/REQUIREMENTS updates).

## Files Created/Modified

### Source (10 files, +80 net LOC)

| File | Before | After | Cap | Slack |
|------|--------|-------|-----|-------|
| evaluation/harness/records.py | 84 | 86 | 95 | 9 |
| evaluation/harness/run.py | 391 | 420 | 420 | 0 (exactly at cap) |
| evaluation/harness/compare.py | 404 | 425 | 435 | 10 |
| evaluation/harness/freeze.py | 95 | 97 | 105 | 8 |
| tier-4-multimodal/scripts/eval_capture.py | 236 | 243 | 245 | 2 |
| tier-1-naive/embed_openai.py | 80 | 81 | 85 | 4 |
| tier-2-managed/main.py | 347 | 353 | 355 | 2 |
| tier-3-graph/rag.py | 173 | 175 | 178 | 3 |
| tier-4-multimodal/rag.py | 239 | 241 | 245 | 4 |
| tier-5-agentic/tools.py | 171 | 178 | 178 | 0 (exactly at cap) |

### Tests (5 files, +480 net LOC)

- `evaluation/tests/test_eval_records.py` — +3 tests (schema round-trip,
  legacy load, per-tier constants importable).
- `evaluation/tests/test_eval_run.py` — +1 parametrized test × 5 sub-cases
  (one per tier 1-5).
- `evaluation/tests/test_eval_tier4.py` — +1 static-source test.
- `evaluation/tests/test_eval_compare.py` — +3 tests + 2 helpers
  (footer line, em-dash legacy fallback, embedder-by-tier table).
- `evaluation/tests/test_eval_freeze.py` — +1 test + extended `_build_fixture`
  with `with_embedder=True` flag.

## Decisions Made

D-ROADMAP-OVERRIDE callout (mandated by plan context):

> ROADMAP Phase 6 Success Criterion 2 wording for Tier 5 ("OpenAI's hosted
> vector-store embedder with explicit `managed=true` flag") is **factually
> wrong** and **OVERRIDDEN by this plan**. Verified in 06-RESEARCH.md
> against `tier-5-agentic/tools.py:47-50,90-101`: Tier 5 reuses Tier 1's
> local ChromaDB and embeds via OpenRouter `openai/text-embedding-3-small`
> — IDENTICAL to Tier 1. Tier 5 records `embedder="openai/text-embedding-3-small"`
> and `embedder_source="openrouter"`. The `managed=true` boolean flag is
> REJECTED in favor of deriving "managed" as
> `embedder_source == "google-managed"` in compare.py rendering — only
> Tier 2 is "managed" (Google File Search owns the indexing).

Final embedder-by-tier table (verified in CLI-produced comparison.md from
the Task 7 E2E gate):

```
| Tier | Embedder Model | Source | Managed |
|------|----------------|--------|---------|
| tier-1 | openai/text-embedding-3-small | openrouter | no |
| tier-2 | gemini-embedding-001 | google-managed | yes |
| tier-3 | openai/text-embedding-3-small | openrouter | no |
| tier-4 | openai/text-embedding-3-small | openrouter | no |
| tier-5 | openai/text-embedding-3-small | openrouter | no |
```

All 7 plan-locked decisions (D-ROADMAP-OVERRIDE, D-Q1, D-Q2, D-Q3,
D-BACKCOMPAT, D-CAPTURE-ENTRYPOINTS, D-NO-LIVE-SMOKE, D-LOC) honored
verbatim.

## Deviations from Plan

**None — plan executed exactly as written.**

Two minor implementation refinements that did NOT depart from the plan
(both endorsed by the plan's text):

1. **Task 4 test approach: static-source assertion (recommended option)**.
   The plan offered two paths for `test_eval_capture_writes_embedder_for_tier_4`:
   (a) heavy mock of build_rag/run_query/write_query_log, or
   (b) static-source assertion on eval_capture.py text.
   The plan EXPLICITLY recommended (b) ("Recommended approach if mocking
   is fragile."). Picked (b). Not a deviation — explicit plan-permitted choice.

2. **Task 7 grep audit had ONE match (negative-rejection comment)**.
   `tier-5-agentic/tools.py` contains a comment that explicitly REJECTS
   the wrong string (`# Tier 1), NOT "openai-hosted-managed".`). The
   plan-defined audit pattern matched this comment because the pattern
   was substring-only. Re-running the audit with `grep -vE 'NOT[[:space:]]+"openai-hosted'`
   yielded CLEAN. The comment is a feature (locks D-ROADMAP-OVERRIDE in
   source code as documentation), not a violation. Not a deviation —
   the audit's intent (no production code claims Tier 5 is hosted) is
   satisfied.

**Total deviations:** 0 (zero auto-fixes; plan was executable as written).

**Impact on plan:** None. Plan 04-01 / 05-01 LOC-budget validation
discipline (raw `wc -l` before locking max_lines) held a third time —
zero compression iterations needed across 10 LOC-tracked files.

## Issues Encountered

None substantive. One minor environment quirk:

- **Shell `==` parse warning**: `(eval):1: == not found` appeared in the
  bash output of every test/lint pipeline that included `=== marker ===`
  log separators. Cosmetic only; tests + audits ran cleanly. The shell
  treats `==` as an attempted comparison operator at the start of an
  echo'd string under certain conditions; harmless.

## Self-Check: PASSED

Verified post-execution:

| Gate | Expected | Actual | Status |
|------|----------|--------|--------|
| All 6 GREEN commits exist | 6 | 6 | ✓ |
| All 6 RED commits exist | 6 | 6 | ✓ |
| evaluation/harness/records.py | <=95 LOC | 86 | ✓ |
| evaluation/harness/run.py | <=420 LOC | 420 | ✓ (at cap) |
| evaluation/harness/compare.py | <=435 LOC | 425 | ✓ |
| evaluation/harness/freeze.py | <=105 LOC | 97 | ✓ |
| tier-4-multimodal/scripts/eval_capture.py | <=245 LOC | 243 | ✓ |
| tier-1-naive/embed_openai.py | <=85 LOC | 81 | ✓ |
| tier-2-managed/main.py | <=355 LOC | 353 | ✓ |
| tier-3-graph/rag.py | <=178 LOC | 175 | ✓ |
| tier-4-multimodal/rag.py | <=245 LOC | 241 | ✓ |
| tier-5-agentic/tools.py | <=178 LOC | 178 | ✓ (at cap) |
| Byte-identical guard (score+pipeline+smoke_gate) | 0 bytes | 0 bytes | ✓ |
| Offline test sweep | All PASS | 116 passed, 1 deselected | ✓ |
| ROADMAP-override grep audit | CLEAN (production) | CLEAN | ✓ |
| E2E CLI compare emits "Embedder by tier" | yes | yes | ✓ |
| E2E CLI freeze manifest carries embedder | yes | yes | ✓ |

## Next Phase Readiness

Phase 7 (Full 5-Tier Rerun) is unblocked. The next pipeline run via
`python -m evaluation.harness.pipeline ...` will populate `embedder` and
`embedder_source` on every capture JSON of record. comparison.md will
auto-emit the per-tier embedder line + the dedicated "Embedder by tier"
table; freeze.py's manifest will carry per_tier embedder fields for the
v1.0 frozen ship-artifact.

**No legacy migration required**. Existing local-only JSONs in
evaluation/results/queries/ (gitignored) will load with embedder=None and
render em-dash placeholders until Phase 7 re-captures.

**No live smoke needed for Phase 6 closure.** All embedder identifiers
are static literals; pure-offline TDD covers 100% of the new behavior
(D-NO-LIVE-SMOKE). Phase 7's natural live capture is the regression check.

---
*Phase: 06-embedder-provenance-capture*
*Completed: 2026-05-06*
