---
phase: 01-tier-5-adapter-fix
plan: 02
subsystem: evaluation
tags: [smoke-gate, ragas, tier-5, retrieved-contexts, smoke-test, cli-flag, pydantic, live-test]

# Dependency graph
requires:
  - phase: 01-tier-5-adapter-fix
    provides: "Plan 01-01 — Tier 5 adapter walk over RunResult.new_items + RAG_DEBUG_TIER5_TRACING toggle (commit baaa573)"
  - phase: 132-research
    provides: "Pitfall 1 (item.output not raw_item) and Pitfall 5 (classify-before-denominator) guards exercised by the live smoke"
provides:
  - "--smoke-question-ids CLI flag on evaluation.harness.run with module-level DEFAULT_SMOKE_IDS constant (3 single-hop + 2 multi-hop, locked by D-03)"
  - "evaluation/harness/smoke_gate.py — pure-function classifier (classify_row) + gate (evaluate_smoke) + I/O wrapper (evaluate_smoke_from_paths) + __main__ CLI"
  - "SmokeGateResult Pydantic model recording verdict ∈ {PASS, FAIL, INCONCLUSIVE} with classified row counts and ratio"
  - "evaluation/tests/test_eval_smoke_gate.py — 6 unit tests pinning classification + gate behavior including the canonical Pitfall-5 case (3 populated + 1 truncated + 1 empty → PASS)"
  - "test_eval_smoke_tier5_full_pipeline live test mirroring the Tier 1 wiring under @pytest.mark.live"
  - "Live-smoke gate verdict of record: PASS (5/5 measurable populated, ratio 1.00, all RAGAS metrics non-NaN) — Tier 5 fix sticks; Phase 1 ships clean"
affects: [01-03, phase-7-full-rerun, phase-9-frozen-doc, ragas-rerun, blog-publication]

# Tech tracking
tech-stack:
  added: []  # No new runtime deps; reuses Pydantic v2 + existing harness modules
  patterns:
    - "Pattern: pure-function gate with I/O wrapper (evaluate_smoke vs evaluate_smoke_from_paths) — trivially unit-testable, CLI-thin"
    - "Pattern: Literal-typed verdict enums (Verdict, Classification) for machine-checkable provenance into SUMMARY.md"
    - "Pattern: shared compare._latest() newest-mtime resolution for query/score artifact discovery (no duplicated glob helpers)"
    - "Pattern: --smoke-question-ids filter applied BEFORE --limit in run.amain (preserves user-given ID order; rejects unknowns with exit code 2)"

key-files:
  created:
    - "evaluation/harness/smoke_gate.py"
    - "evaluation/tests/test_eval_smoke_gate.py"
    - ".planning/phases/01-tier-5-adapter-fix/01-02-SUMMARY.md"
  modified:
    - "evaluation/harness/run.py"
    - "evaluation/tests/test_eval_run.py"
    - "evaluation/tests/test_eval_smoke_live.py"

key-decisions:
  - "Live smoke gate PASS verdict on 2026-05-04 — 5/5 measurable populated, ratio 1.00, all RAGAS metrics non-NaN; Plan 01-03 fallback runbook remains dormant scaffolding for Phase 7"
  - "Tier 5 retrieved_contexts populated as expected — every entry in all 5 records starts with [paper_id=, n_ctx ranges 4-6, no Python repr leakage (Pitfall 1 of 132-RESEARCH guard intact end-to-end against live API)"
  - "DEFAULT_SMOKE_IDS locked to (single-hop-001, single-hop-002, single-hop-003, multi-hop-001, multi-hop-002) per D-03; same 5 IDs reused across Phase 1 (Tier 5) and Phase 2 (Tier 4) smokes"
  - "Smoke gate is a pure function over Pydantic-typed inputs (evaluate_smoke); I/O wrapper (evaluate_smoke_from_paths) reads via existing read_query_log + score.py's bare-list ScoreRecord JSON format"
  - "Capture cost $0.0048 and judge cost $0.00 — both well under the $0.05/run Pitfall-7 (132-RESEARCH) cost guard; smoke can be re-run cheaply for any future regression check"

patterns-established:
  - "Pattern: smoke gate as gate-of-record — the live test is a guardrail (collect-only verified), but the human-verify checkpoint is the authoritative landing point recorded in this SUMMARY"
  - "Pattern: ratio-on-measurable-subset gate (≥0.8) excludes Pitfall-9 honest empties and Pitfall-8 truncations from the denominator — naive 3/5=60% would FAIL but exclusion-aware gate PASSes"
  - "Pattern: SmokeGateResult JSON copy-pasted into the plan SUMMARY for Phase 7 to read; downstream phases never re-run the gate, they trust this artifact"

requirements-completed: [TIER-03]

# Metrics
duration: ~10min (Tasks 1+2 8 min on 2026-05-04T14:34Z–14:55Z; Task 3 checkpoint resolution + SUMMARY ~2 min on 2026-05-04T18:53Z)
completed: 2026-05-04
---

# Phase 1 Plan 02: Smoke Flag + Smoke Gate + Live Verification Summary

**Live Tier 5 smoke gate returned PASS (5/5 measurable populated, ratio 1.00, all RAGAS metrics non-NaN) — confirms Plan 01-01's adapter walk fix landed against the real OpenRouter API; Phase 1 ships clean and Plan 01-03 fallback runbook stays dormant scaffolding.**

## Smoke Gate Verdict (gate-of-record)

**Verdict: PASS** — recorded by the human-verify checkpoint on 2026-05-04 against live artifacts captured at 2026-05-04T18:48:17Z.

```json
{
  "verdict": "PASS",
  "n_total": 5,
  "n_populated": 5,
  "n_empty_no_tool_calls": 0,
  "n_agent_truncated": 0,
  "n_measurable": 5,
  "ratio": 1.0,
  "non_nan_faithfulness_count": 5,
  "non_nan_context_precision_count": 5,
  "message": "PASS: 5/5 measurable populated (ratio=1.00 >= 0.8); all RAGAS metrics non-NaN."
}
```

### Per-Row Classification

| qid             | n_ctx | classification | error | All `[paper_id=` prefix? |
| --------------- | ----- | -------------- | ----- | ------------------------- |
| single-hop-001  | 5     | populated      | null  | yes                       |
| single-hop-002  | 6     | populated      | null  | yes                       |
| single-hop-003  | 5     | populated      | null  | yes                       |
| multi-hop-001   | 4     | populated      | null  | yes                       |
| multi-hop-002   | 5     | populated      | null  | yes                       |

All 5 ScoreRecord rows have `nan_reason: null` (judge ran on every cell). Faithfulness range: 0.20–0.83. The metric values themselves are not the gate (the gate is "non-NaN"); the values feed Phase 7's full rerun comparison.

### Cost Guard (Pitfall 7 of 132-RESEARCH)

| Phase | Path | USD |
| ----- | ---- | --- |
| Capture (5 questions) | `evaluation/results/costs/tier-5-eval-20260504T184817Z.json` | $0.00480 |
| RAGAS judge (5 rows) | `evaluation/results/costs/ragas-judge-tier-5-20260504T184956Z.json` | $0.00000 |
| **Total** | — | **$0.00480** |

Both well below the $0.05/run budget (capture used 10,331 input + 680 output tokens via `google/gemini-2.5-flash`; judge cost recorded as $0.00 — likely free-tier or cached embeddings during scoring).

### Live Artifacts (referenced, not committed — gitignored runtime output)

- `evaluation/results/queries/tier-5-2026-05-04T18_48_17Z.json` (5 records, captured at git_sha=`ee0a394`)
- `evaluation/results/metrics/tier-5-2026-05-04T18_48_17Z.json` (5 ScoreRecord rows, all non-NaN)
- `evaluation/results/costs/tier-5-eval-20260504T184817Z.json`
- `evaluation/results/costs/ragas-judge-tier-5-20260504T184956Z.json`

**Why not committed:** `evaluation/results/queries/`, `metrics/`, and `costs/` are gitignored runtime artifacts (verified via `git check-ignore`); the per-run JSONs are reproducible by re-running `python -m evaluation.harness.run --tiers 5 --smoke-question-ids ... --yes`. Phase 7's full rerun will produce the canonical artifacts under a single git SHA.

## Performance

- **Duration:** ~10 min total (Tasks 1+2 ~8 min on 2026-05-04T14:34–14:55Z; checkpoint resolution + SUMMARY ~2 min on 2026-05-04T18:53Z)
- **Started:** 2026-05-04T14:34:00Z (Task 1 RED tests)
- **Completed:** 2026-05-04T18:55:00Z (this SUMMARY)
- **Tasks:** 3 (2 auto + 1 checkpoint:human-verify)
- **Files modified:** 3
- **Files created:** 3 (incl. this SUMMARY)

## Accomplishments

- Plan 01-01's adapter walk fix is end-to-end verified against the live OpenRouter API: 5/5 questions produce `retrieved_contexts` populated from real `ToolCallOutputItem.output` payloads, every entry provenance-prefixed `[paper_id=<id>]` (Pitfall 1 of 132-RESEARCH guard intact end-to-end).
- Tier 5's RAGAS NaN count drops from 30/30 baseline to 0/5 measurable on the smoke set (and the gate is engineered to handle the Pitfall-9 honest-empty case via exclusion-aware ratio when it occurs at 30-question scale).
- `--smoke-question-ids` flag + `DEFAULT_SMOKE_IDS` constant ready for reuse: Phase 2 (Tier 4 smoke) and any future regression check imports the same 5-tuple from `evaluation.harness.run` rather than redefining.
- `smoke_gate.py` is a pure-function module that any phase can call against an arbitrary `(QueryLog, list[ScoreRecord])` pair to get a typed `SmokeGateResult` — no shell-out, no I/O coupling beyond the optional `evaluate_smoke_from_paths` shim.
- D-04 measurable-threshold logic exercised: ratio = 5/5 = 1.00 ≥ 0.8 AND `non_nan_faithfulness_count == n_populated` AND `non_nan_context_precision_count == n_populated` → verdict PASS.
- Plan 01-03's fallback runbook (committed in `74663a7`) remains dormant — `[debug-tier5]` extras stay opt-in, no STACK.md mutations, no agent prompt changes (D-08 preserved).

## Task Commits

Each task was committed atomically:

1. **Task 1: `--smoke-question-ids` flag + `smoke_gate.py` module + unit tests (TDD)** — `ec0afd5` (feat)
2. **Task 2: Live smoke test mirroring Tier 1 — wires the full pipeline end-to-end (TDD)** — `ad788fc` (test)
3. **Task 3: User runs the live smoke and approves the gate verdict** — RESOLVED (PASS) by user verbatim reply `approved` on 2026-05-04 against artifacts at `evaluation/results/{queries,metrics,costs}/tier-5-*-2026-05-04T18*.json`. No commit (checkpoint outcome captured in this SUMMARY).

**Intermediate state commit:** `ee0a394` (docs: STATE.md mid-checkpoint snapshot showing 2/3 tasks committed and awaiting human verification — captures the git SHA used during the live capture run).

**Plan metadata:** Final commit recorded by the continuation agent — see git log for hash.

_Note: Task 1 followed TDD (RED was implicit in the failing imports / missing flag prior to commit; GREEN landed in `ec0afd5` with all 6 unit tests + 2 run-flag tests passing). Task 2 was TDD with collect-only verification (the live test itself is gated behind `@pytest.mark.live` and was exercised by the human-verify checkpoint, not by CI)._

## Files Created/Modified

- `evaluation/harness/run.py` — added `DEFAULT_SMOKE_IDS` module-level tuple constant, `--smoke-question-ids` argparse flag in `build_parser()`, and the filter block in `amain()` after `_load_golden_qa()` (rejects unknown IDs with exit code 2 + red-printed message).
- `evaluation/harness/smoke_gate.py` — NEW module (~140 LOC). Exports `Classification`, `Verdict`, `SmokeGateResult`, `classify_row`, `evaluate_smoke`, `evaluate_smoke_from_paths`, `__main__` CLI resolving latest artifacts via `compare._latest`.
- `evaluation/tests/test_eval_run.py` — added `test_run_smoke_ids_filters_correctly` and `test_run_smoke_ids_unknown_returns_2`.
- `evaluation/tests/test_eval_smoke_gate.py` — NEW (~310 LOC, 6 tests). Pins classification (`populated` / `empty_no_tool_calls` / `agent_truncated`) and the three verdict paths (PASS, FAIL, INCONCLUSIVE) including the canonical Pitfall-5 case (3 populated + 1 truncated + 1 empty → PASS not FAIL).
- `evaluation/tests/test_eval_smoke_live.py` — added `test_eval_smoke_tier5_full_pipeline` mirroring `test_eval_smoke_tier1_full_pipeline`. Reuses `live_eval_keys_ok` and `tier1_index_present` fixtures (Tier 5's `search_text_chunks` reads from Tier 1's ChromaDB; reusing the fixture is the correct skip path).

## Decisions Made

- **Gate-of-record is the human checkpoint, not the live test.** The live test (`test_eval_smoke_tier5_full_pipeline`) is a guardrail under `@pytest.mark.live` and is collect-only verified in CI. The authoritative PASS/FAIL/INCONCLUSIVE landing point is the `python -m evaluation.harness.smoke_gate --tier 5` invocation operated by the user at the checkpoint and recorded above.
- **Pure-function gate with thin I/O wrapper.** `evaluate_smoke(query_log, scores) -> SmokeGateResult` is fully testable with synthetic Pydantic inputs (no disk, no API). The CLI / live-test shim `evaluate_smoke_from_paths` does the I/O. This shape was driven by Pitfall 5 of 132-RESEARCH: classification must happen before the denominator is computed, and a pure function is the only way to make that contract trivially testable.
- **`compare._latest` newest-mtime resolution reused.** Both the `__main__` block in `smoke_gate.py` and the live test glob the queries/metrics dirs by tier prefix and pick the newest mtime — exactly the same idiom `compare.py` uses. No new glob helper introduced.
- **Cost guard at the SUMMARY level, not the gate level.** The cost ceiling ($0.05/run) is documented and verified by the operator at the checkpoint; it is NOT a hard exit in `smoke_gate.py` (which would tightly couple gate logic to cost-ledger format). The SUMMARY records the totals so Phase 7 can compare them against full-run costs.

## Deviations from Plan

None — plan executed exactly as written. Tasks 1+2 landed cleanly against the contracts; the live smoke run produced the happy-path verdict (PASS) on the first invocation, exercising the no-deviation-needed path.

The only worth-noting note is process-shaped, not code-shaped: a Plan 01-03 commit (`3e8c601`) inadvertently included in-progress Plan 01-02 test files due to shared git index across parallel executors (documented in `01-03-SUMMARY.md`). No content was lost or corrupted; Plan 01-02's `ec0afd5` re-stages those files atomically with the rest of Task 1's changes. The lesson is captured in 01-03-SUMMARY for future parallel-execution plans (use git worktrees per executor or scope the index per commit).

---

**Total deviations:** 0 from the plan contract.
**Impact on plan:** None. Plan 01-02 shipped with the gate verdict matching the optimistic expectation: the 5-line adapter root cause from Plan 01-01 was indeed the entire bug, the live API behavior is consistent with the unit-test mocks, and the harness scaffolding (DEFAULT_SMOKE_IDS, smoke_gate, live test) is ready to be reused by Phase 2 (Tier 4 smoke) verbatim.

## Issues Encountered

None — all 6 unit tests passed on first GREEN run for Task 1; the live smoke completed in under 2 minutes with no transient API failures; the gate produced the canonical PASS message without any need for the INCONCLUSIVE escape hatch.

The Plan 01-01 deferred-items issue (pre-existing tier-2 SecretStr test failure in `evaluation/tests/test_eval_adapters.py`) is unchanged by this plan and remains tracked in `deferred-items.md` for v1.1 cleanup.

## Verification Results

All 6 success-criteria items confirmed green:

1. **Goal-backward truth #1** — `python -m evaluation.harness.run --tiers 5 --smoke-question-ids single-hop-001,single-hop-002,single-hop-003,multi-hop-001,multi-hop-002 --yes` produced exactly 5 records in the captured query log; user-supplied ID order preserved.
2. **Goal-backward truth #2** — `from evaluation.harness.run import DEFAULT_SMOKE_IDS` succeeds; tuple equals `("single-hop-001", "single-hop-002", "single-hop-003", "multi-hop-001", "multi-hop-002")` per D-03.
3. **Goal-backward truth #3** — `--smoke-question-ids` filter rejects unknown IDs with exit code 2 and red-printed `Unknown question ids: [...]` message (verified by `test_run_smoke_ids_unknown_returns_2` unit test).
4. **Goal-backward truths #4–#6** — `smoke_gate.evaluate_smoke` is a pure function returning typed `SmokeGateResult`; ratio-on-measurable threshold exercised at 0.8 (4/5 equivalent); INCONCLUSIVE verdict path covered by `test_gate_inconclusive_too_few_measurable`.
5. **Goal-backward truth #7** — `test_eval_smoke_tier5_full_pipeline` is collected under `pytest -m live --collect-only`; both Tier 1 and Tier 5 live smokes are visible.
6. **Phase-gate checkpoint outcome recorded** — PASS verdict captured above with full SmokeGateResult JSON; Plan 01-02 ships; Plan 01-03 fallback scaffolding remains dormant per D-06 (instrument-first ordering not invoked).

## Threat Model Compliance

| Threat ID | Disposition | Status |
|-----------|-------------|--------|
| T-02-01 (T) — `--smoke-question-ids` argv injection | accept | Confirmed: argparse + `.split(",")` produces strings only used as `dict[str, dict]` keys; no shell eval surface. |
| T-02-02 (I) — live test prints `OPENROUTER_API_KEY` | mitigate | Confirmed: `live_eval_keys_ok` fixture reads via `os.environ.get`; the new tier-5 live test follows the same pattern (no echoing). |
| T-02-03 (D) — adapter explodes contexts → cost spike | mitigate | Confirmed: capture cost $0.00480 << $0.05 budget; Plan 01-01's first-occurrence-wins dedup keeps `len(retrieved_contexts) ≤ 6` per question. |
| T-02-04 (T) — gate bypass via hand-crafted metrics JSON | accept | `_latest()` mtime resolution is intentional; no adversarial threat model. |
| T-02-05 (R) — gate result non-determinism | mitigate | Confirmed: `evaluate_smoke` is pure over Pydantic types; verdict reproducible from the on-disk artifacts. The live API call is the only non-deterministic surface and is bounded by the INCONCLUSIVE escape hatch. |

## User Setup Required

None — no external service configuration required by Plan 01-02 itself. The `OPENROUTER_API_KEY` was already configured for Plan 01-01's live testing and was reused for the smoke run. The `[debug-tier5]` extras from Plan 01-03 are NOT installed and remain opt-in; default install is byte-identical.

## Next Phase Readiness

- **Phase 1 ready for verification.** All 3 plans (01-01, 01-02, 01-03) complete. Phase verification (or chain-advance to Phase 2) is the next step.
- **Phase 2 (Tier 4 Graphml Regeneration) inherits the harness scaffolding.** `DEFAULT_SMOKE_IDS` and `smoke_gate.evaluate_smoke` are reusable verbatim — Phase 2's smoke gate just imports them and points at `tier-4-*.json` artifacts. No re-implementation needed.
- **Phase 7 (Full 5-Tier Rerun) reads this SUMMARY** to confirm the Tier 5 fix landed before committing the rerun budget. The PASS verdict + SmokeGateResult JSON above is the authoritative artifact; Phase 7 does NOT re-run the smoke gate, it trusts this record.
- **Plan 01-03 fallback runbook stays dormant.** No `evaluation/results/diagnostics/tier-5-fallback-*.json` artifact was produced by this plan — the runbook was not exercised because the smoke PASSed on first try. Phase 9's frozen handoff doc records "Tier 5 ships clean; no fallback invocation".
- **No blockers.** STACK.md decision tree (simplify schema → switch model slug → bump openai-agents) is NOT needed.

## Self-Check: PASSED

**Files exist:**
- FOUND: `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/smoke_gate.py`
- FOUND: `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/harness/run.py` (modified)
- FOUND: `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_smoke_gate.py`
- FOUND: `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_run.py` (modified)
- FOUND: `/Users/patrykattc/work/git/rag-architecture-patterns/evaluation/tests/test_eval_smoke_live.py` (modified)

**Live artifacts exist (read-only — referenced, not committed):**
- FOUND: `evaluation/results/queries/tier-5-2026-05-04T18_48_17Z.json` (16100 bytes, 5 records)
- FOUND: `evaluation/results/metrics/tier-5-2026-05-04T18_48_17Z.json` (825 bytes, 5 ScoreRecord rows)
- FOUND: `evaluation/results/costs/tier-5-eval-20260504T184817Z.json` ($0.00480 total)
- FOUND: `evaluation/results/costs/ragas-judge-tier-5-20260504T184956Z.json` ($0.00 total)
- VERIFIED: `git check-ignore evaluation/results/queries/tier-5-2026-05-04T18_48_17Z.json` returns the path (gitignored)

**Commits exist:**
- FOUND: `ec0afd5` (feat: Task 1 — flag + smoke_gate + unit tests)
- FOUND: `ad788fc` (test: Task 2 — live smoke test)
- FOUND: `ee0a394` (docs: STATE.md mid-checkpoint snapshot)

**Verification grep guards:**
- PASS: `jq '.records[] | (.retrieved_contexts | length)'` returns `5,6,5,4,5` — all ≥ 1 (D-04 measurable threshold)
- PASS: `jq '.records[] | [.retrieved_contexts[] | startswith("[paper_id=")] | all'` returns `true` for all 5 — Pitfall 1 of 132-RESEARCH guard intact end-to-end
- PASS: `jq '[.[] | .nan_reason] | unique'` on metrics file returns `[null]` — no NaN exclusions (n_measurable = n_total = 5)
- PASS: `python -m evaluation.harness.smoke_gate --tier 5` returns the SmokeGateResult JSON quoted above on demand (idempotent, ~10ms, no API calls)

---

*Phase: 01-tier-5-adapter-fix*
*Plan: 02*
*Completed: 2026-05-04*
