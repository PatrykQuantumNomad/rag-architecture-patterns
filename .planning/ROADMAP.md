# Roadmap: RAG Architecture Patterns — v1.0

## Overview

Fix two ship-blocker bugs (Tier 5 `empty_contexts`, Tier 4 graphml regen), add a small amount of harness scaffolding (`freeze.py` + `pipeline.py` + per-row NaN reason + embedder provenance), then re-run all 5 tiers on a single date with a single git SHA, run a multi-judge spot-check to bound family bias, and produce one immutable frozen markdown doc (`evaluation/results/frozen/eval-numbers-v1.0.md`) that copy-pastes into the external blog repo with full provenance and honest disclaimers. Phases 1 and 2 are parallel-friendly bug fixes. Phases 3–6 are parallel-friendly harness/provenance work that depends on neither tier fix. Phases 7–9 are strictly sequential and gate on everything before them.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Tier 5 Adapter Fix** - Walk `RunResult.new_items` for `ToolCallOutputItem.output` so Tier 5 stops returning 30/30 `empty_contexts`, smoke-tested before any full rerun ✓ 2026-05-04 (smoke PASS 5/5, ratio 1.00)
- [x] **Phase 2: Tier 4 Graphml Regeneration** - Wipe `rag_anything_storage/tier-4-multimodal/`, re-ingest from MineRU JSON parsed outside the sandbox, smoke-tested before any full rerun ✓ 2026-05-05 [4/4 plans delivered; gap closure landed via Plan 02-04: JUDGE_MAX_TOKENS=8192 in score._build_judge → Tier 4 smoke PASS (5/5 faithfulness=1.0) AND Tier 5 smoke PASS (Phase 1 regression check intact)]
- [x] **Phase 3: NaN Reason Instrumentation** - Distinguish `empty_contexts` vs `empty_statements` vs `json_parse_failure` in per-row metrics output ✓ 2026-05-05 [3/3 plans delivered; HARN-05 closed at unit (03-01) + integration (03-02) + live (03-03) levels; live smoke verdict PASS, n_unknown_nan=0 against real Gemini 2.5 Flash on 5-question Tier 5 capture, ~$0.014 cost vs $0.05 cost guard]
- [ ] **Phase 4: Freeze Tool** - `evaluation/harness/freeze.py` writes immutable `frozen/eval-numbers-vX.Y.md` + sidecar manifest with git SHA, capture timestamps, library versions
- [ ] **Phase 5: Pipeline Driver** - `evaluation/harness/pipeline.py` runs capture → score → compare → freeze in one command, with single-tier rerun support
- [ ] **Phase 6: Embedder Provenance Capture** - Per-tier embedder model name recorded in capture JSON so the embedder-confound disclosure is data-backed
- [ ] **Phase 7: Full 5-Tier Rerun** - Capture all 5 tiers × 30 questions on one date with one git SHA, NaN counts down to expected residuals
- [ ] **Phase 8: Multi-Judge Spot-Check** - Re-score 5 questions × 3 tiers with a non-Gemini judge, capture delta in structured JSON
- [ ] **Phase 9: Frozen Handoff Doc** - Produce `evaluation/results/frozen/eval-numbers-v1.0.md` containing rollup, per-class breakdown, per-tier provenance, multi-judge delta, embedder table, and honest disclaimers

## Phase Details

### Phase 1: Tier 5 Adapter Fix
**Goal**: Tier 5 evaluation produces non-empty `retrieved_contexts` extracted from agent tool outputs, verified on a 5-question smoke test before any rerun budget is committed.
**Depends on**: Nothing (first phase, parallel with Phase 2)
**Requirements**: TIER-01, TIER-03 (Tier 5 portion)
**Success Criteria** (what must be TRUE):
  1. User can run `python -m evaluation.harness.run --tiers 5` against 5 smoke-test questions and see `retrieved_contexts` populated from `ToolCallOutputItem.output` (snippets from `search_text_chunks` / abstracts from `lookup_paper_metadata`) in the resulting `queries/tier-5-*.json`
  2. User can confirm via `evaluation/harness/score.py` that the smoke-test 5 questions produce <5/5 `empty_contexts` NaNs (down from 30/30 baseline)
  3. User can read `evaluation/harness/adapters/tier_5.py` and see the hard-coded `retrieved_contexts=[]` at line 125 replaced by an iteration over `result.new_items` filtered by `ToolCallOutputItem`
  4. If the smoke test still shows empty contexts, the diagnosis decision tree (simplify schema → switch model slug → bump openai-agents) is followed in order before declaring the fix incomplete
**Plans**: 3 plans (2 waves)
- [x] 01-01-PLAN.md — Adapter walk + conditional tracing toggle (TIER-01) — completed 2026-05-04 (commit baaa573)
- [x] 01-02-PLAN.md — Smoke flag + smoke gate + live test + human-verify checkpoint (TIER-03) — completed 2026-05-04 (commits ec0afd5, ad788fc; live smoke gate verdict PASS)
- [x] 01-03-PLAN.md — Fallback scaffolding ([debug-tier5] extra + diagnostics.py + runbook) — completed 2026-05-04 (commits 4be0f0a, 3e8c601, 99e02b0; runbook dormant — smoke PASSed)

### Phase 2: Tier 4 Graphml Regeneration
**Goal**: Tier 4 has a clean, verifiable LightRAG graph rebuilt from MineRU-parsed JSON (parsed outside the sandbox per Phase 139 evidence), verified on a 5-question smoke test before any rerun budget is committed.
**Depends on**: Nothing (first phase, parallel with Phase 1)
**Requirements**: TIER-02, TIER-03 (Tier 4 portion)
**Success Criteria** (what must be TRUE):
  1. User can confirm `rag_anything_storage/tier-4-multimodal/` was deleted (not moved) and rebuilt from a MineRU pass run on the host machine outside the sandbox
  2. User can verify post-ingest that `graph_chunk_entity_relation.graphml` exists, is non-empty, and node/edge counts are logged for provenance
  3. User can run `python -m evaluation.harness.run --tiers 4 --tier-4-from-cache` against 5 smoke-test questions and see populated `retrieved_contexts` in the resulting `queries/tier-4-*.json`
  4. User can confirm the smoke-test 5 questions produce <5/5 `empty_contexts` NaNs (down from 30/30 baseline)
  5. The MineRU + LightRAG + RAG-Anything library versions are recorded so the graph state is reproducible
**Plans**: 4 plans (3 waves; 02-02 is a parallel non-blocking track for Phase 7 prep — does NOT gate Phase 2 ship; 02-04 is gap closure for the 02-03 smoke FAIL)
- [x] 02-01-PLAN.md — Wipe + ingest_from_mineru.py + log_graph_stats.py + provenance manifest (TIER-02; 3 tasks — Task 1 ingest helper, Task 2a stats helper + tests, Task 2b live rebuild + manifest) — **COMPLETE 2026-05-05** (smoke-only, 3 papers, Option B)
- [x] 02-02-PLAN.md — parse_missing_papers.py + host MineRU top-up checkpoint for 4 missing golden_qa papers (TIER-02; **parallel non-blocking — Phase 7 prep**, may resolve as `verdict=descoped` without blocking Phase 2 ship) — **COMPLETE 2026-05-05** (4/4 papers parsed on host, ~49 min)
- [x] 02-03-PLAN.md — eval_capture --smoke-question-ids + Tier 4 live smoke test + autonomous capture/score/gate (TIER-03; depends on 02-01 only — smoke-set source papers are in the existing 75-paper cache rebuilt by 02-01) — **COMPLETE 2026-05-05** (deliverables shipped; smoke gate FAIL on judge max_tokens routed to Plan 02-04 gap-closure where it was resolved)
- [x] 02-04-PLAN.md — Gap closure: bump `JUDGE_MAX_TOKENS=8192` in `score._build_judge` (TDD red→green) + re-score Tier 4 against existing capture + double smoke gate (Tier 4 PASS gap + Tier 5 PASS regression check) (TIER-03; gap_closure; depends on score.py only — no graph touch, no re-capture) — **COMPLETE 2026-05-05** (~12 min; commits cdbc376 RED, 7fc6d66 GREEN, plan-metadata for SUMMARY; Tier 4 smoke PASS 5/5 faithfulness=1.0, Tier 5 smoke PASS regression check intact)

### Phase 3: NaN Reason Instrumentation
**Goal**: Every NaN in per-row RAGAS metrics output carries a structured `nan_reason` so reviewers can distinguish "tier failed to retrieve" from "judge failed to decompose claims" from "Gemini returned malformed JSON".
**Depends on**: Nothing (parallel with Phases 1, 2, 4, 5, 6)
**Requirements**: HARN-05
**Success Criteria** (what must be TRUE):
  1. User can read any `metrics/tier-{N}-*.json` produced after this phase and see a per-row `nan_reason` field with one of `empty_contexts`, `empty_statements`, `json_parse_failure`, or `null` (no NaN)
  2. User can confirm `evaluation/harness/score.py`'s `_short_circuit_nan()` already-present pattern is extended to tag the new reasons rather than swallowing them as a single `nan_reason="nan"`
  3. User can confirm the rollup in `comparison.md` aggregates NaN counts by reason (e.g., `tier-5: 2 empty_contexts, 1 empty_statements`) rather than a single opaque `n_NaN`
**Plans**: 3 plans (3 waves; W1 TDD-only pure units, W2 wiring + integration + compare regression, W3 live smoke backstop with checkpoint:human-verify)
- [x] 03-01-PLAN.md — TDD red→green for NaNReasonTracer + _classify_post_evaluate_nan pure helpers (HARN-05; type:tdd; 2 tasks) — **COMPLETE 2026-05-05** (~5 min wall; commits e97e864 RED, bc80825 GREEN; +109 LOC score.py / +253 LOC test_eval_score.py; test count 13→28; all 7 verification gates PASS; HARN-05 closure pending Plan 03-02 wiring)
- [x] 03-02-PLAN.md — Wire tracer into score_query_log + integration tests with stub LLMs + compare.py rollup regression test proving zero compare.py change needed (HARN-05; depends_on 03-01) — **COMPLETE 2026-05-05** (~12 min wall; single atomic commit fe52528; +30/-6 score.py wiring; +214 test_eval_score.py with 4 stub-LLM integration tests; +63 test_eval_compare.py with 1 rollup regression test; compare.py byte-identical)
- [x] 03-03-PLAN.md — Live smoke backstop asserting unknown_nan==0 against existing Tier 5 capture; non-autonomous human-verify checkpoint (HARN-05; depends_on 03-02) — **COMPLETE 2026-05-05** (~10 min wall + 78.78s live test; commit 512ad54 test + plan-metadata docs commit; live smoke verdict PASS: n_total=5, n_unknown_nan=0, n_scored_post_short_circuit=5; ~$0.014 cost vs $0.05 cost guard; HARN-05 closed end-to-end at unit + integration + live levels)

### Phase 4: Freeze Tool
**Goal**: A single command produces an immutable, copy-pasteable frozen markdown artifact under `evaluation/results/frozen/` with a sidecar manifest that records exactly which capture / score / compare files fed it and at what git SHA.
**Depends on**: Nothing (parallel with Phases 1, 2, 3, 5, 6)
**Requirements**: HARN-03, HARN-04
**Success Criteria** (what must be TRUE):
  1. User can run `python -m evaluation.harness.freeze --version 1.0` (or equivalent) and see `evaluation/results/frozen/eval-numbers-v1.0.md` written as a copy of the current `comparison.md`
  2. User can re-run the freeze command and see it refuse with a clear error message ("eval-numbers-v1.0.md already frozen — bump version or pass --force") — frozen docs do not silently overwrite
  3. User can read `evaluation/results/frozen/eval-numbers-v1.0.manifest.json` and find: git SHA, freeze timestamp, per-tier capture/score/metrics source paths with mtimes, judge model, generation model per tier, and pinned versions of `lightrag-hku`, `raganything`, `openai-agents`, `ragas`
  4. User can confirm `freeze.py` is approximately 60 LOC of pure-Python (no new external deps) and lives at `evaluation/harness/freeze.py`
**Plans**: TBD

### Phase 5: Pipeline Driver
**Goal**: Capture → score → compare → freeze runs as one command with a single git SHA and ISO timestamp captured at start, and re-running a single tier does not invalidate the captured runs of the other four.
**Depends on**: Phase 4 (calls `freeze()` as the final stage)
**Requirements**: HARN-01, HARN-02
**Success Criteria** (what must be TRUE):
  1. User can run `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` and see capture → score → compare → freeze execute in sequence, with one git SHA recorded at start and propagated to every per-tier output JSON
  2. User can run `python -m evaluation.harness.pipeline --tiers 4 --tier-4-from-cache <path>` and see only Tier 4's capture and metrics regenerate; tiers 1, 2, 3, 5 retain their previous run data and are picked up by `_latest()` mtime resolution in `compare.py`
  3. User can confirm `pipeline.py` calls existing `run.amain()` / `score.amain()` / `compare._run()` as in-process function calls (not subprocesses) so exit codes propagate and asyncio loops are reused
  4. User can confirm cost-surprise prompts fire once per phase (not five times) when running a full sweep
**Plans**: TBD

### Phase 6: Embedder Provenance Capture
**Goal**: Every per-tier capture JSON records the embedding model used by that tier so the frozen doc's embedder-confound table is generated from data, not authored from memory.
**Depends on**: Nothing (parallel with Phases 1, 2, 3, 4, 5)
**Requirements**: CAP-03
**Success Criteria** (what must be TRUE):
  1. User can read any `queries/tier-{N}-*.json` produced after this phase and find a top-level `embedder` field (e.g., `text-embedding-3-small`, `text-embedding-004`, `openai-hosted-managed`) plus its source (`openrouter`, `google-managed`, `openai-direct`, `openai-hosted`)
  2. User can confirm Tier 1, 3, 4 record `text-embedding-3-small` (matching pinned config); Tier 2 records Google's managed embedder; Tier 5 records OpenAI's hosted vector-store embedder with an explicit `managed=true` flag
  3. User can run `compare.py` and see the embedder fields aggregated into the rollup so the frozen doc can render an embedder-by-tier table without manual entry
**Plans**: TBD

### Phase 7: Full 5-Tier Rerun
**Goal**: All 5 tiers × 30 questions captured and scored on a single date with a single git SHA, producing fresh per-tier `queries/`, `costs/`, and `metrics/` files that feed the rollup.
**Depends on**: Phases 1, 2, 3, 5, 6 (Tier 5 fix smoke-tested, Tier 4 graph clean and smoke-tested, NaN reason field present, pipeline driver exists, embedder field present)
**Requirements**: CAP-01
**Success Criteria** (what must be TRUE):
  1. User can run `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --yes` and see fresh `queries/tier-{N}-{TS}.json`, `costs/tier-{N}-eval-{TS}.json`, `costs/ragas-judge-tier-{N}-{TS}.json`, `metrics/tier-{N}-{TS}.json` for every tier with timestamps within the same date
  2. User can grep all 10 output JSONs (5 capture + 5 score) and find the same git SHA recorded in each — no SHA drift across tiers
  3. User can confirm Tier 4 NaN count is <5/30 (down from 30/30 baseline, gated on Phase 2 fix sticking)
  4. User can confirm Tier 5 NaN count is <5/30 (down from 30/30 baseline, gated on Phase 1 fix sticking)
  5. User can confirm total spend stayed within the documented $1–3 budget per full sweep, recorded in capture+judge cost ledgers
**Plans**: TBD

### Phase 8: Multi-Judge Spot-Check
**Goal**: 5 questions × 3 tiers (15 cells) re-scored with a non-Gemini judge (Claude Haiku or GPT-4.1-mini) so the family-bias disclosure in the frozen doc cites a measured delta, not just published-paper magnitudes.
**Depends on**: Phase 7 (needs primary numbers to delta against)
**Requirements**: CAP-02
**Success Criteria** (what must be TRUE):
  1. User can run a re-score command with `--judge <non-gemini-slug>` against a 5-question × 3-tier subset and see a structured JSON written under `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`
  2. User can read that JSON and find, per cell: primary-judge score (Gemini), secondary-judge score, and delta — across faithfulness, answer_relevancy, context_precision
  3. User can confirm the spot-check spend stayed within $0.10–0.30 (cost recorded in `costs/multi-judge-spotcheck-{TS}.json`)
  4. User can confirm the secondary judge's model + version is recorded in the JSON for provenance (no opaque "Claude")
**Plans**: TBD

### Phase 9: Frozen Handoff Doc
**Goal**: Produce `evaluation/results/frozen/eval-numbers-v1.0.md` — the single immutable artifact this repo ships to the external blog repo, containing every number, every disclaimer, and every provenance line a hostile re-reader needs.
**Depends on**: Phase 4 (freeze tool exists), Phase 7 (real numbers exist), Phase 8 (multi-judge delta exists), Phase 3 (NaN reasons populate the breakdown)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05
**Success Criteria** (what must be TRUE):
  1. User can open `evaluation/results/frozen/eval-numbers-v1.0.md` and find a 5-row tier rollup table with columns: faithfulness, answer_relevancy, context_precision, mean_latency_s, total_cost_usd, cost_per_query_usd, n, n_nan
  2. User can find a 15-row per-question-class table (5 tiers × 3 classes: single-hop / multi-hop / multimodal) matching `comparison.md` format
  3. User can find a per-tier provenance block with capture timestamp, generation model + version, git SHA, judge model + version, and judge embedder — sufficient for any reader to reproduce the run
  4. User can find honest disclaimers covering: small-N (n=30 magnitudes-only), self-grading bias with measured multi-judge delta from Phase 8, multimodal-tier-1-3-text-only de-emphasis, embedder-confound table from Phase 6, and any residual NaN reasons from Phase 3
  5. User can copy-paste the entire frozen doc into the blog repo's `rag-architecture-patterns.mdx` and have all numbers and disclosures travel together — no external lookups required
  6. User can confirm `eval-numbers-v1.0.manifest.json` sits beside the markdown and the freeze command refuses to clobber if re-run
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Phases 1, 2, 3, 4, 6 are parallel-friendly (different files, no shared state); Phase 5 depends on Phase 4; Phase 7 depends on 1, 2, 3, 5, 6; Phase 8 depends on 7; Phase 9 depends on 4, 7, 8.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Tier 5 Adapter Fix | 3/3 | ✓ Verified | 2026-05-04 |
| 2. Tier 4 Graphml Regeneration | 4/4 | ✓ Verified | 2026-05-05 |
| 3. NaN Reason Instrumentation | 3/3 | ✓ Verified | 2026-05-05 |
| 4. Freeze Tool | 0/TBD | Not started | - |
| 5. Pipeline Driver | 0/TBD | Not started | - |
| 6. Embedder Provenance Capture | 0/TBD | Not started | - |
| 7. Full 5-Tier Rerun | 0/TBD | Not started | - |
| 8. Multi-Judge Spot-Check | 0/TBD | Not started | - |
| 9. Frozen Handoff Doc | 0/TBD | Not started | - |
