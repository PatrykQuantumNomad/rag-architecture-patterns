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
- [x] **Phase 4: Freeze Tool** - `evaluation/harness/freeze.py` writes immutable `frozen/eval-numbers-vX.Y.md` + sidecar manifest with git SHA, capture timestamps, library versions ✓ 2026-05-05 [1/1 plan delivered; HARN-03 + HARN-04 closed at unit + CLI levels; freeze.py at exactly 95 LOC honoring max_lines:95 hard cap; sidecar manifest carries git_sha + git_dirty + ISO 8601 Z frozen_at + per-tier capture/cost/metrics relative paths + mtimes + judge {model, embedder, max_tokens=8192} + library_versions for the 4 critical libs (lightrag-hku 1.4.15 / raganything 1.2.10 / openai-agents 0.14.6 / ragas 0.4.3) + 2 bonus (litellm 1.83.0, chromadb 1.5.8); refuse-to-clobber default with prescribed wording; --force overwrites both md AND manifest; RuntimeError + exit 2 BEFORE any file write if any of the 4 critical libs missing; Phase 5 forward-contract signature locked at `freeze(version, force, results_dir, source) -> Path`; zero modifications to run.py / compare.py / score.py]
- [x] **Phase 5: Pipeline Driver** - `evaluation/harness/pipeline.py` runs capture → score → compare → freeze in one command, with single-tier rerun support ✓ 2026-05-06 [2/2 plans delivered; HARN-01 closed end-to-end at unit + integration + live levels (Plan 05-01 + Plan 05-02); HARN-02 closed at unit + integration levels (live-level coverage exceeds cost ceiling for a single test); Plan 05-02 live verdict PASS at total_usd=$0.007010 / 158.02s wall on Tier 5 × 5q against real OpenRouter API]
- [x] **Phase 6: Embedder Provenance Capture** - Per-tier embedder model name recorded in capture JSON so the embedder-confound disclosure is data-backed ✓ 2026-05-06 [1/1 plan delivered; CAP-03 closed at unit + integration levels via 6 atomic RED→GREEN pairs (12 commits a6c3155..5d13879); QueryLog gains Optional[embedder, embedder_source]; 5 tier modules carry per-tier EMBEDDER_SOURCE constants; run.py + tier-4 eval_capture.py thread fields through both capture entry points (D-CAPTURE-ENTRYPOINTS); compare.py emits per-tier embedder line + dedicated "Embedder by tier" table; freeze.py manifest carries per_tier embedder fields; D-ROADMAP-OVERRIDE locked (Tier 5 records SAME embedder as Tier 1: openai/text-embedding-3-small via openrouter — verified against tier-5-agentic/tools.py:47-50,90-101); forward-contract guard intact (0 bytes diff for score.py / pipeline.py / smoke_gate.py); all 10 LOC budgets honored zero compression iterations; 116→126 offline tests pass]
- [x] **Phase 7: Full 5-Tier Rerun** - Capture all 5 tiers × 30 questions on one date with one git SHA, NaN counts down to expected residuals ✓ 2026-05-07 [3/3 plans delivered; CAP-01 closed at sweep_sha=75f6f1b on date 2026-05-07; T4 NaN 2/30, T5 NaN 0/30, total sweep $0.439 (vs $3 ceiling); Tier 4 graph rebuilt to 28597 nodes / 80419 edges from 79 papers / $24.79 host run; forward-contract guard intact end-to-end (0 bytes diff vs pre-Phase-7 baseline 03f9ce1 across 6 harness modules); 5/5 ROADMAP success criteria verified (SC-5 human-confirmed via dashboard 2026-05-07)]
- [x] **Phase 8: Multi-Judge Spot-Check** - Re-score 5 questions × 3 tiers with a non-Gemini judge, capture delta in structured JSON ✓ 2026-05-07 [2/2 plans delivered; CAP-02 closed end-to-end at unit + integration + live levels; gsd-verifier verdict passed 4/4 SCs; live test verdict CLEAN PASS at total_usd=$0.12225 (40.75% of $0.30 SOFT envelope, 24.45% of $0.50 HARD ceiling); 15 cells produced (5 IDs × 3 tiers), 0/15 secondary nan_reasons; dual-SHA provenance verified at LIVE level (source_capture_git_sha=75f6f1b ≠ HEAD 3f37e4b); secondary judge anthropic/claude-haiku-4.5 max_tokens=8192; per-tier mean Δ-faithfulness preview for Phase 9 frozen-doc family-bias disclosure: tier-1 -0.035, tier-4 +0.010, tier-5 -0.164, overall -0.063; forward-contract guard intact end-to-end (0 bytes diff across 9 RAW-LOCKed harness modules — second pass in Phase 8); offline regression 116→128 PASS]
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
**Plans**: 1 plan (1 wave)
- [x] 04-01-PLAN.md — TDD red→green for evaluation/harness/freeze.py (HARN-03 + HARN-04; pure-Python CLI + in-process freeze() function for Phase 5; 10 unit tests + CLI quality gate; ~74 LOC ±15 tolerance, hard cap 95) — **COMPLETE 2026-05-05** (~18 min wall; commits 9588056 RED, 7881e87 GREEN; 95 LOC freeze.py exactly at hard cap; 10/10 tests PASS; full offline suite 88 passed excluding pre-existing tier_2.py adapter fail; 5/5 CLI quality gate sub-checks PASS; live artifacts deleted post-verification; Phase 5 forward-contract locked)

### Phase 5: Pipeline Driver
**Goal**: Capture → score → compare → freeze runs as one command with a single git SHA and ISO timestamp captured at start, and re-running a single tier does not invalidate the captured runs of the other four.
**Depends on**: Phase 4 (calls `freeze()` as the final stage)
**Requirements**: HARN-01, HARN-02
**Success Criteria** (what must be TRUE):
  1. User can run `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` and see capture → score → compare → freeze execute in sequence, with one git SHA recorded at start and propagated to every per-tier output JSON
  2. User can run `python -m evaluation.harness.pipeline --tiers 4 --tier-4-from-cache <path>` and see only Tier 4's capture and metrics regenerate; tiers 1, 2, 3, 5 retain their previous run data and are picked up by `_latest()` mtime resolution in `compare.py`
  3. User can confirm `pipeline.py` calls existing `run.amain()` / `score.amain()` / `compare._run()` as in-process function calls (not subprocesses) so exit codes propagate and asyncio loops are reused
  4. User can confirm cost-surprise prompts fire once per phase (not five times) when running a full sweep
**Plans**: 2 plans (2 waves; 05-02 depends_on 05-01 — wave 2 is the live smoke backstop)
- [x] 05-01-PLAN.md — TDD red→green: pipeline.py (≤220 raw LOC) + ~6-LOC additive run.py kwarg plumb-through + 12 offline tests (10 unit + 2 integration). Closes HARN-01 + HARN-02 at unit + integration levels. **COMPLETE 2026-05-06** (~12 min wall; commits 3259ab1 RED + aa1640c GREEN; 189/220 LOC pipeline.py with 14% buffer; 15/15 offline tests PASS; run.py +10 raw insertions / -3 deletions; byte-identical guards intact for score.py / compare.py / freeze.py; regression baseline 88 → 103 PASS; 6/6 CLI quality gate sub-checks PASS; two Rule-1/3 micro-deviations rolled into the GREEN commit — see SUMMARY)
- [x] 05-02-PLAN.md — Live smoke backstop: 1 @pytest.mark.live test against real OpenRouter API (Tier 5 × 5q, ≤$0.05). Closes HARN-01 at live level. Non-autonomous (cost-acknowledgement checkpoint). **COMPLETE 2026-05-06** (~25 min wall + dual live invocations 222s+158s; commit 117d595 test + plan-finalization docs commit; live verdict PASS: rc=0, total_usd=$0.007010, queries.git_sha=fd287f0 (matches HEAD), 1/1/1/1 tier-5 queries+metrics+tier-5-eval+ragas-judge files on disk, comparison.md regenerated with tier-5 row; offline regression 15/15 PASS unchanged; pipeline.py + run.py + score.py + compare.py + freeze.py byte-identical to Plan 05-01 — 0 bytes diff total across 5 files; 2 Rule-1 deviations rolled into the test commit (cost-JSON schema-adapt + monkeypatch shared.cost_tracker.DEFAULT_COSTS_DIR for tmp_path isolation — pre-existing harness mis-feature in run.py:273 + score.py:518 calling tracker.persist() with no dest_dir; v1.1 hardening item filed); total live spend ~$0.014 across both invocations, well under $0.05 single-run ceiling)

### Phase 6: Embedder Provenance Capture
**Goal**: Every per-tier capture JSON records the embedding model used by that tier so the frozen doc's embedder-confound table is generated from data, not authored from memory.
**Depends on**: Nothing (parallel with Phases 1, 2, 3, 4, 5)
**Requirements**: CAP-03
**Success Criteria** (what must be TRUE):
  1. User can read any `queries/tier-{N}-*.json` produced after this phase and find a top-level `embedder` field (e.g., `text-embedding-3-small`, `text-embedding-004`, `openai-hosted-managed`) plus its source (`openrouter`, `google-managed`, `openai-direct`, `openai-hosted`)
  2. User can confirm Tier 1, 3, 4 record `text-embedding-3-small` (matching pinned config); Tier 2 records Google's managed embedder; Tier 5 records OpenAI's hosted vector-store embedder with an explicit `managed=true` flag
  3. User can run `compare.py` and see the embedder fields aggregated into the rollup so the frozen doc can render an embedder-by-tier table without manual entry

> **D-ROADMAP-OVERRIDE (Plan 06-01):** Success Criterion 2 above is OVERRIDDEN by 06-RESEARCH.md verification. Tier 5 actually reuses Tier 1's local ChromaDB and embeds via OpenRouter `openai/text-embedding-3-small` — IDENTICAL to Tier 1, NOT "OpenAI's hosted vector-store embedder". The `managed=true` flag is rejected; "managed" is derived as `embedder_source == "google-managed"` (Tier 2 only). See 06-01-PLAN.md SUMMARY block.

**Plans**: 1 plan (1 wave; pure-offline TDD red→green; ~25-40 net LOC across 10 files)
- [x] 06-01-PLAN.md — TDD red→green: QueryLog Optional[embedder, embedder_source] + per-tier EMBEDDER_SOURCE constants (5 tier modules) + run.py / eval_capture.py / compare.py / freeze.py field plumb-through + 7-9 new offline tests + embedder-by-tier table in comparison.md. Closes CAP-03. Forward-contract guards: score.py / pipeline.py / smoke_gate.py byte-identical post-merge. **COMPLETE 2026-05-06** (~18 min wall; 12 atomic commits a6c3155..5d13879 / RED+GREEN pairs across 6 tasks; 116→126 offline tests pass; D-ROADMAP-OVERRIDE locked in source via tier-5 inline rejection comment + override callout in 06-RESEARCH.md and ROADMAP.md; all 10 LOC budgets honored zero compression iterations — third consecutive plan applying Plan 04-01's "validate sketch via raw wc -l" lesson)

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
**Plans**: 3 plans (3 waves; 07-01 pre-flight smoke gates, 07-02 Tier 4 host-machine full-corpus rebuild, 07-03 live full-sweep + post-sweep verification — all three are non-autonomous with cost-ack or host-execution checkpoints)
- [ ] 07-01-PLAN.md — Pre-flight: 6 environment/dependency gates + recreate `tier-2-managed/.store_id` + 5q smoke per tier × 5 tiers + Tier 4 passthrough SHA verification (resolves 07-RESEARCH.md Open Q1) (CAP-01 enabler; ~$0.50–1.00, ceiling $1.50)
- [ ] 07-02-PLAN.md — Tier 4 full-corpus rebuild on host (79 papers via `ingest_from_mineru.py --reset --yes`) + new graph-stats provenance manifest committed beside 2026-05-05 smoke manifest + post-rebuild 5q smoke gate (CAP-01 enabler; ~$15–35, ceiling $50, 15–25h wall, MUST run on host not sandbox)
- [ ] 07-03-PLAN.md — Live full-sweep: Tier 4 30q live capture via `eval_capture.py` → `pipeline.py --tiers 1,2,3,4,5 --tier-4-from-cache <path> --yes` → post-sweep verifier (5 ROADMAP success criteria) → human-verify checkpoint (CAP-01 closure; ~$0.84, ceiling $3.00 per ROADMAP SC-5)

### Phase 8: Multi-Judge Spot-Check
**Goal**: 5 questions × 3 tiers (15 cells) re-scored with a non-Gemini judge (Claude Haiku or GPT-4.1-mini) so the family-bias disclosure in the frozen doc cites a measured delta, not just published-paper magnitudes.
**Depends on**: Phase 7 (needs primary numbers to delta against)
**Requirements**: CAP-02
**Success Criteria** (what must be TRUE):
  1. User can run a re-score command with `--judge <non-gemini-slug>` against a 5-question × 3-tier subset and see a structured JSON written under `evaluation/results/metrics/multi-judge-spotcheck-{TS}.json`
  2. User can read that JSON and find, per cell: primary-judge score (Gemini), secondary-judge score, and delta — across faithfulness, answer_relevancy, context_precision
  3. User can confirm the spot-check spend stayed within $0.10–0.30 (cost recorded in `costs/multi-judge-spotcheck-{TS}.json`)
  4. User can confirm the secondary judge's model + version is recorded in the JSON for provenance (no opaque "Claude")
**Plans**: 2 plans (2 waves; 08-01 pure-offline TDD red→green, 08-02 live smoke backstop with cost-ack checkpoint — same Phase 1/2/3/5 offline-first → live-backstop pattern)
- [x] 08-01-PLAN.md — TDD red→green for `evaluation/harness/multi_judge_spotcheck.py` (≤230 raw LOC; landed at 227) + 12 offline tests (1 Task 0 pre-flight + 11 unit/integration) covering schema / signed-delta / source-SHA pinning via `_read_source_sha` (BLOCKER #3 fix — reads `src_log.git_sha`, NEVER `_git_sha()`) / provenance / cost-ledger dest_dir isolation / 15-cell shape / ID consistency / fallback estimator / forward-contract guard. Closes CAP-02 at unit + integration levels. Forward-contract guard intact (0 bytes diff across `pipeline.py / run.py / score.py / compare.py / freeze.py / smoke_gate.py / records.py / shared/cost_tracker.py / shared/pricing.py`). Spend $0.00. — **COMPLETE 2026-05-07** (~30 min wall; commits 54e62c0 Task 0 pre-flight + 1e9db22 RED + 3baa8a8 GREEN; offline regression 116→128 PASS; LOC budget honored at 227/230 after 5 compression iterations from 365 raw initial draft).
- [x] 08-02-PLAN.md — Live smoke backstop: 1 `@pytest.mark.live` test driving `multi_judge_spotcheck.amain` against the Phase 7 sweep_sha=`75f6f1b` capture + real OpenRouter API (Claude Haiku 4.5 secondary judge). Closes CAP-02 at live level. Non-autonomous (cost-ack checkpoint). Spend ≤ $0.50 HARD ceiling, ~$0.12 projected, $0.10–0.30 ROADMAP envelope. — **COMPLETE 2026-05-07** (~10 min orchestrator wall + 6m1s live test wall; commits 3f37e4b test + 676f07d docs; verdict CLEAN PASS at total_usd=$0.12225; 0/15 secondary nan_reasons; multi_judge_spotcheck.py UNTOUCHED — byte-identical to Plan 08-01 GREEN 3baa8a8; forward-contract guard intact for the SECOND TIME in Phase 8; offline regression 128 PASS preserved exactly)

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
**Plans**: 2 plans

Plans:
**Wave 1**
- [x] 09-01-PLAN.md — Inline Phase 8 delta + finalize comparison.md disclaimers

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 09-02-PLAN.md — Freeze v1.0 markdown + manifest + immutability checks

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Phases 1, 2, 3, 4, 6 are parallel-friendly (different files, no shared state); Phase 5 depends on Phase 4; Phase 7 depends on 1, 2, 3, 5, 6; Phase 8 depends on 7; Phase 9 depends on 4, 7, 8.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Tier 5 Adapter Fix | 3/3 | ✓ Verified | 2026-05-04 |
| 2. Tier 4 Graphml Regeneration | 4/4 | ✓ Verified | 2026-05-05 |
| 3. NaN Reason Instrumentation | 3/3 | ✓ Verified | 2026-05-05 |
| 4. Freeze Tool | 1/1 | ✓ Complete | 2026-05-05 |
| 5. Pipeline Driver | 2/2 | ✓ Complete | 2026-05-06 |
| 6. Embedder Provenance Capture | 1/1 | ✓ Verified | 2026-05-06 |
| 7. Full 5-Tier Rerun | 3/3 | ✓ Verified | 2026-05-07 |
| 8. Multi-Judge Spot-Check | 2/2 | ✓ Verified | 2026-05-07 |
| 9. Frozen Handoff Doc | 1/2 | In Progress|  |
