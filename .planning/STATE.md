---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
stopped_at: "Phase 2 Plan 02-04 complete: gap closure landed for the Plan 02-03 FAIL. Bumped JUDGE_MAX_TOKENS=8192 in score._build_judge (TDD red→green; commits cdbc376 + 7fc6d66), re-scored Tier 4 against existing capture (no re-capture, no graph touch), and confirmed smoke_gate verdict=PASS for BOTH Tier 4 (gap closure target — non_nan_faithfulness_count went 1→5) AND Tier 5 (Phase 1 regression check — byte-identical PASS). Phase 2 ship gate now satisfied at smoke-gate level; ROADMAP success criterion #4 (<5/5 empty_contexts NaNs on smoke) cleared with 0/5 NaNs of any reason. Phase 2 fully delivered (4/4 plans complete)."
last_updated: "2026-05-05T15:17:00Z"
last_activity: "2026-05-05 — Phase 02 Plan 02-04 executed end-to-end (~12 min): Task 1 TDD added 2 unit tests (test_build_judge_passes_max_tokens, test_build_judge_max_tokens_is_named_constant) — RED commit cdbc376 verified failing then GREEN commit 7fc6d66 added JUDGE_MAX_TOKENS=8192 module-level constant + threaded max_tokens=JUDGE_MAX_TOKENS into llm_factory call (13/13 tests pass); Task 2 re-scored Tier 4 via score --tiers 4 --yes (5/5 faithfulness=1.0, was 4/5 NaN), Tier 4 smoke gate PASS, Tier 5 smoke gate PASS (Phase 1 regression check intact). Rule-3 deviation: project .gitignore excludes evaluation/results/metrics/ and evaluation/results/costs/*.json so the plan's Step E artifact commit was rolled into the SUMMARY commit with verdict JSON captured verbatim for provenance equivalence."
progress:
  total_phases: 9
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Produce reproducible, honest numbers for the blog — every claim backed by a captured run with full provenance.
**Current focus:** Phase 2 (Tier 4 Graphml Regeneration) — all 4 plans complete; Plan 02-04 gap closure landed both Tier 4 (was FAIL) and Tier 5 (Phase 1 regression check) at verdict=PASS; phase ship gate cleared.

## Current Position

Phase: 2 of 9 (Tier 4 Graphml Regeneration) — COMPLETE (all 4 plans delivered, smoke gate PASS, ship gate cleared)
Plan: 4 of 4 complete in Phase 2; Plan 02-04 closed the Plan 02-03 smoke FAIL via TDD red→green on score.py max_tokens wiring
Status: Plan 02-01 landed smoke-only graphml (2886 nodes / 7056 edges, 3 papers); Plan 02-02 top-up MineRU cache to 79 papers (approved, non-blocking); Plan 02-03 deliverables shipped (--smoke-question-ids flag + helper, live smoke test, context-probe API-drift fix) + smoke_gate verdict=FAIL surfaced; Plan 02-04 closed the gap with JUDGE_MAX_TOKENS=8192 in score._build_judge — Tier 4 PASS (5/5 faithfulness=1.0) AND Tier 5 PASS (Phase 1 provenance intact).
Last activity: 2026-05-05 — Phase 02 Plan 02-04 gap closure: TDD red→green for max_tokens wiring + Tier 4 re-score against existing capture + double smoke gate PASS verdict captured verbatim in 02-04-SUMMARY.md.

Progress: [████░░░░░░] 22% phase-level (1/9 phases fully complete by ROADMAP framing; Phase 2 cleared smoke ship gate but not yet flipped to [x] in ROADMAP — orchestrator's call) | 100% plan-level for active phases (7/7 plans across Phases 1+2)
Phase 2 ship gate is now CLEARED at the smoke-gate level. Both Tier 4 and Tier 5 smoke verdicts are PASS under the same score._build_judge config, so the score.py change is cross-tier-validated.

## Performance Metrics

**Velocity:**

- Total plans completed: 6
- Average duration: ~22 min
- Total execution time: ~2.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-tier-5-adapter-fix | 3 | ~22 min | ~7 min |
| 02-tier-4-graphml-regeneration | 4 | ~142 min (~50 min Plan 02-01 + ~50 min Plan 02-02 host MineRU + ~30 min Plan 02-03 + ~12 min Plan 02-04) | ~36 min |

**Recent Trend:**

- Last 7 plans: 01-01 (8 min), 01-03 (4 min), 01-02 (~10 min), 02-01 (~50 min), 02-02 (~50 min host MineRU), 02-03 (~30 min Task 1 + Task 2 with 3 capture cycles), 02-04 (~12 min — score-only TDD + re-score + double smoke gate)
- Trend: live-ingest plans are 5-10× pure-code plans; smoke verification plans (Plan 02-03) sit in between; gap-closure score-only plans (Plan 02-04) are 2-3× pure-code plans (one TDD cycle + one live re-score + two cheap gate calls).

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.0 scope: fix-and-ship the existing 5 tiers; no new architectures, no new questions
- Multi-judge spot-check (CAP-02) is in v1.0 scope as *measurement* (5q × 3 tiers); full mitigation deferred to v1.1
- Single eval-date for all 5 tiers; budget ~$1-3 per full rerun, 2-3 reruns total
- Tier 4 MineRU ingest must run outside the sandbox (Phase 139 evidence)
- Tier 5 adapter reads `item.output` not `item.raw_item` (Pitfall 1 of 132-RESEARCH HARD invariant) — `raw_item` is a stringified Responses-API payload
- Tier 5 contexts deduped first-occurrence-wins on `(paper_id, page)` for chunks and `(paper_id)` for metadata abstracts
- Tier 5 tracing toggle gated on `RAG_DEBUG_TIER5_TRACING` env var (default disabled — byte-identical behavior; opt-in for OpenInference fallback)
- `[debug-tier5]` extra is opt-in only (CLEANUP-02 of REQUIREMENTS.md v1.1) — default `uv sync` does NOT install OpenInference / Phoenix / OpenTelemetry; `uv sync --extra debug-tier5` activates the diagnostic stack on demand (D-09)
- Fallback log persistence (`FallbackLog`) reuses `model_dump_json(indent=2)` and `_git_sha` / `_ts` / `_ts_for_filename` helpers from `evaluation.harness.run` — single source of truth for SHA + ISO 8601 Z conventions
- Fallback runbook encodes D-06 instrument-first ordering verbatim (Phoenix spans first, STACK.md mutations second); `bump_openai_agents` is LAST RESORT requiring user authorization via `checkpoint:decision`
- Live smoke gate PASS verdict on 2026-05-04 — 5/5 measurable populated, ratio 1.00, all RAGAS metrics non-NaN; Plan 01-03 fallback runbook remains dormant scaffolding for Phase 7 to read
- Tier 5 retrieved_contexts populated as expected — every entry in all 5 records starts with `[paper_id=`, n_ctx range 4-6, no Python repr leakage (Pitfall 1 of 132-RESEARCH guard intact end-to-end against live API)
- Capture cost $0.0048 + judge cost $0.00 on the 5-question smoke (well under the $0.05/run Pitfall-7 cost guard); smoke is cheap to re-run for any future regression check
- **Phase 2 Plan 02-01 Task 2b ran smoke-only (3 of 75 papers: 2005.11401, 2004.04906, 2002.08909) per orchestrator Option B; full 75-paper ingest deferred to Phase 7 pre-rerun. Reason: measured paper-1 wall time of ~21 min projected ~15–25h / $1.50–3 vs plan budget of 10–30 min / $0.50–1.00. Phase 2 must-haves are met for the smoke set; Phase 7 is the architecturally correct place for the remaining 72 papers.**
- Phase 2 graphml ground truth captured at 2026-05-05T11:14:40Z: 2886 nodes / 7056 edges from 3 smoke papers, raganything==1.2.10, lightrag-hku==1.4.15, mineru==3.1.4 — this manifest is the version-of-record for Phase 9's frozen doc
- RAG-Anything 1.2.10 + lightrag-hku 1.4.15 require two integration fixes for Tier 4: (a) bypass the mineru-CLI parser-version probe at construction time when ingesting cached JSON via `insert_content_list`, and (b) forward `OPENROUTER_API_KEY` into `os.environ` at script entry because lightrag's openai_complete_if_cache reads it lazily inside async closures, not from the SecretStr-wrapped settings object — both fixes live in `tier-4-multimodal/scripts/ingest_from_mineru.py` (committed 5bc3f24)
- Plan 02-02 host MineRU top-up: 4/4 papers parsed successfully on host (approved branch). Wall time ~49 min total. The 4 new papers are in the MineRU cache but NOT in the smoke-only graphml; Phase 7 will re-rebuild over the full 79-paper cache.
- Plan 02-03 ships --smoke-question-ids on tier-4-multimodal/scripts/eval_capture.py with single-source-of-truth import of DEFAULT_SMOKE_IDS from evaluation.harness.run (Pitfall 5 of 02-RESEARCH.md). 12 unit tests cover the new pure helper _filter_qa(); test_eval_smoke_tier4_full_pipeline live test added (collected under -m live alongside tier-1 + tier-5).
- Plan 02-03 Rule-1 auto-fix: eval_capture.py's context probe used `rag.aquery(... param=...)` which fails under RAG-Anything 1.2.10 (signature does not accept `param=`); fix calls `rag.lightrag.aquery(... param=QueryParam(...))` directly, mirroring what RAGAnything itself does internally. Without the fix, retrieved_contexts was empty for all queries, masking real signal.
- Plan 02-03 smoke verdict: FAIL on faithfulness NaN (4/5 rows). Root cause is RAGAS faithfulness metric's _create_statements step hitting Gemini's 1024-token output cap when extracting atomic claims from long Tier 4 hybrid-mode answers (2011-2575 chars). NOT a Plan 02-01 regression: n_populated=5/5, ratio=1.0, context_precision 5/5 non-NaN, answer_relevancy 5/5 non-NaN, no Python repr leak. Recommended gap-closure: add max_tokens=8192 to score._build_judge's litellm.completion config and re-run score only.
- Plan 02-04 closed the Plan 02-03 FAIL via the recommended one-line fix: added module-level constant JUDGE_MAX_TOKENS=8192 beside JUDGE_LLM_SLUG_DEFAULT in evaluation/harness/score.py (line 70) and threaded max_tokens=JUDGE_MAX_TOKENS into the llm_factory(...) call inside _build_judge (line 131). Two new unit tests in evaluation/tests/test_eval_score.py monkeypatch ragas.llms.llm_factory + ragas.embeddings.{base.,}embedding_factory and assert the wiring offline (TDD: RED commit cdbc376 → GREEN commit 7fc6d66, all 13 tests pass). Re-scored Tier 4 against existing capture: 5/5 faithfulness=1.0 (was 4/5 NaN). Smoke gate verdict=PASS for Tier 4 (gap closure target) AND Tier 5 (Phase 1 regression check — byte-identical to 2026-05-04 PASS).
- Plan 02-04 Rule-3 deviation: project .gitignore (set at Phase 131 init) excludes evaluation/results/metrics/ and evaluation/results/costs/*.json as 'regenerable runtime intermediates'. The plan's Step E artifact commit was incompatible with this convention (which Plan 02-03 already followed); adapted to roll Step E into the SUMMARY commit with both SmokeGateResult JSON blocks captured verbatim in 02-04-SUMMARY.md for provenance equivalence. No Rule 4 architectural change made.

### Pending Todos

None yet.

### Blockers/Concerns

- Tier 5 root cause is high-confidence (5-line adapter bug at `tier_5.py:125`) but Phase 1 must include a fallback path: if smoke test still shows empty contexts after the fix, follow STACK.md decision tree (simplify schema → switch model slug → bump openai-agents) before declaring done
- OpenRouter passthrough may obscure the exact Gemini model version (`gemini-2.5-flash-001` vs bare `gemini-2.5-flash`) for provenance; Phase 9 must document the version-unknown case explicitly if unresolvable
- Phase 7 pre-rerun ingest must process 72 remaining papers (`tier-4-multimodal/output/` minus 3 smoke papers minus 4 Plan-02-02 fresh-MineRU papers); projected wall ~15–25h / cost ~$15–35
- Phase 7 ingest run will hit OpenRouter vision-LLM `400 Invalid URL format` errors on larger figures (base64 image data exceeding URL length limit); recommend pre-warming `kv_store_llm_response_cache.json` or routing vision direct to OpenAI/Gemini (off-OpenRouter) to mitigate
- ~~Plan 02-03 smoke gate FAIL pending gap-closure~~ **CLEARED 2026-05-05 by Plan 02-04** — JUDGE_MAX_TOKENS=8192 wired into score._build_judge, Tier 4 re-scored, smoke gate PASS for both Tier 4 (gap target) and Tier 5 (regression check). See 02-04-SUMMARY.md for verbatim verdict JSON.
- **Judge cost ledger underreports on LiteLLM completions**: Plan 02-04's new `ragas-judge-tier-4-20260505T151051Z.json` records $0 despite real spend (smaller now since no retries, but still non-zero per OpenRouter dashboard). Tracked as v1.1 follow-up — augment score.py to parse usage from LiteLLM ModelResponse bodies even when token_usage_parser misses calls. Cross-referenced in both 02-03-SUMMARY.md and 02-04-SUMMARY.md.

## Deferred Items

Items acknowledged and carried forward as v1.1+:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Cleanup | CLEANUP-01: consolidate `tier-N-name/` ↔ `tier_N_name/` shim folders | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |
| Cleanup | CLEANUP-02: `[debug-tier5]` extras for OpenInference + Phoenix | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |
| Methodology | METH-01: bootstrap 95% CI in compare.py | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |
| Methodology | METH-02: full cross-judge re-scoring (all 30 × 5) | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |
| Methodology | METH-03: per-tier per-class min/max/stdev | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |
| Blog Iteration | BLOG-01: rebuttal run pipeline | Tracked in REQUIREMENTS.md v1.1 | 2026-05-04 (init) |

## Session Continuity

Last session: 2026-05-05T15:17:00Z
Stopped at: Phase 2 Plan 02-04 gap closure complete: JUDGE_MAX_TOKENS=8192 landed in score._build_judge via TDD red→green (commits cdbc376 + 7fc6d66), Tier 4 re-scored against existing capture (5/5 faithfulness=1.0), Tier 4 smoke gate PASS, Tier 5 smoke gate PASS (Phase 1 regression check intact). Phase 2 ship gate cleared at smoke-gate level; all 7 plans across Phases 1+2 complete. Orchestrator can now route to Phase 3+ (NaN reason instrumentation, freeze tool, pipeline driver, embedder provenance — all parallel-friendly per ROADMAP overview).
Resume file: .planning/phases/02-tier-4-graphml-regeneration/02-04-SUMMARY.md
