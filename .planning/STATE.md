---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
stopped_at: "Phase 3 Plan 03-01 complete: NaNReasonTracer(BaseCallbackHandler) + _classify_post_evaluate_nan pure helper landed in evaluation/harness/score.py via TDD red→green (commits e97e864 + bc80825). 15 new unit tests appended (test count 13→28); 109 pure-addition LOC in score.py (no protected function bodies modified — _short_circuit_nan, _to_float_or_none, _build_judge, score_query_log all byte-identical). All 7 verification gates pass. HARN-05 ships in stages: 03-01 = standalone units, 03-02 = wire into score_query_log + integration test, 03-03 = live smoke backstop."
last_updated: "2026-05-05T17:50:53Z"
last_activity: "2026-05-05 — Phase 03 Plan 03-01 executed end-to-end (~5 min): Task 1 RED appended 15 tests to evaluation/tests/test_eval_score.py covering 6 NaNReasonTracer scenarios (A-F: ROW-only no-capture, METRIC RagasOutputParserException, METRIC LLMDidNotFinishException, RAGAS_PROMPT inherits via parent walk, two metrics same row, idempotence first-wins) + 8 _classify_post_evaluate_nan branches (G-N: non-NaN→None, captured RagasOutputParserException, captured LLMDidNotFinishException, faithfulness→empty_statements, answer_relevancy→empty_questions, context_precision→invalid_verdicts, unknown metric→unknown_nan+caplog WARNING, captured-unknown-type→falls to semantic) + 1 smoke import test — all 15 fail with ImportError (RED commit e97e864). Task 2 GREEN added imports (logging, BaseCallbackHandler, ChainType, _NAN_REASON_LOG) + NaNReasonTracer class with on_chain_start ChainType-dispatch + on_chain_error idempotent capture + _classify_post_evaluate_nan precedence ladder, placed BETWEEN _short_circuit_nan and _to_float_or_none — all 28 tests pass (commit bc80825). Zero deviations; plan executed exactly as written. Phase 3 progress: 1/3 plans complete; Plan 03-02 will wire tracer into score_query_log."
progress:
  total_phases: 9
  completed_phases: 2
  total_plans: 10
  completed_plans: 8
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Produce reproducible, honest numbers for the blog — every claim backed by a captured run with full provenance.
**Current focus:** Phase 3 (NaN Reason Instrumentation) — Plan 03-01 complete (units shipped via TDD); Plan 03-02 (wiring into score_query_log + integration tests) is next.

## Current Position

Phase: 3 of 9 (NaN Reason Instrumentation) — IN PROGRESS (1/3 plans delivered)
Plan: 1 of 3 complete in Phase 3; Plan 03-01 shipped NaNReasonTracer + _classify_post_evaluate_nan as standalone testable units via TDD red→green (offline; no live LLM call required)
Status: Phase 1+2 fully delivered (7/7 plans, both ship gates cleared at smoke-gate level). Phase 3 Plan 03-01 ships the two new units in isolation with 15 unit tests (all 28 tests pass, 13 existing untouched). Plan 03-02 wires NaNReasonTracer into evaluate(callbacks=[tracer], ...) inside score_query_log and post-processes each row's metrics through _classify_post_evaluate_nan to populate ScoreRecord.nan_reason. Plan 03-03 is the live smoke backstop with a checkpoint:human-verify gate.
Last activity: 2026-05-05 — Phase 03 Plan 03-01 executed end-to-end (~5 min): TDD red→green for NaNReasonTracer (BaseCallbackHandler subclass) + _classify_post_evaluate_nan (pure post-evaluate classifier with precedence ladder). 15 new tests, 109 LOC additions to score.py, 0 protected function bodies modified.

Progress: [████████░░] 80% plan-level (8/10 plans across Phases 1+2+Phase-3-Plan-1; Phase 3 has 3 plans, currently 1/3) | 22% phase-level (2/9 phases fully complete: Phase 1 + Phase 2 cleared in ROADMAP; Phase 3 in progress)
Phase 3 progress: 1/3 plans complete (Plan 03-01 units shipped). HARN-05 closure pending Plan 03-02 wiring + Plan 03-03 live smoke backstop.

## Performance Metrics

**Velocity:**

- Total plans completed: 8
- Average duration: ~21 min (~169 min total / 8 plans)
- Total execution time: ~2.8 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-tier-5-adapter-fix | 3 | ~22 min | ~7 min |
| 02-tier-4-graphml-regeneration | 4 | ~142 min (~50 min Plan 02-01 + ~50 min Plan 02-02 host MineRU + ~30 min Plan 02-03 + ~12 min Plan 02-04) | ~36 min |
| 03-nan-reason-instrumentation | 1 of 3 | ~5 min (Plan 03-01 only) | ~5 min |

**Recent Trend:**

- Last 8 plans: 01-01 (8 min), 01-03 (4 min), 01-02 (~10 min), 02-01 (~50 min), 02-02 (~50 min host MineRU), 02-03 (~30 min Task 1 + Task 2 with 3 capture cycles), 02-04 (~12 min — score-only TDD + re-score + double smoke gate), 03-01 (~5 min — pure-offline TDD for two coupled units)
- Trend: live-ingest plans are 5-10× pure-code plans; smoke verification plans (Plan 02-03) sit in between; gap-closure score-only plans (Plan 02-04) are 2-3× pure-code plans; pure-offline TDD-only plans (Plan 03-01) are the cheapest at ~5 min wall (no LLM call, no graph touch, no capture re-run — just RED tests + GREEN implementation + verification gates).

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
- Plan 03-01 ships HARN-05 stage 1 (units only): NaNReasonTracer(BaseCallbackHandler) + _classify_post_evaluate_nan pure helper landed in evaluation/harness/score.py via TDD red→green (RED commit e97e864 with 15 failing tests, GREEN commit bc80825 with 109 LOC of impl). The two units are tightly coupled (classifier reads tracer.errors dict) but both pure → land in a single plan. Plan 03-02 wires them into score_query_log via evaluate(callbacks=[tracer], ...) and post-processes each row through _classify_post_evaluate_nan to populate ScoreRecord.nan_reason; Plan 03-03 is the live smoke backstop with checkpoint:human-verify.
- Plan 03-01 idempotence semantics chose 'first wins' (leaf-most error preserved) over 'last wins' (root-most error preserved). Rationale: RAGAS_PROMPT is the leaf and that's where RagasOutputParserException originates; METRIC and ROW catch-and-re-raise the same exception, so leaf is the most specific signal for HARN-05 reason classification. If a future RAGAS version starts wrapping leaf exceptions in higher-level types (e.g. METRIC catches and re-raises as RagasMetricException), this decision needs revisiting — Plan 03-02 integration tests with a stub LLM should surface that scenario.
- Plan 03-01 _classify_post_evaluate_nan precedence ladder: captured-exception-type FIRST (specific — json_parse_failure / llm_did_not_finish), then per-metric semantic NaN paths (general — empty_statements / empty_questions / invalid_verdicts), then 'unknown_nan' + WARNING log (defensive — Pitfall 7 of 03-RESEARCH.md says never silently drop NaN). Captured-but-unknown-type (e.g. 'TimeoutError') falls through to per-metric semantic mapping rather than 'unknown_nan' so a known-metric NaN with a novel exception type still gets a meaningful reason string. Test N covers this exact branch.
- Plan 03-01 placement: NaNReasonTracer + _classify_post_evaluate_nan inserted BETWEEN _short_circuit_nan (existing, byte-identical) and _to_float_or_none (existing, byte-identical) so all NaN-related helpers cluster together. score_query_log signature unchanged (verified via `git diff | grep "def score_query_log"` returning empty); Plan 03-02 will modify it.
- Plan 03-01 zero deviations: plan executed exactly as written. langchain_core + ragas.callbacks imports verified working in .venv before any code change (one extra `python -c "..."` sanity check, not a deviation). DeprecationWarnings on `from ragas.metrics import ...` in score.py:202 predate Plan 03-01 (visible in Plan 02-04 final pytest run) — out of scope per RULE 4.

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

Last session: 2026-05-05T17:50:53Z
Stopped at: Phase 3 Plan 03-01 complete: NaNReasonTracer(BaseCallbackHandler) + _classify_post_evaluate_nan pure helper landed in evaluation/harness/score.py via TDD red→green (commits e97e864 + bc80825). 15 new unit tests, +109 LOC score.py, +253 LOC test_eval_score.py. All 28 tests pass; all 7 verification gates PASS. Zero deviations. Phase 3 progress: 1/3 plans complete. Plan 03-02 will wire NaNReasonTracer into score_query_log's evaluate(callbacks=[tracer], ...) and post-process each row through _classify_post_evaluate_nan to populate ScoreRecord.nan_reason; Plan 03-03 is the live smoke backstop with checkpoint:human-verify.
Resume file: .planning/phases/03-nan-reason-instrumentation/03-01-SUMMARY.md
