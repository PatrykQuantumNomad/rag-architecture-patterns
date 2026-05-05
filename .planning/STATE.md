---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 3 Plan 03-03 complete: live smoke backstop test test_eval_smoke_nan_reasons added to evaluation/tests/test_eval_smoke_live.py (commit 512ad54) and verified end-to-end against real OpenRouter Gemini 2.5 Flash judge — verdict 2026-05-05 PASS (n_total=5, n_unknown_nan=0, n_scored_post_short_circuit=5; ~$0.014 cost vs $0.05 cost guard). HARN-05 closed at unit (Plan 03-01) + integration (Plan 03-02) + live (Plan 03-03) levels. Phase 3 complete (3/3 plans). Single process deviation routed through user-approved checkpoint:decision: previous agent accidentally executed the live test once during a deselection-check verification step (~$0.005-0.02 extra spend; root cause is pyproject.toml not declaring addopts='-m \"not live\"' — v1.1 hardening). Phase 4 (Freeze Tool) is next."
last_updated: "2026-05-05T19:15:00.000Z"
last_activity: 2026-05-05
progress:
  total_phases: 9
  completed_phases: 3
  total_plans: 11
  completed_plans: 10
  percent: 91
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Produce reproducible, honest numbers for the blog — every claim backed by a captured run with full provenance.
**Current focus:** Phase 3 complete (NaN Reason Instrumentation, all 3 plans delivered, HARN-05 closed end-to-end). Phase 4 (Freeze Tool) is next.

## Current Position

Phase: 3 of 9 (NaN Reason Instrumentation) — COMPLETE (3/3 plans delivered, HARN-05 closed end-to-end). Next: Phase 4 (Freeze Tool).
Plan: 3 of 3 complete in Phase 3; Plan 03-03 added test_eval_smoke_nan_reasons live backstop and verified live verdict PASS (n_unknown_nan=0) against real Gemini 2.5 Flash output.
Status: Ready to execute Phase 4
Last activity: 2026-05-05

Progress: [█████████░] 91%
Phase 3 progress: 3/3 plans complete. HARN-05 delivered at unit (03-01) + integration (03-02) + live (03-03) levels.

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: ~19 min (~191 min total / 10 plans)
- Total execution time: ~3.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-tier-5-adapter-fix | 3 | ~22 min | ~7 min |
| 02-tier-4-graphml-regeneration | 4 | ~142 min (~50 min Plan 02-01 + ~50 min Plan 02-02 host MineRU + ~30 min Plan 02-03 + ~12 min Plan 02-04) | ~36 min |
| 03-nan-reason-instrumentation | 3 of 3 | ~27 min (~5 min Plan 03-01 + ~12 min Plan 03-02 + ~10 min Plan 03-03 with 78s live test wall) | ~9 min |

**Recent Trend:**

- Last 10 plans: 01-01 (8 min), 01-03 (4 min), 01-02 (~10 min), 02-01 (~50 min), 02-02 (~50 min host MineRU), 02-03 (~30 min Task 1 + Task 2 with 3 capture cycles), 02-04 (~12 min — score-only TDD + re-score + double smoke gate), 03-01 (~5 min — pure-offline TDD for two coupled units), 03-02 (~12 min — wiring + 4 integration tests with stub LLMs + 1 compare regression test, single atomic commit), 03-03 (~10 min wall — 1 live-marked test, 78.78s live invocation, ~$0.014 cost)
- Trend: live-ingest plans are 5-10× pure-code plans; smoke verification plans (Plan 02-03) sit in between; gap-closure score-only plans (Plan 02-04) are 2-3× pure-code plans; pure-offline TDD-only plans (Plan 03-01) are the cheapest at ~5 min wall; live-smoke-backstop plans (Plan 03-03) are 2× pure-offline (~10 min — code is small, but live test wall is the dominant cost).

*Updated after each plan completion*
| Phase 03 P02 | 12min | 2 tasks | 3 files |
| Phase 03 P03 | 10min | 2 tasks | 1 source + 4 docs |

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
- [Phase ?]: Plan 03-02 wires NaNReasonTracer into score_query_log via callbacks=[tracer] alongside existing kwargs (Pitfall 6 of 03-RESEARCH.md — RAGAS appends its own callbacks rather than replacing); _classify_post_evaluate_nan called per-metric with documented faithfulness > AR > CP precedence in BOTH the dataframe branch AND the result.scores fallback branch (parity for older RAGAS 0.4.x patches). 4 new offline integration tests + 1 compare regression test; no live API spend. Single atomic commit fe52528.
- [Phase ?]: Plan 03-02 clean-path test assertion relaxed per the plan's own docstring guidance — observed RAGAS 0.4.3 scores _CleanLLM as AR=1.0, CP~=1.0, faithfulness NaN with classified 'json_parse_failure' (RagasOutputParserException on NLI prompt). This is EXACTLY the wiring chain HARN-05 needs. Strict 'is None' replaced with 'in {None, json_parse_failure, llm_did_not_finish, empty_statements, empty_questions, invalid_verdicts}' + disallowed silent-drop state check. Rule-1 deviation tracked in 03-02-SUMMARY.md.
- [Phase ?]: Plan 03-02 verifies Architectural Responsibility Map claim 'compare.py needs ZERO change for HARN-05' end-to-end: test_aggregate_tier_with_new_reasons exercises BOTH aggregate_tier (nan_breakdown buckets json_parse_failure + empty_statements + empty_contexts) AND emit_markdown footer rendering with new reason strings + joined-sorted breakdown line, all without modifying compare.py. Gate 4 (git diff compare.py == 0 lines) confirms byte-identical.
- Plan 03-03 live smoke backstop verified — n_unknown_nan=0 against real Gemini judge output. HARN-05 closed at unit + integration + live levels. Live verdict 2026-05-05 PASS: n_total=5, n_unknown_nan=0, n_scored_post_short_circuit=5; ~$0.014 cost (45 judge calls × ~$0.0003) vs $0.05 cost guard; 78.78s wall time. The classifier covers every real RAGAS 0.4.3 NaN path actually exercised by the Tier 5 smoke set against Gemini 2.5 Flash. Phase 7's full 5-tier rerun re-running this same live test against the new captures is the regression check that the wiring continues to cover every actual NaN path post-Phase-3.
- Plan 03-03 single process deviation routed through user-approved checkpoint:decision: previous agent accidentally executed the live test once during a deselection-check verification step (~$0.005-0.02 extra spend; total live cost across both invocations bounded at ~$0.028 — well under the $0.05 cost guard). Root cause: pyproject.toml registers the `live` marker under `[tool.pytest.ini_options].markers` but does NOT declare `addopts = "-m 'not live'"`, so a bare `pytest path/to/file.py -v` (no -m flag) silently runs live tests. The Phase 1 Plan 01-02 + Phase 2 Plan 02-03 + Plan 03-03 plans all assumed live-deselect-by-default semantics that pytest does not actually enforce without explicit addopts. Out of scope per RULE 4 — pyproject.toml is not in Plan 03-03's files_modified list and the change has cross-cutting impact on every other live test in the repo. Tracked as v1.1 hardening below.
- Plan 03-03 token-counts-zero observation is provenance noise, not a gate failure: the verbatim stdout shows judge_input_tokens=0 / judge_output_tokens=0 even though real spend occurred (~$0.014 estimate consistent with plan a-priori ~$0.005-0.02). Same v1.1 follow-up tracked since Plan 02-04 (judge cost ledger underreports on LiteLLM completions) — augment score.py's token_usage_parser to parse usage from LiteLLM ModelResponse bodies even when the parser misses some calls. The cost guard is bounded by question×metric×call count regardless.

### Pending Todos

None yet.

### Blockers/Concerns

- Tier 5 root cause is high-confidence (5-line adapter bug at `tier_5.py:125`) but Phase 1 must include a fallback path: if smoke test still shows empty contexts after the fix, follow STACK.md decision tree (simplify schema → switch model slug → bump openai-agents) before declaring done
- OpenRouter passthrough may obscure the exact Gemini model version (`gemini-2.5-flash-001` vs bare `gemini-2.5-flash`) for provenance; Phase 9 must document the version-unknown case explicitly if unresolvable
- Phase 7 pre-rerun ingest must process 72 remaining papers (`tier-4-multimodal/output/` minus 3 smoke papers minus 4 Plan-02-02 fresh-MineRU papers); projected wall ~15–25h / cost ~$15–35
- Phase 7 ingest run will hit OpenRouter vision-LLM `400 Invalid URL format` errors on larger figures (base64 image data exceeding URL length limit); recommend pre-warming `kv_store_llm_response_cache.json` or routing vision direct to OpenAI/Gemini (off-OpenRouter) to mitigate
- ~~Plan 02-03 smoke gate FAIL pending gap-closure~~ **CLEARED 2026-05-05 by Plan 02-04** — JUDGE_MAX_TOKENS=8192 wired into score._build_judge, Tier 4 re-scored, smoke gate PASS for both Tier 4 (gap target) and Tier 5 (regression check). See 02-04-SUMMARY.md for verbatim verdict JSON.
- **Judge cost ledger underreports on LiteLLM completions**: Plan 02-04's new `ragas-judge-tier-4-20260505T151051Z.json` records $0 despite real spend (smaller now since no retries, but still non-zero per OpenRouter dashboard); manifested again in Plan 03-03's live smoke as `judge_input_tokens=0 / judge_output_tokens=0`. Tracked as v1.1 follow-up — augment score.py to parse usage from LiteLLM ModelResponse bodies even when token_usage_parser misses calls. Cross-referenced in 02-03-SUMMARY.md, 02-04-SUMMARY.md, and 03-03-SUMMARY.md.
- **Live tests not deselected by default in pyproject.toml** (NEW v1.1 hardening item, surfaced by Plan 03-03 process deviation): pyproject.toml declares the `live` marker but does NOT include `addopts = "-m 'not live'"` under `[tool.pytest.ini_options]`. Consequence: a bare `pytest evaluation/tests/test_eval_smoke_live.py -v` (no -m flag) silently runs live tests, including any newly-added ones. Plan 03-03 manifested this gap when a previous agent accidentally executed the live nan_reasons test during a deselection-check verification step. Recommended v1.1 fix: add `addopts = "-m 'not live'"` and update the live-test invocation pattern in plans + CLAUDE.md to use `-m live -k <test>` explicitly. Out of scope per Plan 03-03's RULE 4 (pyproject.toml not in files_modified; cross-cutting impact on every other live test in repo).

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
| Tooling | Pytest live-marker deselect-by-default: add `addopts = "-m 'not live'"` to `[tool.pytest.ini_options]` so a bare `pytest path/to/file.py -v` cannot accidentally consume API budget | New v1.1 item | 2026-05-05 (Plan 03-03) |
| Tooling | LiteLLM judge token-usage parser: augment score.py's `token_usage_parser=get_token_usage_for_openai` to parse usage from LiteLLM ModelResponse bodies even when the parser misses some calls | Tracked since Plan 02-04, re-confirmed by Plan 03-03 | 2026-05-05 (re-confirmed) |

## Session Continuity

Last session: 2026-05-05T19:15:00.000Z
Stopped at: Phase 3 Plan 03-03 complete — live smoke backstop test_eval_smoke_nan_reasons added to evaluation/tests/test_eval_smoke_live.py (commit 512ad54) and verified end-to-end against real OpenRouter Gemini 2.5 Flash judge — verdict 2026-05-05 PASS (n_total=5, n_unknown_nan=0, n_scored_post_short_circuit=5; ~$0.014 cost vs $0.05 cost guard; 78.78s wall time). HARN-05 closed at unit (Plan 03-01) + integration (Plan 03-02) + live (Plan 03-03) levels. Phase 3 complete (3/3 plans). Single process deviation routed through user-approved checkpoint:decision: previous agent accidentally executed the live test once during a deselection-check verification step (~$0.005-0.02 extra spend; root cause is pyproject.toml not declaring `addopts = "-m 'not live'"` — v1.1 hardening). Phase 4 (Freeze Tool) is next.
Resume file: None
