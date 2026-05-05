---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
stopped_at: "Phase 2 Plan 02-01 complete (smoke-only ingest per orchestrator Option B): 2886 nodes / 7056 edges in graphml; 3/3 smoke papers in kv_store_full_docs.json; provenance manifest at evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json (committed at 39f02cd); 72 papers explicitly deferred to Phase 7 pre-rerun ingest; cost $1.89. Plans 02-02 and 02-03 remain in Phase 2."
last_updated: "2026-05-05T11:35:00Z"
last_activity: "2026-05-05 — Phase 02 Plan 02-01 Task 2b smoke-only ingest landed (3 of 75 papers per orchestrator Option B); diagnostics manifest committed at 39f02cd; cost $1.89 against bounded smoke budget"
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 6
  completed_plans: 5
  percent: 83
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Produce reproducible, honest numbers for the blog — every claim backed by a captured run with full provenance.
**Current focus:** Phase 2 (Tier 4 Graphml Regeneration) — Plan 02-01 complete (smoke-only); Plans 02-02 + 02-03 remain

## Current Position

Phase: 2 of 9 (Tier 4 Graphml Regeneration) — Plan 02-01 complete
Plan: 1 of 3 complete in Phase 2 (02-02 and 02-03 remain)
Status: Plan 02-01 landed smoke-only graphml (2886 nodes / 7056 edges, 3 papers); 72 papers deferred to Phase 7 per orchestrator Option B
Last activity: 2026-05-05 — Phase 02 Plan 02-01 Task 2b smoke-only ingest committed at 39f02cd

Progress: [█░░░░░░░░░] 11% (1/9 phases complete; Phase 2 in progress 1/3 plans)

## Performance Metrics

**Velocity:**

- Total plans completed: 4
- Average duration: ~20 min
- Total execution time: ~1.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-tier-5-adapter-fix | 3 | ~22 min | ~7 min |
| 02-tier-4-graphml-regeneration | 1 | ~50 min (Plan 02-01 Task 2b live ingest) | ~50 min |

**Recent Trend:**

- Last 5 plans: 01-01 (8 min), 01-03 (4 min), 01-02 (~10 min incl. live smoke + checkpoint), 02-01 (~50 min Task 2b live ingest, smoke-only)
- Trend: live-ingest plans are an order of magnitude longer than pure-code plans, as expected

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

### Pending Todos

None yet.

### Blockers/Concerns

- Tier 5 root cause is high-confidence (5-line adapter bug at `tier_5.py:125`) but Phase 1 must include a fallback path: if smoke test still shows empty contexts after the fix, follow STACK.md decision tree (simplify schema → switch model slug → bump openai-agents) before declaring done
- OpenRouter passthrough may obscure the exact Gemini model version (`gemini-2.5-flash-001` vs bare `gemini-2.5-flash`) for provenance; Phase 9 must document the version-unknown case explicitly if unresolvable
- Phase 7 pre-rerun ingest must process 72 remaining papers (`tier-4-multimodal/output/` minus 3 smoke papers minus 4 Plan-02-02 fresh-MineRU papers); projected wall ~15–25h / cost ~$15–35
- Phase 7 ingest run will hit OpenRouter vision-LLM `400 Invalid URL format` errors on larger figures (base64 image data exceeding URL length limit); recommend pre-warming `kv_store_llm_response_cache.json` or routing vision direct to OpenAI/Gemini (off-OpenRouter) to mitigate

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

Last session: 2026-05-05T11:35:00Z
Stopped at: Phase 2 Plan 02-01 complete (smoke-only Task 2b per orchestrator Option B). Provenance manifest at `.planning/phases/02-tier-4-graphml-regeneration/02-01-SUMMARY.md` and `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json`; storage at `rag_anything_storage/tier-4-multimodal/` (gitignored). 72 papers deferred to Phase 7. Next: Plan 02-02 (parse 4 missing papers) and Plan 02-03 (smoke verification).
Resume file: .planning/phases/02-tier-4-graphml-regeneration/02-01-SUMMARY.md
