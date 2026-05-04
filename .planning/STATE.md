---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
stopped_at: Phase 1 Plan 03 complete — fallback scaffolding (debug-tier5 extra + diagnostics module + runbook) landed
last_updated: "2026-05-04T14:51:01Z"
last_activity: 2026-05-04 — Phase 01 Plan 03 executed (debug-tier5 extra + FallbackLog + fallback runbook)
progress:
  total_phases: 9
  completed_phases: 0
  total_plans: 3
  completed_plans: 2
  percent: 22
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Produce reproducible, honest numbers for the blog — every claim backed by a captured run with full provenance.
**Current focus:** Phase 1 (Tier 5 Adapter Fix)

## Current Position

Phase: 1 of 9 (Tier 5 Adapter Fix)
Plan: Plan 01 complete; Plan 03 complete (Wave 2 — parallel with Plan 02). Plan 02 status owned by its executor.
Status: In progress
Last activity: 2026-05-04 — Phase 01 Plan 03 executed (debug-tier5 extra + FallbackLog + fallback runbook)

Progress: [██░░░░░░░░] 22%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: 6 min
- Total execution time: 0.20 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-tier-5-adapter-fix | 2 | 12 min | 6 min |

**Recent Trend:**

- Last 5 plans: 01-01 (8 min), 01-03 (4 min)
- Trend: -

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

### Pending Todos

None yet.

### Blockers/Concerns

- Tier 5 root cause is high-confidence (5-line adapter bug at `tier_5.py:125`) but Phase 1 must include a fallback path: if smoke test still shows empty contexts after the fix, follow STACK.md decision tree (simplify schema → switch model slug → bump openai-agents) before declaring done
- OpenRouter passthrough may obscure the exact Gemini model version (`gemini-2.5-flash-001` vs bare `gemini-2.5-flash`) for provenance; Phase 9 must document the version-unknown case explicitly if unresolvable

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

Last session: 2026-05-04T14:51:01Z
Stopped at: Phase 01 Plan 03 complete — fallback scaffolding landed (commits 4be0f0a, 3e8c601, 99e02b0). Plan 01-02 ran in parallel; its status is owned by that executor's SUMMARY.md.
Resume file: .planning/phases/01-tier-5-adapter-fix/01-02-PLAN.md (or next phase if 01-02 also complete)
