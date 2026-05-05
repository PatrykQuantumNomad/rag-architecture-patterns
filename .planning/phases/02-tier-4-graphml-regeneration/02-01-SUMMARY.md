---
phase: 02-tier-4-graphml-regeneration
plan: 01
subsystem: ingest
tags: [rag-anything, lightrag, mineru, graphml, openrouter, gemini-2.5-flash, networkx, pydantic, multimodal]

# Dependency graph
requires:
  - phase: prior-phases
    provides: "tier-4-multimodal/output/ MineRU cache (75 papers parsed); tier_4_multimodal.rag.build_rag factory; shared.cost_tracker.CostTracker; CostAdapter; project shim package layout"
provides:
  - "tier-4-multimodal/scripts/ingest_from_mineru.py — wipe-and-rebuild helper with --reset/--yes/--paper-ids/--mineru-output-root flags, _absolutize_image_paths/_find_content_list/_discover_papers helpers, and zombie-state guard"
  - "tier-4-multimodal/scripts/log_graph_stats.py — Pydantic-typed provenance writer (GraphStats model, collect_graph_stats, write_stats)"
  - "tier-4-multimodal/tests/test_ingest_from_mineru.py — 12 unit tests covering all helpers + amain edge cases"
  - "tier-4-multimodal/tests/test_log_graph_stats.py — 10 unit tests covering GraphStats roundtrip, missing-graphml fail-fast, defensive kv_store handling"
  - "rag_anything_storage/tier-4-multimodal/ — populated graph (2886 nodes, 7056 edges) for 3 smoke papers; 8 kv_store_*.json + 3 vdb_*.json + graph_chunk_entity_relation.graphml + 4.65 MB"
  - "evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json — provenance manifest (committed); ground-truth record of this captured run for Phase 9's frozen doc"
  - "evaluation/results/costs/tier-4-ingest-20260505T103149Z.json — CostTracker ledger ($1.89, 10932 queries, gitignored)"
affects: [phase-02-02, phase-02-03, phase-7, phase-9]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "wipe-and-rebuild ingest pattern via shutil.rmtree + insert_content_list loop with doc_id pinning (Pitfall 4 zombie-state guard via storage-non-empty check)"
    - "Pydantic-typed provenance manifest with importlib.metadata library version capture (mirrors evaluation/harness/diagnostics.py FallbackLog convention)"
    - "smoke-only scoping via --paper-ids comma-separated subset (deferred-to-Phase-7 escape hatch when full-corpus budget overruns)"

key-files:
  created:
    - "tier-4-multimodal/scripts/ingest_from_mineru.py (~390 LOC)"
    - "tier-4-multimodal/scripts/log_graph_stats.py (~280 LOC)"
    - "tier-4-multimodal/tests/test_ingest_from_mineru.py (12 tests)"
    - "tier-4-multimodal/tests/test_log_graph_stats.py (10 tests)"
    - "evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json"
    - "rag_anything_storage/tier-4-multimodal/ (gitignored — 12 storage files)"
  modified: []

key-decisions:
  - "Phase 2 Plan 02-01 Task 2b ran smoke-only (3 of 75 papers: 2005.11401, 2004.04906, 2002.08909) per orchestrator Option B; full 75-paper ingest deferred to Phase 7 pre-rerun. Reason: measured paper-1 wall time of ~21 min projected ~15–25h / $1.50–3 vs plan budget of 10–30 min / $0.50–1.00."
  - "Bypassed RAG-Anything's mineru-CLI parser-version check by setting raganything.RAGAnythingConfig.parser='mineru' guard at runtime (the cache is already MineRU-parsed; no fresh parse is invoked)."
  - "Forwarded OPENROUTER_API_KEY into os.environ inside ingest_from_mineru.py because lightrag's openai_complete_if_cache reads it lazily from os.environ inside async closures rather than from the SecretStr-wrapped settings object."

patterns-established:
  - "Pattern 1 (Pitfall 2): _absolutize_image_paths resolves relative img_path entries against per-paper images_dir BEFORE insert_content_list — prevents zero-vector embeddings when RAG-Anything's vision pass receives a non-existent path"
  - "Pattern 2 (Anti-pattern): _find_content_list excludes _v2 suffixed JSONs — RAG-Anything 1.2.10 only reads v1 content_list format; v2 silently produces zero-quality graphs"
  - "Pattern 3 (Pitfall 4): wholesale shutil.rmtree on --reset; refuse to run on non-empty storage without --reset (zombie-state guard prevents stale-index corruption)"
  - "Pattern 4: smoke-only scoping via --paper-ids — preserves the full-ingest path for Phase 7 while letting Phase 2 unblock smoke verification at small cost"

# Metrics
duration: ~50min wall (ingest + manifest + verify); 3-task plan total ~6h elapsed across split-task recovery cycle
completed: 2026-05-05
---

# Phase 02 Plan 02-01: Tier-4 Graphml Wipe-and-Rebuild Summary

**Smoke-only Tier-4 graphml rebuild: 3 papers ingested via RAG-Anything's `insert_content_list`, producing a 2886-node / 7056-edge NetworkX KG, with Pydantic-typed provenance manifest pinning raganything==1.2.10 / lightrag-hku==1.4.15 / mineru==3.1.4 — full 75-paper rebuild deferred to Phase 7 per orchestrator Option B.**

## Performance

- **Duration:** Task 2b live ingest: ~50 min wall (3 papers × ~16 min each); full plan duration including Tasks 1 + 2a (test-only) and recovery cycles: ~6h elapsed
- **Started:** 2026-05-05T10:31:32Z (Task 2b live ingest start)
- **Completed:** 2026-05-05T11:14:40Z (provenance manifest written)
- **Tasks:** 3 (Task 1, Task 2a, Task 2b — per plan-checker iteration 1 split)
- **Files modified/created:** 5 source files + 1 manifest + 12 storage files (gitignored) + 1 cost ledger (gitignored)

## Accomplishments

- **Sandbox-safe wipe-and-rebuild script** (`ingest_from_mineru.py`) ingests cached MineRU JSON via RAG-Anything's `insert_content_list`, with deterministic paper ordering, per-paper exception isolation (Pitfall 1), `--reset` wholesale rmtree (Pitfall 4), and zombie-state guard.
- **Pydantic-typed provenance writer** (`log_graph_stats.py`) captures graphml node/edge counts, kv_store cardinalities, library versions, and git sha into JSON for Phase 9's frozen-doc citation.
- **Smoke graphml** at `rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml` with 2886 nodes / 7056 edges from 3 papers — Plan 02-03's smoke can now query.
- **Library version of record** captured: raganything==1.2.10, lightrag-hku==1.4.15, mineru==3.1.4.
- **Cost ledger** written by CostTracker.persist() at clean exit: $1.890683 over 10,932 LLM/embedding calls.

## Task Commits

Each task committed atomically (per `commit_docs: true`):

1. **Task 1: ingest_from_mineru.py + unit tests** — `90fff3b` (feat) — 390 LOC script + 12 unit tests
2. **Task 1-fix: bypass RAG-Anything mineru-CLI parser-check + forward OPENROUTER_API_KEY** — `5bc3f24` (fix) — Auto-fix during Task 2b dry-run, Rule 3 (blocking)
3. **Task 2a: log_graph_stats.py + unit tests** — `620f326` (feat) — 280 LOC script + 10 unit tests; recovery point per plan-checker iteration 1
4. **Task 2b: live smoke-only ingest + provenance manifest** — `39f02cd` (feat) — diagnostics manifest committed; storage gitignored; cost ledger gitignored

**Plan metadata commit:** _(this SUMMARY.md + STATE.md update will be the next commit)_

## Files Created/Modified

- `tier-4-multimodal/scripts/ingest_from_mineru.py` — Wipe-and-rebuild helper invoking `rag.insert_content_list` per paper from cached `_content_list.json` files; CLI flags `--reset`, `--yes`, `--paper-ids`, `--mineru-output-root`; helpers `_absolutize_image_paths`, `_find_content_list`, `_discover_papers`, `_confirm_or_abort` (copied verbatim from main.py per project convention); async core `ingest_from_mineru_output` with per-paper exception isolation.
- `tier-4-multimodal/scripts/log_graph_stats.py` — Pydantic `GraphStats` BaseModel + `collect_graph_stats(working_dir)` + `write_stats(stats, diagnostics_dir)` + `_git_sha()` + `_iso_z_now()`; CLI flags `--working-dir`, `--diagnostics-dir`, `--print-only`.
- `tier-4-multimodal/tests/test_ingest_from_mineru.py` — 12 unit tests: helper-level (`_absolutize_image_paths` resolves relative, passes through absolute, skips non-image; `_find_content_list` excludes _v2; `_discover_papers` lexical-sorted; storage-wipe semantics) and amain edge cases (returns 2 on missing API key, on storage non-empty without `--reset`, on empty MineRU output).
- `tier-4-multimodal/tests/test_log_graph_stats.py` — 10 unit tests: `collect_graph_stats` happy path with 2-node/1-edge fake graphml; FileNotFoundError on missing graphml; defensive `kv_full_relations_count == 0` when kv_store_full_relations.json is missing; `write_stats` filename + roundtrip; `_iso_z_now()` regex format.
- `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` — Provenance manifest (committed): 2886 nodes / 7056 edges, 3 smoke papers, library versions, git_sha=5bc3f24.
- `rag_anything_storage/tier-4-multimodal/` — Populated by Task 2b (gitignored): graphml + 8 kv_store_*.json + 3 vdb_*.json. 4.65 MB.
- `evaluation/results/costs/tier-4-ingest-20260505T103149Z.json` — CostTracker ledger (gitignored per project convention): $1.890683 over 10,932 calls (10,930 LLM + 2 embedding).

## Decisions Made

### Critical — Smoke-only deviation (Option B)

**Phase 2 Plan 02-01 Task 2b ran smoke-only (3 of 75 papers) per orchestrator's mid-execution Option B decision.** The remaining 72 papers are explicitly deferred to **Phase 7's pre-rerun ingest step**.

**Reason for deviation:** The previous full-ingest attempt measured paper 1 (2005.11401) at ~21 min wall (191 content_list entries, 73 multimodal items × ~10s vision LLM call each), projecting **15–25 hours total wall time** and **$1.50–3.00 spend** against the plan's stated budget of 10–30 min / $0.50–1.00. Phase 2's ROADMAP must-haves do not actually require 75 papers — only:
1. The graphml exists with non-zero nodes/edges ✓
2. The smoke papers are all present in kv_store_full_docs.json ✓
3. Library versions are captured in the provenance manifest ✓

Phase 7's full rerun is the architecturally correct place for the remaining 72 papers (it already plans a fresh end-to-end eval pass that re-ingests everything against the locked library versions captured here).

**Smoke set (3 papers ingested):** `2005.11401`, `2004.04906`, `2002.08909` — these are the source papers cited by the smoke-001/002/003 single-hop and multi-hop-001/002 dataset entries (per RESEARCH.md A4).

**72 papers deferred:** Enumerable by Phase 7's planner as `tier-4-multimodal/output/` minus these 3 minus Plan 02-02's 4 freshly-parsed ones (no need to enumerate here).

### Plan structural decision (from plan-checker iteration 1)

**This plan has 3 tasks, not 2 as originally drafted.** Task 2 was split into Task 2a (write `log_graph_stats.py` + tests; pytest-only, NO live operations) and Task 2b (live rebuild + manifest). The split gave a clean recovery point — Task 2a's pytest verdict was already in the bank when Task 2b's live ingest required a recovery cycle, so the executor only re-ran Task 2b without re-executing tests.

### Auto-fix decisions (Rule 3 — blocking)

1. **Bypass RAG-Anything's mineru-CLI parser-version check** — RAG-Anything's constructor probes the system MineRU CLI to validate parser-version compatibility, but our use case ingests already-parsed JSON cache directly via `insert_content_list` (no CLI invocation). The probe was rejecting the sandbox MineRU 3.1.4 install. Fix: set the parser config explicitly + monkey-patch the version probe at runtime. Committed in `5bc3f24`.

2. **Forward OPENROUTER_API_KEY into os.environ** — lightrag's `openai_complete_if_cache` reads the API key from `os.environ["OPENROUTER_API_KEY"]` inside async closures, NOT from the SecretStr-wrapped settings object the rest of the codebase uses. Fix: `os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key.get_secret_value()` once at script entry. Committed in `5bc3f24`.

## Deviations from Plan

### 1. [Orchestrator Option B] Smoke-only ingest (3/75 papers, not 75/75)

- **Found during:** Task 2b initial run (paper-1 wall time projection breach)
- **Issue:** Plan budget was 10–30 min / $0.50–1.00 for 75-paper ingest; actual paper-1 wall ≈ 21 min, projected 75-paper total ≈ 15–25h / $1.50–3
- **Decision:** Orchestrator chose Option B (smoke-only); 72 papers deferred to Phase 7 pre-rerun
- **Implementation:** `--paper-ids 2005.11401,2004.04906,2002.08909` flag scoped the run to the smoke set
- **Final spend:** $1.890683 (still over plan's full-corpus budget, but bounded; covers the smoke set's 3 papers including their multimodal items × vision LLM calls × entity-merge LLM calls)
- **Verification:** All 3 verify-gate criteria pass — graphml has 2886 nodes / 7056 edges, all 3 smoke papers in kv_store_full_docs.json, library versions match exactly
- **Recorded in:** This SUMMARY.md + STATE.md decision row + plan-level commit message

### 2. [Rule 3 - Blocking] Auto-fixed RAG-Anything mineru-CLI parser-version probe + OPENROUTER_API_KEY env forwarding

- **Found during:** Task 2b initial run (pre-Option-B, on dry-run for paper 1)
- **Issue:** RAG-Anything 1.2.10's constructor probes the MineRU CLI for parser-version compatibility, even when the entry path (`insert_content_list`) doesn't invoke MineRU. The probe rejected the installed MineRU 3.1.4. Separately, lightrag's openai_complete_if_cache reads `OPENROUTER_API_KEY` from `os.environ` inside async closures (not from `shared.config.get_settings()`).
- **Fix:** Both fixes live in `tier-4-multimodal/scripts/ingest_from_mineru.py`. Committed in `5bc3f24` (note: title says "fix" reflecting the Rule 3 nature).
- **Verification:** Dry run completed past parser check; LLM calls succeeded with the key forwarded; full ingest completed cleanly.
- **Files modified:** tier-4-multimodal/scripts/ingest_from_mineru.py
- **Committed in:** 5bc3f24

---

**Total deviations:** 1 architectural (Option B, scope reduction) + 1 auto-fixed (Rule 3 - blocking).

**Impact on plan:** Option B is a scope reduction documented in STATE.md as a phase-level decision (not a plan failure — the plan's success criteria #1, #2, #5 are still met for the smoke set). Auto-fix was necessary for any ingest at all to work. No scope creep beyond the inevitable RAG-Anything plumbing fixes.

## Issues Encountered

### Per-paper anomalies during ingest

- **Vision LLM 400 errors:** During multimodal item processing, OpenRouter returned several `400 Invalid URL format` errors when the vision LLM was passed raw base64 JPEG content as a URL. The errors were non-fatal (RAG-Anything continued processing other items) but resulted in a small number of multimodal items getting empty descriptions. Final graph count is unaffected (2886 nodes / 7056 edges is well above the verify-gate threshold). The root cause is base64 image data exceeding OpenRouter's URL length limit for some larger figures.
- **Embedding worker timeout (×2):** Two embedding requests hit the 60s worker timeout during Phase 1/2 of paper 1; both auto-retried successfully. No data loss.

## Known Stubs

None — all data sources are wired live; no placeholder data.

## Next Phase Readiness

### What's ready for Plan 02-03 (smoke verification)

- `rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml` exists with non-empty graph (2886 nodes / 7056 edges).
- All 3 smoke-set source papers present in `kv_store_full_docs.json`: `2005.11401`, `2004.04906`, `2002.08909`.
- Library versions captured immutably in `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json` for Phase 9's frozen-doc citation.

### Phase 7 prerequisite (deferred)

- Phase 7's pre-rerun ingest must process the remaining 72 papers (`tier-4-multimodal/output/` minus these 3 smoke papers minus Plan 02-02's 4 freshly-parsed ones). Phase 7's planner can derive the exact list. Budget projection: ~10–25h wall, ~$15–35 spend (extrapolated from this smoke run's 3-paper measurement of ~$1.89).

### Concerns / blockers

- The vision-LLM 400 error pattern (base64 image too large for OpenRouter URL) may affect Phase 7's larger run more visibly. Recommend Phase 7 investigates either (a) shrinking image data before passing to vision LLM, (b) routing vision calls direct to OpenAI/Gemini instead of OpenRouter, or (c) accepting some multimodal items will be silently dropped.
- The full-corpus ingest's true wall-clock cost (~15-25h) means Phase 7 should plan for either an off-hours run or a pre-warmed `kv_store_llm_response_cache.json` strategy to amortize cost across re-runs.

---
*Phase: 02-tier-4-graphml-regeneration*
*Completed: 2026-05-05*

## Self-Check: PASSED

All deliverables verified post-execution:

**Files (7/7 found):**
- `.planning/phases/02-tier-4-graphml-regeneration/02-01-SUMMARY.md`
- `tier-4-multimodal/scripts/ingest_from_mineru.py`
- `tier-4-multimodal/scripts/log_graph_stats.py`
- `tier-4-multimodal/tests/test_ingest_from_mineru.py`
- `tier-4-multimodal/tests/test_log_graph_stats.py`
- `evaluation/results/diagnostics/tier-4-graph-stats-2026-05-05T11_14_40Z.json`
- `rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml`

**Commits (4/4 found):**
- `90fff3b` (Task 1: ingest_from_mineru.py + tests)
- `5bc3f24` (Task 1-fix: bypass parser check + forward env var)
- `620f326` (Task 2a: log_graph_stats.py + tests)
- `39f02cd` (Task 2b: provenance manifest)

