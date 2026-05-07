# Requirements: RAG Architecture Patterns

**Defined:** 2026-05-04
**Core Value:** Produce reproducible, honest numbers for the blog — every claim in the post must be backed by a captured run with full provenance.

## v1.0 Requirements

Requirements for the eval-numbers handoff to the external blog repo. Each maps to roadmap phases.

### Tier Fixes

Bug fixes that gate every other v1.0 requirement — the eval cannot ship credible numbers until both tiers stop returning 30/30 NaN.

- [x] **TIER-01**: User can run Tier 5 evaluation and get non-empty `retrieved_contexts` populated by walking `RunResult.new_items` for `ToolCallOutputItem` (replaces the hard-coded `retrieved_contexts=[]` at `evaluation/harness/adapters/tier_5.py:125`) — completed 2026-05-04 in Phase 01 Plan 01 (commit baaa573)
- [x] **TIER-02**: User can regenerate clean Tier 4 graphml from a wiped `rag_anything_storage/tier-4-multimodal/` directory by running MineRU CLI on the host machine (outside the sandbox) and feeding parsed JSON into RAG-Anything ingestion — completed 2026-05-05 in Phase 02 Plan 02-01 (commit 39f02cd; smoke-only 3-paper rebuild per orchestrator Option B; full 75-paper rebuild deferred to Phase 7 pre-rerun)
- [~] **TIER-03**: User can verify each tier's fix on a 5-question smoke test before committing to a full 30-question rerun (catches regressions before spending the rerun budget) — Phase 1 half complete (Tier 5 smoke PASS, 2026-05-04); Phase 2 half deliverables shipped 2026-05-05 (Plan 02-03 commits 1bc1ba1 / 17142f8 / 693495e) but smoke gate FAIL on judge max_tokens — pending one-line gap-closure in `evaluation/harness/score.py::_build_judge` (`max_tokens=8192` on litellm.completion)

### Eval Harness

New code added under `evaluation/harness/` to make capture → score → rollup → freeze a single, reproducible operation.

- [x] **HARN-01**: User can run `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5` to execute capture → score → rollup → freeze in one command, with a single git SHA and ISO timestamp captured at start ✓ 2026-05-06 (Phase 5 Plan 05-01 + Plan 05-02: pipeline.py composes capture → score → compare → freeze in-process via 189 raw LOC; sweep_sha = run._git_sha() + sweep_ts = run._ts() captured ONCE at amain entry, threaded into run.amain via new git_sha_override + ts_override kwargs on _capture_tier; verified at unit by Plan 05-01 Test 4 (test_sha_propagation), Test 5 (test_full_sweep_order), Test 14 (test_e2e_with_stub_stages); verified at integration by Plan 05-01 Test 15 (test_single_tier_rerun_preserves_others — proves SHA lands inside written tier-4 JSON on disk, W5 closure); verified at live by Plan 05-02 test_pipeline_live_tier5_smoke (verdict PASS 2026-05-06: rc=0, 158.02s wall, total_usd=$0.007010 ≤ $0.05 D-LIVE-COST-CEILING, queries.git_sha=fd287f0 matches HEAD, 1/1/1/1 tier-5 queries+metrics+tier-5-eval+ragas-judge files on disk, comparison.md regenerated with tier-5 row))
- [x] **HARN-02**: User can re-run only one tier (e.g. `--tiers 4`) without invalidating the captured runs of other tiers (relies on `_latest()` mtime resolution in `compare.py`) ✓ 2026-05-06 (Phase 5 Plan 05-01: compare stage hard-codes tiers="1,2,3,4,5" per D-Q4 so single-tier reruns produce a complete comparison.md; verified by Test 12 (tier-4-from-cache passthrough) + Test 15 (integration: pre-existing tier-{1,2,3,5} files byte-identical after a tiers=4 rerun, regenerated comparison.md contains all 5 tier rows))
- [x] **HARN-03**: User can produce a frozen artifact via `evaluation/harness/freeze.py` that refuses to clobber an existing `frozen/eval-numbers-vX.Y.md` — frozen docs are immutable once written ✓ 2026-05-05 (Phase 4 Plan 04-01: refuse-clobber wording `"already frozen — bump version or pass --force"` exits 2; --force overwrites both md AND manifest; verified by tests test_freeze_refuses_clobber + test_freeze_force_overwrites + CLI quality gate sub-checks 1+2+3)
- [x] **HARN-04**: User can read a sidecar `frozen/eval-numbers-vX.Y.manifest.json` recording the git SHA, capture timestamps per tier, judge model, generation models per tier, and all relevant library versions (lightrag-hku, raganything, openai-agents, ragas) ✓ 2026-05-05 (Phase 4 Plan 04-01: manifest carries `$schema_version` + `version` + `frozen_at` ISO 8601 Z + `git_sha` + `git_dirty` + `source_markdown` + `source_markdown_mtime` + `frozen_markdown` + `judge {model, embedder, max_tokens=8192}` + `per_tier {tier-1..tier-5: status:present + generation_model + capture_timestamp + capture_git_sha + queries/cost/metrics relative paths + mtimes}` + `library_versions {lightrag-hku 1.4.15, raganything 1.2.10, openai-agents 0.14.6, ragas 0.4.3, litellm 1.83.0, chromadb 1.5.8}` + `python_version`; verified by tests test_manifest_top_level_fields + test_manifest_per_tier_provenance + test_manifest_library_versions + test_manifest_judge_block + test_manifest_missing_tier + CLI gate 5)
- [x] **HARN-05**: User can distinguish RAGAS NaN reasons (`empty_contexts` vs. `empty_statements` vs. `json_parse_failure`) in per-row metrics output rather than seeing a single `NaN`

### Capture & Bias

Eval-running activities — the actual numbers production for the blog.

- [ ] **CAP-01**: User can capture a full 5-tier × 30-question RAGAS run on a single date with a single git SHA recorded in every per-tier output JSON
- [x] **CAP-02**: User can run a multi-judge spot-check re-scoring 5 questions × 3 tiers (15 cells) with a non-Gemini judge (Claude Haiku or GPT-4.1-mini), and the resulting delta-from-primary-judge is captured in a structured JSON for the frozen doc
- [ ] **CAP-03**: User can verify per-tier embedding model is recorded in capture JSON so the embedder-confound disclosure in the frozen doc is data-backed (not narrative)

### Handoff Doc

The single artifact that ships from this repo to the blog repo.

- [ ] **DOC-01**: User can produce `evaluation/results/frozen/eval-numbers-v1.0.md` containing the full per-tier rollup table (faithfulness, answer_relevancy, context_precision, mean_latency_s, total_cost_usd, cost_per_query_usd, n, n_nan)
- [ ] **DOC-02**: User can find a per-question-class breakdown (single-hop / multi-hop / multimodal) in the frozen doc, matching `comparison.md` format
- [ ] **DOC-03**: User can find a per-tier provenance line in the frozen doc with capture timestamp, generation model, git SHA, and judge model — sufficient for any reader to reproduce
- [ ] **DOC-04**: User can find honest disclaimers in the frozen doc covering: sample-size limits (30 × 5 too small for stat-sig), self-grading bias (with multi-judge spot-check delta as evidence), multimodal limitation per tier, embedder-confound table, and any RAGAS NaN reasons that remain in the final numbers
- [ ] **DOC-05**: User can copy-paste the frozen doc into the blog repo's `rag-architecture-patterns.mdx` and have all numbers and disclosures travel together (no external lookups required)

## v1.1 Requirements

Deferred to a future milestone. Tracked but not in current roadmap.

### Cleanup

- **CLEANUP-01**: Consolidate `tier-{N}-{name}/` ↔ `tier_{N}_{name}/` folder pairs into a single layout (the underscore folders are setuptools shims; revisit whether the workaround can be removed via `[project.scripts]` entries)
- **CLEANUP-02**: Add OpenInference + Phoenix optional `[debug-tier5]` extras for future Tier 5 debugging without committing the diagnostic stack to default install

### Methodology

- **METH-01**: Bootstrap 95% CI on per-tier rollup means (~10 LOC numpy in `compare.py`) for blog-feedback rebuttals
- **METH-02**: Cross-judge full re-scoring (all 30 × 5 with second judge model) — full bias mitigation, not just measurement
- **METH-03**: Per-tier per-class min/max/stdev added to rollup (currently mean-only)

### Blog Iteration

- **BLOG-01**: Rebuttal run pipeline — invoked when blog feedback questions a specific tier's numbers; produces a dated `eval-numbers-vX.Y.md` re-run for that tier only

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Adding new tiers / new RAG architectures | v1.0 is fix-and-ship the existing 5; the blog narrative is built around them |
| Adding new questions to the golden set | 30 is locked (single-hop / multi-hop / multimodal split is the blog's evidence structure) |
| Statistical-significance testing | Sample size (30 × 5) is too small; magnitude-only is the blog's framing |
| Consolidating `tier-N-name/` ↔ `tier_N_name/` folders | Underscore folders are intentional setuptools shims; deferred to v1.1 cleanup |
| Full self-grading bias mitigation (re-running everything with multiple judges) | Cost-prohibitive and out of scope; spot-check **measurement** is in v1.0, full mitigation is v1.1+ |
| Re-architecting evaluation harness, cost-tracker, or shared layer | v1.0 minimizes new abstractions; only adds `freeze.py` + `pipeline.py` |
| Switching providers (e.g. moving Tier 1 off OpenRouter) | Existing provider mix is locked for v1.0 |
| Cross-language ports of any tier | Python-only; locked by `pyproject.toml` |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TIER-01 | Phase 1 | Complete (2026-05-04, commit baaa573) |
| TIER-02 | Phase 2 | Complete-with-deviation (Plan 02-01, 2026-05-05, commit 39f02cd; smoke-only 3-paper graphml per orchestrator Option B; full 75-paper rebuild deferred to Phase 7 pre-rerun) |
| TIER-03 | Phase 1 + Phase 2 (split: Tier 5 smoke in P1, Tier 4 smoke in P2) | P1 half complete (2026-05-04, Tier 5 smoke PASS); P2 half deliverables shipped (2026-05-05, Plan 02-03) but smoke gate FAIL on judge max_tokens — pending one-line gap-closure in score.py before requirement is fully satisfied |
| HARN-01 | Phase 5 | Complete end-to-end (Plan 05-01 unit + integration 2026-05-06; Plan 05-02 live verdict PASS at total_usd=$0.007010 / 158.02s wall on Tier 5 × 5q, queries.git_sha=fd287f0 matches HEAD, 2026-05-06) |
| HARN-02 | Phase 5 | Complete (Plan 05-01, 2026-05-06) |
| HARN-03 | Phase 4 | Complete (2026-05-05; Plan 04-01 commits 9588056 RED + 7881e87 GREEN; refuse-clobber wording verified by test_freeze_refuses_clobber + CLI gate 2; --force overwrite verified by test_freeze_force_overwrites + CLI gate 3) |
| HARN-04 | Phase 4 | Complete (2026-05-05; Plan 04-01 commits 9588056 RED + 7881e87 GREEN; sidecar manifest with $schema_version + git_sha + git_dirty + frozen_at + per-tier provenance + library_versions for 4 critical libs verified by test_manifest_top_level_fields + test_manifest_per_tier_provenance + test_manifest_library_versions + test_manifest_judge_block + test_manifest_missing_tier + CLI gate 5) |
| HARN-05 | Phase 3 | Complete (2026-05-05; end-to-end delivery: Plan 03-01 NaNReasonTracer + _classify_post_evaluate_nan units via TDD red→green commits e97e864 + bc80825; Plan 03-02 wiring into score_query_log via callbacks=[tracer] + per-metric precedence chain in commit fe52528; Plan 03-03 live smoke backstop test_eval_smoke_nan_reasons asserting n_unknown_nan==0 against real Gemini 2.5 Flash output, verdict PASS in commit 512ad54) |
| CAP-01 | Phase 7 | Pending |
| CAP-02 | Phase 8 | Complete |
| CAP-03 | Phase 6 | Pending |
| DOC-01 | Phase 9 | Pending |
| DOC-02 | Phase 9 | Pending |
| DOC-03 | Phase 9 | Pending |
| DOC-04 | Phase 9 | Pending |
| DOC-05 | Phase 9 | Pending |

**Coverage:**
- v1.0 requirements: 16 total
- Mapped to phases: 16 ✓
- Unmapped: 0 ✓

**Mapping notes:**
- TIER-03 (smoke test) is intentionally split across Phases 1 and 2 because each tier fix needs its own smoke gate before committing to a full rerun. Each phase owns the smoke test for its tier; the requirement is satisfied only when both halves pass.
- Phase 9 carries all 5 DOC-* requirements because they are inseparable: the frozen doc is one artifact and the requirements are sub-features of that artifact.
- No requirement is duplicated across phases; no v1.0 requirement is orphaned.

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after roadmap creation (traceability filled in)*
