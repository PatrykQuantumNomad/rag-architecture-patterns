# Milestones: RAG Architecture Patterns

Historical record of shipped versions. Most recent first.

---

## v1.0 Eval Handoff (Shipped: 2026-05-08)

**Delivered:** Reproducible, honest cost / quality / latency numbers for 5 RAG architectures over a shared 100-paper / 581-figure corpus, packaged as a single immutable markdown artifact (`eval-numbers-v1.0.md` + sidecar manifest) ready to copy-paste into the external blog post.

**Phases completed:** 1–9 (21 plans total)

**Key accomplishments:**

- Fixed both ship-blocker NaN regressions: Tier 5 `empty_contexts` 30/30 → 0/30 (walked `RunResult.new_items` for `ToolCallOutputItem.output`) and Tier 4 graphml 30/30 → 2/30 (host-rebuilt to 28,597 nodes / 80,419 edges from 79 papers).
- Built single-command pipeline driver (`pipeline.py`) that captures one git SHA at start and propagates it across capture → score → compare → freeze; verified live on Tier 5 smoke at $0.007.
- Captured the full 5-tier × 30-question RAGAS sweep on a single date (sweep_sha=`75f6f1b`, 2026-05-07) with $0.439 spend (14.6% of $3.00 ceiling).
- Measured self-grading bias via Claude Haiku 4.5 multi-judge spot-check: 15 cells (5 IDs × 3 tiers), per-tier ΔF T1 -0.035 / T4 +0.010 / T5 -0.164, $0.122 spend; cited inline in the frozen doc disclosures.
- Instrumented per-row `nan_reason` (4 pre-evaluate + 6 post-evaluate reasons including `unknown_nan` safety net) and per-tier `embedder` + `embedder_source` fields — every NaN and every embedder claim in the frozen doc is data-backed, not narrative.
- Shipped `eval-numbers-v1.0.md` (5582 bytes) + `eval-numbers-v1.0.manifest.json` (3701 bytes, git_sha=`b7564f6`, git_dirty=false) under refuse-to-clobber semantics — frozen doc is immutable once written.

**Stats:**

- 100 files created/modified
- 23,950 insertions across the v1.0 git range
- Total Python LOC: 12,206 (931 LOC new harness scaffolding: `pipeline.py` 189 + `multi_judge_spotcheck.py` 227 + `freeze.py` 97 + `smoke_gate.py` 298 + `diagnostics.py` 120)
- 9 phases, 21 plans, ~56 tasks
- 5 days execution (2026-05-04 → 2026-05-08), milestone closure 2026-05-09

**Cost (cumulative across phases):** ~$25.32, dominated by $24.85 one-time Tier 4 host rebuild (79 papers, 13h55m unattended). Final sweep: $0.439. Multi-judge spot-check: $0.122. Smoke gates and pre-flight: ~$0.07.

**Git range:** `baaa573` (feat(01-01): walk ToolCallOutputItems for retrieved_contexts) → `80f8467` (docs(09): align validation and summaries)

**Tag:** `v1.0`

**What's next:** v1.1+ is responsive to blog feedback — rebuttal runs, additional tiers post-publication, or v1.1 hardening (LiteLLM judge token-parser fix, `tracker.persist()` results-dir plumb-through, full cross-judge re-scoring METH-02). No v1.1 commitment yet.

---
