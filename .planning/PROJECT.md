# RAG Architecture Patterns

## What This Is

A workshop repo that produces measured cost / quality / latency numbers for 5 RAG architectures (Naive ChromaDB, Managed Gemini File Search, Graph LightRAG, Multimodal RAG-Anything, Agentic OpenAI Agents) over a shared 100-paper, 581-figure corpus. v1.0 shipped the immutable artifact `evaluation/results/frozen/eval-numbers-v1.0.md` (+ sidecar manifest) for copy-paste into the external blog post. Code exists in service of the blog, not the other way around.

## Core Value

Produce reproducible, honest numbers for the blog — every claim in the post must be backed by a captured run with full provenance (timestamp, model, git SHA, judge model, embedder).

## Requirements

### Validated

<!-- Cumulative record across milestones. Items marked v1.0 shipped 2026-05-08. -->

**Pre-v1.0 (already in place at init):**
- ✓ Curated 100-paper / 581-figure RAG corpus (LFS)
- ✓ 30-question hand-authored golden Q&A (10 single-hop / 10 multi-hop / 10 multimodal)
- ✓ Tiers 1–5 end-to-end ingest + query CLIs (`tier-{N}-{name}/main.py`)
- ✓ Shared cost-tracking infrastructure (`CostTracker`, pricing table, JSON persistence)
- ✓ RAGAS evaluation harness (faithfulness, answer_relevancy, context_precision)
- ✓ Tier rollup comparison generator (`python -m evaluation.harness.compare`)

**Shipped in v1.0 (2026-05-08):**
- ✓ Tier 5 adapter walks `RunResult.new_items` for `ToolCallOutputItem.output` (TIER-01) — v1.0
- ✓ Tier 4 graphml rebuilt from MineRU JSON parsed on host (28,597 nodes / 80,419 edges from 79 papers) (TIER-02) — v1.0
- ✓ Per-tier 5-question smoke gate before full rerun (TIER-03) — v1.0
- ✓ Single-command pipeline driver `pipeline.py` with single-SHA invariant (HARN-01) — v1.0
- ✓ Single-tier rerun preserves other tiers via `_latest()` mtime resolution (HARN-02) — v1.0
- ✓ Freeze tool with refuse-to-clobber + sidecar manifest (HARN-03, HARN-04) — v1.0
- ✓ Per-row `nan_reason` distinguishing 4 pre-evaluate + 6 post-evaluate reasons (HARN-05) — v1.0
- ✓ Full 5-tier × 30-question RAGAS sweep, single date, single git SHA `75f6f1b` (CAP-01) — v1.0 (T4 NaN 2/30, T5 NaN 0/30; $0.439 spend vs $3.00 ceiling)
- ✓ Multi-judge spot-check (Claude Haiku 4.5 secondary, 15 cells, structured delta JSON) (CAP-02) — v1.0 ($0.12225 spend; per-tier ΔF T1 -0.035, T4 +0.010, T5 -0.164)
- ✓ Per-tier embedder provenance recorded in capture JSON (CAP-03) — v1.0 (D-ROADMAP-OVERRIDE: T5 same embedder as T1)
- ✓ Frozen handoff doc `eval-numbers-v1.0.md` with rollup, per-class breakdown, provenance, embedder table, multi-judge delta, honest disclaimers (DOC-01..DOC-05) — v1.0

### Active

No milestone currently scoped. Backlog candidates listed below (carried over from `milestones/v1.0-REQUIREMENTS.md` "v1.1 Requirements" section). Run `/gsd:new-milestone` to scope and prioritize.

**Blog feedback / publication response (highest leverage):**
- [ ] BLOG-01: Rebuttal run pipeline — invoked when blog feedback questions a specific tier's numbers; produces a dated `eval-numbers-vX.Y.md` re-run for that tier only

**Methodology depth:**
- [ ] METH-01: Bootstrap 95% CI on per-tier rollup means (~10 LOC numpy in `compare.py`) for blog-feedback rebuttals
- [ ] METH-02: Cross-judge full re-scoring (all 30 × 5 with second judge model) — full bias mitigation, not just measurement
- [ ] METH-03: Per-tier per-class min/max/stdev added to rollup (currently mean-only)

**Tooling hardening (surfaced during v1.0 execution):**
- [ ] TOOL-01: Add `addopts = "-m 'not live'"` to `pyproject.toml [tool.pytest.ini_options]` so bare pytest can't accidentally consume API budget
- [ ] TOOL-02: Augment `score.py` token_usage_parser to parse usage from LiteLLM ModelResponse bodies — currently RAGAS judge cost ledgers report $0 despite real spend
- [ ] TOOL-03: `tracker.persist()` should honor caller-supplied `--results-dir` (run.py:273, score.py:518)
- [ ] TOOL-04: `tier-4-multimodal/scripts/eval_capture.py` should prepend `Path(sys.executable).parent` to `os.environ['PATH']` so mineru parser-installation probe finds `.venv/bin/mineru`
- [ ] TOOL-05: RAGAS 0.4.3 → 0.4.x+ migration for `from ragas.metrics import context_precision` DeprecationWarning
- [ ] TOOL-06 (cosmetic): `compare.py` emit `## Embedder by tier` (heading) instead of `**Embedder by tier:**` (bold)
- [ ] TOOL-07: Cost-window selection in verifier scripts — prefer ts-window grep over `ls -t | head -1`

**Cleanup:**
- [ ] CLEANUP-01: Consolidate `tier-{N}-{name}/` ↔ `tier_{N}_{name}/` folder pairs into a single layout

### Out of Scope

- **Adding new tiers / new RAG architectures** — the blog's narrative is built around 5 tiers; new architectures would be a v2.0 scope decision
- **Adding new questions to the golden set** — 30 is locked (single-hop / multi-hop / multimodal split is the blog's evidence structure); new questions invalidate v1.0 baselines
- **Statistical-significance testing** — sample size (30 × 5) is too small; magnitude-only is the blog's framing. METH-01 (bootstrap CI) provides partial mitigation in v1.1.
- **Re-architecting evaluation harness, cost-tracker, or shared layer** — minimize new abstractions; new modules only when a clear seam emerges
- **Switching providers (e.g. moving Tier 1 off OpenRouter)** — locked through v1.x; provider switch is a v2.0 scope decision
- **Cross-language ports of any tier** — Python-only; locked by `pyproject.toml`

## Context

**v1.0 shipped 2026-05-08:**
- Codebase: 12,206 Python LOC across `evaluation/harness/` + `shared/` + 5 tier modules
- New v1.0 harness scaffolding: 931 LOC (`pipeline.py` 189, `multi_judge_spotcheck.py` 227, `freeze.py` 97 (at hard cap), `smoke_gate.py` 298, `diagnostics.py` 120)
- Tech stack: Python 3.10+, RAGAS 0.4.3, lightrag-hku 1.4.15, raganything 1.2.10, openai-agents 0.14.6, mineru 3.1.4, ChromaDB, OpenRouter (Tier 1+ embeddings/chat), Gemini (legacy `shared.llm`)
- Frozen artifact: `evaluation/results/frozen/eval-numbers-v1.0.md` (5582 bytes) + `eval-numbers-v1.0.manifest.json` (3701 bytes, git_sha=`b7564f6`, git_dirty=false)
- Full sweep date: 2026-05-07, sweep_sha=`75f6f1b`, total spend $0.439 (14.6% of $3.00 ceiling)
- Multi-judge spot-check date: 2026-05-07, secondary judge Claude Haiku 4.5, $0.12225 spend
- Tier 4 graph: 28,597 nodes / 80,419 edges from 79 papers (rebuilt on host 2026-05-07; 13h55m wall, $24.85)

**Relationship to the blog repo:**
The blog post lives at `/Users/patrykattc/work/git/PatrykQuantumNomad/src/data/blog/rag-architecture-patterns.mdx` in a separate repo. This repo produced the immutable artifact `eval-numbers-v1.0.md` that gets copy-pasted into the blog's evidence sections. Phase numbering in code comments and `evaluation/results/comparison.md` (e.g. "Phase 131", "Phase 133 BLOG-04", "Phase 138-139") refers to the *blog repo's* GSD planning. This repo's GSD planning starts fresh at Phase 1 and ran 1–9 for v1.0.

**Codebase architecture:**
See `.planning/codebase/` for full architecture / stack / conventions / testing / concerns. Key facts:
- 5 tier directories (`tier-{N}-{name}/`) — each with isolated `requirements.txt` and a `main.py` CLI following the same `--ingest` / `--query` pattern
- Each tier has a sibling `tier_{N}_{name}/` shim package containing only `__init__.py` (Python-importable form because hyphens are illegal in module names; CLEANUP-01 candidate for v1.1)
- Shared utilities in `shared/` (config, llm, embeddings, cost_tracker, pricing, display, loader)
- Evaluation harness in `evaluation/harness/`
- Storage isolated per tier (`chroma_db/tier-1-naive/`, `lightrag_storage/tier-3-graph/`, `rag_anything_storage/tier-4-multimodal/`)

**Known issues / technical debt (carried into v1.1):**
- LiteLLM token-usage parser misses OpenRouter responses → all `ragas-judge-*.json` cost ledgers report $0 despite real spend (~$0.45/sweep). Mitigated in P8 via `_estimate_cost_fallback`. Root fix is TOOL-02.
- `tracker.persist()` ignores caller's `--results-dir` (TOOL-03)
- 5 zero-byte placeholder cost JSONs from post-sweep pytest test bleed-through pollute `evaluation/results/costs/`
- Eval result directories (`queries/`, `metrics/`, `costs/`) are gitignored as "regenerable runtime intermediates"; provenance lives in SUMMARY commits + the frozen v1.0 manifest

**Beyond v1.0:**
More milestones are expected after the blog ships — rebuttal runs from blog feedback, additional tiers, or post-publication requests. v1.0 should not encumber future work: v1.0 minimized restructuring and abstractions per design.

## Constraints

- **Tech stack**: Python 3.10+, established per `pyproject.toml` — no language switches.
- **Providers**: OpenRouter for Tier 1+ embeddings/chat; Gemini for legacy `shared.llm`. Don't introduce a new provider for v1.x without explicit decision.
- **Cost discipline**: A full RAGAS run is ~$0.20–0.50 per 30-question suite (excluding Tier 4 host rebuild). v1.0 cumulative cost ~$25.32 (dominated by one-time $24.85 Tier 4 host rebuild). Steady-state sweep cost $0.439. Multi-judge spot-check $0.122.
- **External tooling constraint**: Tier 4's MineRU PDF ingest must run outside the existing sandbox per Phase 139 evidence. v1.0 ran the host rebuild manually for Phase 7 Plan 07-02; v1.1+ should preserve this constraint.
- **Reproducibility**: Every captured run must record timestamp + git SHA + judge model + generation model + embedder so blog citations are reproducible. The frozen v1.0 manifest is the canonical example.
- **Forward-contract RAW-LOCK**: 6 harness modules (`pipeline.py` / `run.py` / `score.py` / `compare.py` / `freeze.py` / `smoke_gate.py`) are byte-identical between phases except in their owning phase. v1.1+ should preserve this discipline.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Workshop repo separate from blog repo | Blog repo has its own deploy/publish lifecycle; eval workspace shouldn't gate it | ✓ Good (already in place at v1.0 init) |
| Same 30-question golden set across all tiers | Tier comparability requires identical prompts | ✓ Good |
| Re-run all 5 tiers fresh for v1.0 (vs. freeze 1–3) | Single eval-date provides cleaner blog story | ✓ Good ($0.439 actual sweep cost; cleaner provenance than incremental freeze) |
| Tier 5 adapter reads `item.output` not `item.raw_item` | `raw_item` is a stringified Responses-API payload (Pitfall 1 of 132-RESEARCH HARD invariant) | ✓ Good (smoke 5/5 + full sweep 0/30 NaN) |
| Tier 5 contexts deduped first-occurrence-wins on `(paper_id, page)` for chunks and `(paper_id)` for metadata | Avoids agent loop blow-up while preserving signal | ✓ Good |
| `[debug-tier5]` extra opt-in only | Default `uv sync` does NOT install OpenInference / Phoenix / OpenTelemetry | ✓ Good |
| Phase 2 Plan 02-01 ran smoke-only (3 of 75 papers) | Measured paper-1 wall ~21 min projected ~15–25h vs plan budget; Phase 7 was architecturally correct place for full ingest | ✓ Good (Phase 7 Plan 07-02 completed full 79-paper rebuild as planned) |
| `JUDGE_MAX_TOKENS=8192` in `score._build_judge` | Gemini 1024-token output cap was truncating RAGAS faithfulness `_create_statements` on long Tier 4 hybrid-mode answers | ✓ Good (faithfulness 4/5 NaN → 5/5 = 1.0 on smoke; full-sweep T4 NaN 2/30) |
| HARN-05 idempotence: 'first wins' (leaf-most error) | RAGAS_PROMPT is the leaf where exceptions originate; specific signal beats general | ✓ Good (live smoke `unknown_nan=0` against real Gemini) |
| Phase 4 freeze refuse-to-clobber default; `--force` overwrites BOTH md AND manifest | Frozen docs are immutable once written; explicit override required | ✓ Good |
| Phase 5 in-process composition (not subprocesses) | Exit codes propagate, asyncio loops reused, single SHA captured at amain entry | ✓ Good (live test PASS at $0.007) |
| D-ROADMAP-OVERRIDE Phase 6: T5 embedder is `openai/text-embedding-3-small` via openrouter (same as T1), NOT "OpenAI hosted vector-store embedder" | Stale ROADMAP wording contradicted by `tier-5-agentic/tools.py:47-50,90-101` direct verification | ✓ Good (locked in 3 places: ROADMAP inline + plan frontmatter + source comment) |
| Phase 6 `managed` derived as `embedder_source == "google-managed"` | Only Tier 2 is "managed" (Google File Search owns indexing); rejected boolean `managed=true` flag | ✓ Good |
| Phase 8 BLOCKER #3 fix: `_read_source_sha` reads `src_log.git_sha`, NEVER `_git_sha()` | Multi-judge re-scores against a PINNED Phase 7 capture; current HEAD irrelevant | ✓ Good (dual-SHA `source_capture_git_sha=75f6f1b ≠ HEAD 3f37e4b` verified live) |
| Phase 8 secondary judge `anthropic/claude-haiku-4.5 max_tokens=8192` | Non-Gemini family for genuine bias measurement; Haiku 4.5 cheap enough to fit envelope | ✓ Good (clean PASS at $0.12225 = 40.75% of $0.30 envelope) |
| Phase 8 fallback cost estimator when LiteLLM usage parser returns 0 | Pre-existing token-parser gap on OpenRouter; honest accounting > silent $0 | ✓ Good |
| RAW-LOCK forward-contract enforced across all 9 phases | Six harness modules byte-identical end-to-end except in owning phase | ✓ Good (verified at every phase boundary; 0 bytes diff) |
| Underscore tier folders are setuptools shims, not duplicates | Python module names can't contain hyphens | ⚠️ Revisit in v1.1 (CLEANUP-01) — defer until clear seam emerges |

---
*Last updated: 2026-05-09 after v1.0 milestone completion*
