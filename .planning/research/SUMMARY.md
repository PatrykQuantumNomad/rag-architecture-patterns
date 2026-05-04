# Project Research Summary

**Project:** RAG Architecture Patterns — 5-Tier Eval v1.0 (Frozen Blog Handoff)
**Domain:** Fix-and-freeze milestone: repair Tier 4 graphml + Tier 5 empty_contexts, produce a credible frozen numbers doc for an external blog post
**Researched:** 2026-05-04
**Confidence:** HIGH (stack and architecture HIGH; features and pitfalls MEDIUM-HIGH)

## Executive Summary

This is a subsequent-milestone project, not greenfield. The five RAG tiers (Naive ChromaDB, Gemini File Search, LightRAG Graph, RAG-Anything Multimodal, OpenAI Agents) are already built and partially evaluated. The v1.0 goal is narrow: fix two blocking bugs, re-run all five tiers on a single evaluation date, and produce a frozen markdown handoff doc (`eval-numbers-v1.0.md`) that is copy-pasteable into an external blog repo with full provenance and honest disclaimers. The harness infrastructure from Phase 131 is correct and complete; the two NEW pieces needed (freeze.py + pipeline.py) are approximately 200 LOC total.

The two ship-blockers are both well-understood. Tier 5's 30/30 NaN is caused by a five-line adapter bug at `evaluation/harness/adapters/tier_5.py:125` that hard-codes `retrieved_contexts=[]` instead of walking `RunResult.new_items` for `ToolCallOutputItem` outputs. Tier 4's corrupt graphml requires wiping `rag_anything_storage/tier-4-multimodal/` entirely (not moving it) and running the MinerU CLI outside the eval sandbox before re-ingesting. Both fixes are surgical and touch no shared infrastructure.

The primary ongoing risk after the two bugs are fixed is credibility of the frozen numbers doc rather than correctness of the code. Self-grading bias (Gemini judging Gemini-generated outputs for Tiers 2-4) is the most likely hostile-reader critique. The mitigation for v1.0 is disclosure — specific, sourced language citing Panickssery 2024 and Wataoka 2024 — not engineering mitigation, which is explicitly out of scope per PROJECT.md. A multi-judge spot-check (re-grade five questions with a non-Gemini judge) is a P2 candidate worth negotiating into v1.0; it takes one afternoon and approximately $0.10-0.30.

## Key Findings

### Recommended Stack

The existing pinned stack in `pyproject.toml` is sound and must not change. `lightrag-hku==1.4.15`, `raganything==1.2.10`, `openai-agents[litellm]==0.14.6`, and `ragas>=0.4.3,<0.5` are all locked to validated behavior; bumping any of them risks index-format breaks, RAGAS quirk regressions, or SDK API changes that would invalidate recent patches.

Two additions are needed. For Tier 4, `mineru[core]>=2.5,<4` must be installed on the host machine (not inside the sandbox) so the MinerU CLI is available for the outside-sandbox ingest pass. For Tier 5 debugging, an optional `debug-tier5` extras group with `openinference-instrumentation-openai-agents==1.4.2` and `arize-phoenix>=6,<8` enables OTel tracing to confirm whether the agent is calling tools at all — relevant only if the five-line adapter fix does not resolve all NaNs.

**Core technologies (keep pinned, do not bump):**
- `lightrag-hku==1.4.15`: KG store for Tiers 3/4; format would break on any version change
- `raganything==1.2.10`: Multimodal wrapper for Tier 4; locked invariants for EMBED_DIMS=1536
- `openai-agents[litellm]==0.14.6`: Tier 5 agent runtime; `RunResult.new_items` / `ToolCallOutputItem` API stable in 0.14.x
- `ragas>=0.4.3,<0.5`: RAGAS scoring; existing patches in `fb397f0`/`186a9c2`/`fb397f0` are 0.4.3-specific

**New additions (scoped and conditional):**
- `mineru[core]>=2.5,<4`: Tier 4 host-only ingest; use `mineru[core]`, never bare `pip install mineru`
- `openinference-instrumentation-openai-agents==1.4.2` + `arize-phoenix>=6`: dev-only debug group; only if Tier 5 adapter fix leaves residual NaNs

### Expected Features

The v1.0 deliverable is a single frozen markdown file, not a new user-facing product feature. Every P1 feature in the frozen doc is already scaffolded in the auto-generated `comparison.md`; the task is to lock it down, reorganize for blog readability, and add a small number of missing items.

**Must have (table stakes — without these the doc is dismissed):**
- Versioned, immutable filename (`eval-numbers-v1.0.md`) with version in both filename and H1 header
- Single eval-date for all 5 tiers stated prominently in the header block
- Per-tier provenance row: timestamp, git SHA, generation model+version, judge model+version, judge embedder
- Judge LLM + embedder named in the first 10 lines, not buried
- Self-grading/family-bias disclaimer citing Panickssery 2024 and Wataoka 2024
- NaN breakdown distinguishing `empty_contexts` from `empty_statements` from `json_parse_failure`
- Multimodal scores for Tiers 1-3 visually de-emphasized (em-dash or "for reference only" sub-table)
- Reproducibility footer: exact regeneration command, dataset manifest hash, repo URL, repo commit SHA
- Sample-size disclaimer ("n=30, magnitudes not p-values") in lead position above the rollup table

**Should have (P2 — negotiate into v1.0):**
- Multi-judge spot-check: re-grade 5 questions across 3 tiers with a non-Gemini judge (Claude Haiku or GPT-4.1-mini); turns acknowledgement of bias into bounded evidence; PROJECT.md marks mitigation out of scope but measurement is different from mitigation
- Per-tier links to raw artifacts (cost JSON, metrics JSON) via relative repo paths
- Hardware/sandbox note for Tier 4 (MinerU must run outside sandbox)

**Defer to v1.1+:**
- Bootstrap 95% CI per metric (~10 LOC numpy change in compare.py)
- Latency percentiles P50/P90/P95 alongside mean
- Per-version changelog (starts at v1.1)
- Per-question min/max/stdev

### Architecture Approach

The five-component pipeline (Capture -> Score -> Rollup -> Summarize -> Freeze) is the standard pattern used by lm-evaluation-harness, HELM, and Inspect AI. Phases 1-4 are fully implemented in Phase 131 (`run.py`, `score.py`, `compare.py`). Phase 5 (Freeze) is the only new engineering work: a `freeze.py` (~60 LOC) that copies `comparison.md` to `evaluation/results/frozen/eval-numbers-v{x.y}.md` and writes a sidecar manifest with source paths and git SHA, plus a `pipeline.py` (~100 LOC) that orchestrates the existing stages with a single `--freeze vX.Y` flag. Capture/score/compare remain separate phases with separate cost ledgers — this is load-bearing for partial reruns (re-run Tier 4 only without re-paying ~$0.60 in Tier 3 LLM costs).

**Major components:**
1. `evaluation/harness/adapters/tier_5.py` (MODIFY) — walk `RunResult.new_items` for `ToolCallOutputItem.output` instead of hard-coding `retrieved_contexts=[]`
2. `rag_anything_storage/tier-4-multimodal/` (DELETE + REBUILD) — wipe entirely; MinerU CLI run outside sandbox produces clean graphml
3. `evaluation/harness/freeze.py` (NEW ~60 LOC) — copy + manifest; refuses to clobber existing version
4. `evaluation/harness/pipeline.py` (NEW ~100 LOC) — single-command driver over existing run/score/compare + new freeze
5. `evaluation/results/frozen/eval-numbers-v1.0.md` (NEW) — the blog-citable frozen artifact; committed and immutable

### Critical Pitfalls

All four researchers independently surfaced the same two ship-blockers. The remaining pitfalls are disclosure/documentation tasks, not engineering work.

1. **Tier 5 `retrieved_contexts=[]` hard-coded (SHIP-BLOCKER)** — `evaluation/harness/adapters/tier_5.py:125` hard-codes `retrieved_contexts=[]`; the 30/30 all-or-nothing NaN confirms this is the root cause. Fix: import `ToolCallOutputItem` from `agents.items`, walk `result.new_items` after `Runner.run()`, collect `item.output` for each `ToolCallOutputItem`. If contexts remain empty after the fix, use OpenInference instrumentation to confirm whether the agent is calling tools — known LiteLLM+Gemini `MALFORMED_FUNCTION_CALL` silent failure is the next suspect (decision tree: simplify schema -> switch model slug -> bump openai-agents).

2. **Tier 4 graphml regen requires clean wipe (SHIP-BLOCKER)** — `rag_anything_storage/tier-4-multimodal/` must be deleted with `rm -rf`, not moved or archived. Run MinerU CLI outside the sandbox (`mineru -p dataset/papers/ -o tier-4-multimodal/output/mineru-raw/ -m auto -b pipeline`), then re-ingest. Verify graphml is non-empty before running eval_capture. Log node/edge counts in provenance.

3. **Self-grading family bias inflates Tiers 2-4 (DISCLOSE-AND-SHIP)** — Judge is `google/gemini-2.5-flash`; Tiers 2-4 generate with Gemini-family models. Published research documents systematic 5-15% score inflation for same-family outputs. Mitigation for v1.0 is disclosure only; the P2 multi-judge spot-check turns acknowledgement into bounded evidence.

4. **RAGAS NaN reason opacity (SHIP-BLOCKER)** — Two distinct failure modes both produce NaN in the faithfulness column: `empty_contexts` vs `empty_statements` (claim decomposition failed). Adding a `nan_reason` breakdown is ~30 minutes of harness work and is mandatory before the frozen doc ships.

5. **Provenance SHA drift (DISCLOSE-AND-SHIP)** — The `freeze.py` manifest solves this mechanically by calling `git rev-parse HEAD` at freeze time. The frozen doc must never hand-edit provenance fields.

## Implications for Roadmap

Based on combined research, the v1.0 milestone maps to six sub-phases. Phases A and B run in parallel; C through F are sequential.

### Phase A: Fix Tier 5 Adapter (SHIP-BLOCKER)
**Rationale:** Highest-leverage single change in the milestone. Five lines in one file; no infrastructure changes. Smoke-test 3-5 questions before committing full rerun budget.
**Delivers:** Tier 5 `retrieved_contexts` populated from `ToolCallOutputItem.output`
**Addresses:** Pitfall 1 (empty_contexts hard-coded), Pitfall 7 (tool routing)
**Avoids:** Paying $0.30-0.50 for a full Tier 5 rerun before confirming the fix works

### Phase B: Tier 4 Graphml Regen (SHIP-BLOCKER, parallel with A)
**Rationale:** Independent of Tier 5; MinerU ingest is the long-running step (~50 min CPU for 100 PDFs); start it early.
**Delivers:** Clean `graph_chunk_entity_relation.graphml` with verified node/edge counts; storage rebuilt from scratch
**Addresses:** Pitfall 2 (partial graph state)
**Avoids:** Non-reproducible Tier 4 numbers; must use `rm -rf`, not `mv`

### Phase C: Harness Freeze + Pipeline (~200 LOC new code)
**Rationale:** The freeze mechanism must exist before the final sweep; test against existing comparison.md before spending the full rerun budget.
**Delivers:** `evaluation/harness/freeze.py` (copy + manifest + SHA capture) and `evaluation/harness/pipeline.py` (single-command driver); `evaluation/results/frozen/` directory
**Uses:** HELM --suite semantics (Pattern 1), Inspect AI eval_set orchestration (Pattern 2), most-recent-by-mtime aggregation (Pattern 3 — already in compare.py)
**Addresses:** Pitfall 8 (SHA drift); Architecture anti-patterns 2 and 4

### Phase D: Full Five-Tier Rerun (single eval-date)
**Rationale:** Gates on Phases A+B smoke tests. This is the single most expensive step ($1-3); run once only after both fixes are verified. Use `--tiers 4 --tier-4-from-cache` for partial rerun if Tier 4 needs a second pass.
**Delivers:** Fresh queries/costs/metrics JSON for all 5 tiers under a single git SHA on a single date
**Addresses:** Single-eval-date gating constraint; Pitfall 8 SHA consistency

### Phase E: Freeze v1.0 Document
**Rationale:** Immediately after Phase D (same session if possible). Upgrade comparison.md format to hit all P1 features before freezing.
**Delivers:** `evaluation/results/frozen/eval-numbers-v1.0.md` with all P1 features: versioned header, single eval-date, per-tier provenance, NaN breakdown by reason, self-grading disclosure, multimodal visual de-emphasis, sample-size disclaimer, reproducibility footer
**Addresses:** All DISCLOSE-AND-SHIP pitfalls (3, 5, 6, 8, 9, 10)

### Phase F: User Decision — Multi-Judge Spot-Check (P2, negotiate)
**Rationale:** PROJECT.md scopes bias mitigation out of scope, but a 5-question spot-check with a non-Gemini judge is measurement, not mitigation. One afternoon; ~$0.10-0.30. Roadmapper must surface this as a user decision: include in v1.0 or defer to v1.1.
**Delivers:** Bias-delta row in the frozen doc's disclaimers section; `--judge` override flag in evaluation harness
**Addresses:** FEATURES.md P2 multi-judge spot-check; highest-leverage credibility lift per features research

### Phase Ordering Rationale

- A and B run in parallel: they touch completely different tiers and codebases
- C before D: test the freeze mechanism against the stale comparison.md; cheaper than discovering a freeze bug after a $2 rerun
- D after A+B smoke tests: the full rerun budget allows 2-3 runs total; don't spend one before fixes are confirmed
- E immediately after D (same session): ensures the eval-date is truly single; do not let comparison.md sit unfrozen overnight
- F is a decision point that must be resolved before E completes; if deferred, document in v1.1 backlog in the frozen doc itself

### Research Flags

Phases needing attention during implementation:
- **Phase A (Tier 5 adapter fix):** If smoke test after the five-line fix still shows empty contexts, follow STACK.md's decision tree in order: (1) simplify `@function_tool` schema, (2) switch model slug, (3) bump openai-agents. Do not skip steps.
- **Phase B (MinerU ingest):** MinerU model cache (~3-5GB) requires first-run download; verify `mineru --help` succeeds and model cache is present before starting the 100-PDF ingest. Set `MINERU_PDF_RENDER_THREADS=4` on CPU-only host to avoid OOM.

Phases with standard, well-documented patterns (no additional research needed):
- **Phase C (freeze.py + pipeline.py):** ARCHITECTURE.md provides complete pseudocode for both files; pattern is a copy + manifest sidecar.
- **Phase D (full rerun):** `python -m evaluation.harness.pipeline --tiers 1,2,3,4,5 --freeze v1.0 --yes` once Phase C exists.
- **Phase E (document authorship):** All disclosure language is already drafted as copy-paste templates in PITFALLS.md Pitfalls 3, 5, 6, 10 disclosure sections.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Context7-verified for openai-agents, lightrag, openinference; official MinerU docs checked; PyPI release dates verified 2026-04-28/29 |
| Features | MEDIUM-HIGH | Academic precedent (Model Cards, BenchmarkCards) HIGH; mapping to blog-handoff convention required synthesis at MEDIUM |
| Architecture | HIGH | Four reference harnesses (lm-eval, HELM, Inspect AI, bigcode) converge on same five-component pattern; internal boundary analysis verified against existing code |
| Pitfalls | MEDIUM-HIGH | RAGAS/openai-agents findings cross-verified via GitHub issues and official docs; LightRAG/RAG-Anything ship-state findings rest on fewer sources |

**Overall confidence:** HIGH for the two ship-blockers and the freeze architecture; MEDIUM-HIGH for documentation/disclosure recommendations.

### Gaps to Address

- **Tier 5 root cause confirmation:** The five-line adapter fix is high-confidence as the primary cause. If contexts remain empty after the fix, the LiteLLM+Gemini compat path is MEDIUM-confidence. The smoke test after Phase A resolves this before committing the full rerun budget.
- **MinerU CPU timing:** Expected ~30-60s per PDF on a modern laptop CPU (community-reported; not benchmarked). Allow up to 90 min buffer for 100-PDF ingest.
- **Multi-judge spot-check scope:** Explicitly a user decision. FEATURES.md rates it "recommend negotiate into v1.0"; PROJECT.md says mitigation is out of scope. Roadmapper must surface this choice to the user before Phase E begins.
- **Model version pin for provenance:** OpenRouter passthrough may obscure the exact Gemini model version (`gemini-2.5-flash-001` vs `gemini-2.5-flash`). If unresolvable, document as "version unknown as of $eval_date" rather than leaving the field blank.

## Sources

### Primary (HIGH confidence)
- Context7 `/websites/openai_github_io_openai-agents-python` — `RunResult.new_items`, `ToolCallOutputItem`, `add_trace_processor`, `set_tracing_disabled`
- Context7 `/hkuds/lightrag` — `initialize_storages`, `NetworkXStorage` graphml layout, `JsonDocStatusStorage`, working_dir isolation
- Context7 `/arize-ai/openinference` — `OpenAIAgentsInstrumentor` install and tracer-provider wiring
- `https://opendatalab.github.io/MinerU/usage/cli_tools/` — full CLI flag list, env vars
- `https://github.com/stanford-crfm/helm` — HELM `--suite` freeze semantics, per-instance vs aggregate artifact split
- `https://inspect.aisi.org.uk/eval-sets.html` — Inspect AI eval_set() orchestration, log directory as durable completion record
- `https://github.com/EleutherAI/lm-evaluation-harness` — `--predict_only`, `--log_samples`, results/samples artifact split
- Model Cards — Mitchell et al. 2018, arXiv 1810.03993
- BenchmarkCards — arXiv 2410.12974
- RAGAS paper — arXiv 2309.15217

### Secondary (MEDIUM confidence)
- Self-Preference Bias in LLM-as-a-Judge — Wataoka et al. 2024, arXiv 2410.21819
- Play Favorites: Statistical Method to Measure Self-Bias — arXiv 2508.06709
- LLM Evaluators Recognize and Favor Their Own Generations — Panickssery et al. NAACL 2024
- RAGAS GitHub issues #1651, #1150, #2073 — NaN failure modes (empty_statements vs empty_contexts)
- openai/openai-agents-python GitHub issues #2257, #1575 — Gemini structured output + tools incompatibility
- BerriAI/litellm GitHub issue #16651 — `MALFORMED_FUNCTION_CALL` silent failures on gemini-2.5-flash

### Tertiary (LOW confidence — breadcrumbs only, not cited as fact)
- OpenAI community forum threads re: vector store status race conditions — context for Tier 5 diagnosis backstop only
- MinerU CPU timing estimates (~30-60s/PDF) — community-reported, not benchmarked

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
