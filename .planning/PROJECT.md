# RAG Architecture Patterns

## What This Is

A workshop repo that produces measured cost / quality / latency numbers for 5 RAG architectures (Naive ChromaDB, Managed Gemini File Search, Graph LightRAG, Multimodal RAG-Anything, Agentic OpenAI Agents) over a shared 100-paper, 581-figure corpus. The numbers feed an external blog post that lives in a separate repo (`/Users/patrykattc/work/git/PatrykQuantumNomad/src/data/blog/rag-architecture-patterns.mdx`). Code exists in service of the blog, not the other way around.

## Core Value

Produce reproducible, honest numbers for the blog — every claim in the post must be backed by a captured run with full provenance (timestamp, model, git SHA, judge model).

## Requirements

### Validated

<!-- Inferred from existing code. Phase numbering in code comments / evaluation/results/comparison.md refers to the BLOG repo's planning, not this repo's. -->

- ✓ Curated 100-paper / 581-figure RAG corpus (LFS) — existing
- ✓ 30-question hand-authored golden Q&A (10 single-hop / 10 multi-hop / 10 multimodal) — existing
- ✓ Tiers 1–5 end-to-end ingest + query CLIs (`tier-{N}-{name}/main.py`) — existing
- ✓ Shared cost-tracking infrastructure (`CostTracker`, pricing table, JSON persistence) — existing
- ✓ RAGAS evaluation harness (faithfulness, answer_relevancy, context_precision) — existing
- ✓ Tier rollup comparison generator (`python -m evaluation.harness.compare`) — existing
- ✓ First-pass eval numbers for Tiers 1, 2, 3 captured 2026-05-02 — existing

### Active

**v1.0 — Ship eval numbers for the blog:**

- [ ] Fix Tier 5 `empty_contexts` adapter bug (`evaluation/harness/adapters/tier_5.py:125` hard-codes `retrieved_contexts=[]`)
- [ ] Regenerate clean Tier 4 graphml (wipe `rag_anything_storage/tier-4-multimodal/`; MineRU ingest must run outside the sandbox)
- [ ] Add freeze step + single-command pipeline driver (`evaluation/harness/freeze.py` + `pipeline.py`)
- [ ] Re-run all 5 tiers on the same date with consistent provenance (single eval-date for the blog)
- [ ] Multi-judge spot-check on 5 questions × 3 tiers with a different judge model (measure family-bias delta)
- [ ] Update `evaluation/results/comparison.md` with all 5 tiers fully populated
- [ ] Produce frozen handoff doc (`evaluation/results/frozen/eval-numbers-v1.0.md`) containing tier rollup, per-question-class breakdown, per-tier provenance, multi-judge bias delta, and honest disclaimers — copy-pasteable into the blog repo

### Out of Scope

- **Adding new tiers / new RAG architectures** — v1.0 is fix-and-ship the existing 5; the blog's narrative is built around them
- **Adding new questions to the golden set** — 30 is locked (single-hop / multi-hop / multimodal split is the blog's evidence structure)
- **Statistical-significance testing** — sample size (30 × 5) is too small; magnitude-only is the blog's framing
- **Consolidating the `tier-N-name/` ↔ `tier_N_name/` folder pairs** — the underscore folders are intentional setuptools shims (Python module names can't contain hyphens); listed as a v1.1+ cleanup candidate
- **Mitigating self-grading bias** (judge LLM = generation LLM family) — acknowledged in disclaimers, not engineered around. *Measurement* of bias via a multi-judge spot-check (5 questions × 3 tiers re-scored with a different judge model) IS in v1.0 scope; full mitigation (re-running everything with multiple judges) is not
- **Re-architecting evaluation harness, cost-tracker, or shared layer** — minimize new abstractions; v1.0 uses what exists

## Context

**Relationship to the blog repo:**
The blog post lives at `/Users/patrykattc/work/git/PatrykQuantumNomad/src/data/blog/rag-architecture-patterns.mdx` in a separate repo. This repo is the *workshop* — it produces a single artifact (`eval-numbers-v1.X.md`) that gets copy-pasted into the blog's evidence sections. Phase numbering in code comments and `evaluation/results/comparison.md` (e.g. "Phase 131", "Phase 133 BLOG-04", "Phase 138-139") refers to the *blog repo's* GSD planning. This repo's GSD planning starts fresh at Phase 1.

**Codebase already mapped:**
See `.planning/codebase/` for full architecture / stack / conventions / testing / concerns. Key facts:

- 5 tier directories (`tier-{N}-{name}/`) — each with isolated `requirements.txt` and a `main.py` CLI following the same `--ingest` / `--query` pattern
- Each tier has a sibling `tier_{N}_{name}/` **shim package** containing only `__init__.py` (Python-importable form because hyphens are illegal in module names; listed in `pyproject.toml:21`)
- Shared utilities in `shared/` (config, llm, embeddings, cost_tracker, pricing, display, loader)
- Evaluation harness in `evaluation/harness/`
- Storage isolated per tier (`chroma_db/tier-1-naive/`, `lightrag_storage/tier-3-graph/`, `rag_anything_storage/tier-4-multimodal/`)

**Current eval state (2026-05-04):**
- Last full run on 2026-05-02 captured Tiers 1, 2, 3 with real RAGAS numbers
- Tiers 4 and 5 produced 30/30 `empty_contexts` NaN
- Recent commits (`fb397f0`, `186a9c2`, `ce5c2ad`, `f0cd134`, `f801916`) fixed RAGAS 0.4.3 quirks, Tier 2 adapter Gemini SecretStr unwrap, and Tier 4 LightRAG initialization — these have not yet been re-evaluated
- Tier 5 root cause is unknown — investigation is part of v1.0 scope

**Beyond v1.0:**
More milestones are expected after the blog ships — rebuttal runs from blog feedback, additional tiers, or post-publication requests. v1.0 should not encumber future work: avoid restructuring, minimize new abstractions, keep changes targeted.

## Constraints

- **Tech stack**: Python 3.10+, established per existing `pyproject.toml` — no language switches.
- **Providers**: OpenRouter for Tier 1+ embeddings/chat; Gemini for legacy `shared.llm`. Don't introduce a new provider for v1.0.
- **Cost discipline**: A full RAGAS run is ~$0.20–0.50 per 30-question suite. Re-running all 5 tiers fresh costs ~$1–3 per run; budget for 2–3 full re-runs total (debug + final capture).
- **External tooling constraint**: Tier 4's MineRU PDF ingest must run outside the existing sandbox per Phase 139 evidence — graphml regeneration is the prereq for the Tier 4 eval run.
- **Reproducibility**: Every captured run must record timestamp + git SHA + judge model + generation model so blog citations are reproducible.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Workshop repo separate from blog repo | Blog repo has its own deploy / publish lifecycle; eval workspace shouldn't gate it | ✓ Good (already in place) |
| Same 30-question golden set across all tiers | Tier comparability requires identical prompts | ✓ Good |
| Re-run all 5 tiers fresh for v1.0 (vs. freeze 1–3) | Single eval-date provides cleaner blog story; cost is ~$1–3 | — Pending |
| Underscore tier folders are setuptools shims, not duplicates | Python module names can't contain hyphens; CLI uses hyphen form, Python imports use underscore form | ⚠️ Revisit in v1.1+ if cleanup is desired (out of scope for v1.0) |

---
*Last updated: 2026-05-04 after initialization*
