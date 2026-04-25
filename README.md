# RAG Architecture Patterns

Indexing the RAG literature itself to compare 5 RAG architectures.

## Blog Post

This repo is the companion to the blog post **[RAG Architecture Patterns: A 5-Tier Evaluation](https://patrykgolabek.dev/blog/rag-architecture-patterns)** (coming soon).

The post walks through the design, tradeoffs, and measured cost/quality of each tier against a single shared corpus — the meta-recursive twist being that the corpus IS the RAG literature. Every architecture indexes the papers that taught us how to build it.

## Architecture Tiers

| Tier | Name | One-line description |
|------|------|----------------------|
| 1 | Naive RAG (ChromaDB) | Embed-and-retrieve baseline using `chromadb` + `google-genai`. Cheapest, most code, hardest to tune. |
| 2 | Managed File Search (Gemini) | Vendor-managed retrieval via Gemini File Search. Zero retrieval code, opaque index. |
| 3 | Graph RAG (LightRAG) | Knowledge-graph RAG with entity/relationship extraction; multi-hop queries become graph walks. |
| 4 | Multimodal RAG (RAG-Anything) | Heterogeneous corpus: PDFs + extracted figures (+ video transcripts when available) in one pipeline. |
| 5 | Agentic RAG (OpenAI Agents) | Agents SDK orchestrating `FileSearchTool` + `@function_tool` over the same shared corpus. |

Each tier lives in its own `tier-{N}-{name}/` directory with an isolated `requirements.txt`. The shared utilities and dataset are common to all five. Phases 128-130 implement these tiers; this phase (127) ships the corpus and shared scaffolding.

## Dataset

A curated citation cluster anchored on the RAG literature itself (meta-recursive — we demonstrate RAG architectures by indexing the RAG papers).

- **100 arXiv papers** (PDFs) covering retrieval-augmented generation, dense retrieval, graph-augmented retrieval, multimodal RAG, and adjacent literature
- **581 pre-extracted figures** (PNG, ≥200x200 px) for multimodal indexing in Tier 4 — 8 figures carry verbatim captions extracted from source PDFs
- **0 video clips** — the video-clip portion was deferred (sandbox could not verify a CC license against the candidate slideslive.com source). See [`.planning/phases/127-repository-skeleton-enterprise-dataset/127-05-SUMMARY.md`](https://github.com/PatrykQuantumNomad/PatrykQuantumNomad/blob/main/.planning/phases/127-repository-skeleton-enterprise-dataset/127-05-SUMMARY.md) in the planning repo. Tier 4's video extension stays a post-Phase-127 enhancement.
- **Total**: 290.0 MB stored via [git-lfs](https://git-lfs.com)
- **Evaluation**: 30 hand-authored golden Q&A in [`evaluation/golden_qa.json`](./evaluation/golden_qa.json) — 10 single-hop / 10 multi-hop citation-chain / 10 multimodal (the 3 video slots from the locked D-04 split were substituted with multimodal extras to compensate for the video deferral; see the evaluation README)

```
dataset/
├── papers/      # 100 PDFs (LFS)
├── images/      # 581 extracted figures (LFS)
├── videos/      # empty — Plan 05 deferred
└── manifests/
    ├── papers.json     # 100 entries with arxiv_id, title, year, abstract
    ├── figures.json    # 581 entries with figure_id, paper_id, bbox, caption
    └── metadata.json   # top-level dataset index (regenerable via scripts/build_metadata.py)
```

Provenance and per-asset licensing in [`dataset/README.md`](./dataset/README.md).

## Repository Structure

```
.
├── dataset/             # Corpus (LFS): papers, figures, manifests
├── evaluation/          # Golden Q&A + per-tier results landing zone
│   ├── golden_qa.json   # 30 hand-authored evaluation questions
│   └── results/         # Per-run cost JSON written by CostTracker.persist
├── shared/              # Cross-tier utilities (config, llm, embeddings, loader,
│                        # cost_tracker, pricing, display) — shared by all 5 tiers
├── scripts/             # One-off CLI tools (curate_corpus, extract_figures,
│                        # cut_video_clips, build_metadata)
├── tests/               # Pytest suite (live tests gated behind @live marker)
├── tier-1-naive/        # Tier 1 — Naive ChromaDB RAG (Phase 128)
├── tier-2-file-search/  # Tier 2 — Gemini File Search (Phase 128)
├── tier-3-graph/        # Tier 3 — LightRAG (Phase 129)
├── tier-4-multimodal/   # Tier 4 — RAG-Anything (Phase 130)
├── tier-5-agentic/      # Tier 5 — OpenAI Agents (Phase 130)
└── pyproject.toml       # Editable install with [shared] + [tier-N] extras
```

## Setup

### Code only (skip LFS objects, ~5 MB)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/PatrykQuantumNomad/rag-architecture-patterns.git
cd rag-architecture-patterns
uv venv && source .venv/bin/activate
uv pip install -e ".[shared]"
cp .env.example .env
# Edit .env to set GEMINI_API_KEY (https://aistudio.google.com/app/apikey)
```

### Full dataset (pulls all LFS objects, ~290 MB)

```bash
git clone https://github.com/PatrykQuantumNomad/rag-architecture-patterns.git
# Or, after a smudge-skipped clone:
git lfs pull
```

### Tier-specific dependencies

```bash
uv pip install -r tier-1-naive/requirements.txt   # or tier-2, tier-3, tier-4, tier-5
```

Each tier's `requirements.txt` is a one-liner pointing at the parent `pyproject.toml` extras (`-e ..[tier-N]`), so the tier deps stay a single source of truth.

### Smoke test (validates `shared/` + Gemini connectivity, <5s, ~$0.0001/run)

```bash
pytest tests/smoke_test.py -m live
```

The smoke test imports every shared module, calls `get_settings()` to verify
`GEMINI_API_KEY` is populated, then issues one real `embed("hello world")`
and one real `complete("Reply with exactly: ok")` call against Gemini's
default flash + embedding-001 models. Per-run cost is ~$0.0001.

### Full test suite

```bash
pytest tests/ -v -m 'not live'   # 49 tests, no API calls, <1s
pytest tests/ -v                 # 49 + 3 live tests with .env set
```

## Evaluation

The 30-question golden set lives at [`evaluation/golden_qa.json`](./evaluation/golden_qa.json) — the same questions are asked of every tier. The shipped split is:

| Bucket | Count | Modality | Hop | Tier this targets |
|--------|------:|----------|-----|-------------------|
| Single-hop text | 10 | text | single | All tiers |
| Multi-hop text (citation-chain) | 10 | text | multi | Tier 3 (Graph), Tier 5 (Agentic) |
| Multimodal | 10 | multimodal | mixed | Tier 4 (Multimodal) |
| Video | 0 | — | — | (deferred — see Dataset note) |

Per-run cost JSON lands under `evaluation/results/costs/`. The full evaluation harness is built in Phase 131.

See [`evaluation/README.md`](./evaluation/README.md) for the entry schema.

## License

- **Code**: Apache-2.0 (see [`LICENSE`](./LICENSE)).
- **Corpus**: Licensed per-asset — see [`dataset/README.md`](./dataset/README.md). PDFs follow [arXiv's distribution license](https://arxiv.org/licenses/); figures inherit their parent paper's terms; future video clips will be individually CC-licensed (BY, BY-SA, BY-NC, or BY-NC-SA — never ND).
