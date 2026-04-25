# RAG Architecture Patterns

Indexing the RAG literature itself to compare 5 architectures.

## Blog Post

This repo is the companion to the blog post **[RAG Architecture Patterns](https://patrykgolabek.dev/blog/rag-architecture-patterns)** (coming soon).

The post walks through the design, tradeoffs, and measured cost/quality of each tier against a single shared corpus — the meta-recursive twist being that the corpus IS the RAG literature. Every architecture indexes the papers that taught us how to build it.

## Architecture Tiers

| Tier | Name | One-line description |
|------|------|----------------------|
| 1 | Naive RAG | Embed-and-retrieve baseline with `chromadb` + `google-genai`. Cheapest, most code, hardest to tune. |
| 2 | Managed File Search | OpenAI Assistants File Search API — zero retrieval code, vendor-managed index. |
| 3 | Graph (LightRAG) | Knowledge-graph RAG with entity/relationship extraction; multi-hop queries become graph walks. |
| 4 | Multimodal (RAG-Anything) | Heterogeneous corpus: PDFs + extracted figures + video transcripts in one retrieval pipeline. |
| 5 | Agentic | OpenAI Agents SDK orchestrating `FileSearchTool` + `@function_tool` over the same corpus. |

Each tier lives in its own `tier-{N}-{name}/` directory with an isolated `requirements.txt`. The shared utilities and dataset are common to all five.

## Dataset

Roughly 100 arXiv papers anchored on the RAG / retrieval-augmented LM citation cluster, augmented with extracted figures and 1–2 CC-licensed video clips — all stored via [git-lfs](https://git-lfs.com) (~300–500 MB total).

- `dataset/papers/` — PDFs (LFS, populated in Plan 04)
- `dataset/images/` — Extracted figures (LFS, populated in Plan 05)
- `dataset/videos/` — CC-licensed talks/explainers (LFS, populated in Plan 05)
- `dataset/manifests/` — JSON manifests with provenance, license, and IDs

See [`dataset/README.md`](./dataset/README.md) for per-asset licensing.

## Setup

```bash
# Code-only clone (skip the ~300–500 MB of LFS binaries)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/PatrykQuantumNomad/rag-architecture-patterns.git
cd rag-architecture-patterns

# Pull the full corpus when you actually want to run a tier
git lfs pull

# Create a venv and install shared utilities
uv venv
uv pip install -e ".[shared]"

# Configure secrets
cp .env.example .env
# Then edit .env and set GEMINI_API_KEY (https://aistudio.google.com/app/apikey)
```

To install a specific tier's dependencies:

```bash
uv pip install -r tier-1-naive/requirements.txt   # or tier-2, tier-3, tier-4, tier-5
```

## Evaluation

A 30-question golden set lives at `evaluation/golden_qa.json` (10 single-hop / 10 multi-hop / 7 multimodal / 3 video) — the same questions are asked of every tier. Per-run cost JSON lands under `evaluation/results/costs/`. The harness itself is built in Phase 131 (Evaluation Harness).

See [`evaluation/README.md`](./evaluation/README.md) for the schema.

## License

- **Code**: Apache-2.0 (see [`LICENSE`](./LICENSE)).
- **Corpus**: Licensed per-asset — see [`dataset/README.md`](./dataset/README.md). PDFs follow [arXiv's distribution license](https://arxiv.org/licenses/); video clips are individually CC-licensed (BY, BY-SA, BY-NC, or BY-NC-SA — never ND).
