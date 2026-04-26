# Tier 1: Naive RAG

> Baseline RAG via the **OpenRouter unified gateway**: ChromaDB + `openai/text-embedding-3-small` (1536-dim) + a configurable chat model (default `google/gemini-2.5-flash`, override with `--model`). Direct API calls, no framework.

This is the **baseline** against which Tiers 2–5 prove their value. Naive RAG is intentionally simple — one embedding model, one vector store, one chat model, and a single retrieval hop. It works extremely well for single-paper questions, and the places where it fails are exactly the motivation for the higher tiers.

Tier 1 routes both embeddings and chat through [OpenRouter](https://openrouter.ai), so a single `OPENROUTER_API_KEY` unlocks the whole pipeline and lets you swap chat models on the command line without touching code (`--model anthropic/claude-haiku-4.5`, `--model openai/gpt-4o-mini`, etc.). The OpenAI Python SDK is fully compatible with OpenRouter — only the `base_url` and key change.

If you read only one tier in this repo, this is the one to start with: every higher tier compares its results against `chroma_db/tier-1-naive/`.

---

## Quickstart

```bash
cd tier-1-naive
uv pip install -r requirements.txt          # or: cd .. && uv pip install -e ".[tier-1]"
cp ../.env.example ../.env                  # then edit and set OPENROUTER_API_KEY (+ GEMINI_API_KEY for Phase 127 smoke test)
python main.py                              # default: ingest if empty, then run canned query
python main.py --model anthropic/claude-haiku-4.5 --query "..."  # swap chat model on the fly
```

Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys); browse models at [openrouter.ai/models](https://openrouter.ai/models). The first run ingests every PDF in `dataset/papers/` into `chroma_db/tier-1-naive/` (one-time cost ~$0.011 for 100 papers) and then answers the canonical demo question. Subsequent runs reuse the index — `cmd_ingest` is idempotent and prints a yellow notice when the collection is already populated.

---

## CLI reference

| Flag | Purpose | Default |
|------|---------|---------|
| `--ingest` | Run ingest (PDFs → chunks → embeddings → ChromaDB). Idempotent. | off |
| `--query "..."` | Run a single retrieval-augmented query against the persisted index. | canned demo question |
| `--top-k N` | Number of chunks to retrieve. | 5 |
| `--reset` | Wipe the collection before `--ingest`. Required when changing HNSW config. | off |
| `--model <slug>` | OpenRouter chat model slug for answer generation. Must exist in `shared/pricing.py` for cost tracking. | `google/gemini-2.5-flash` |

Invoking `python main.py` with **no flags** auto-runs `--ingest` (idempotent) followed by the default `--query` against the default `--model`. This is the canonical "does it work?" demo.

---

## Expected cost (vintage 2026-04)

OpenRouter passes provider pricing through 1:1, so the underlying-vendor numbers below remain accurate. Swap the chat model with `--model` to see a different per-query cost; the embedding cost is fixed by the model used for the index.

| Operation | Cost |
|-----------|------|
| One-time ingest (100 papers, ~570k embed tokens via `openai/text-embedding-3-small`) | ~$0.011 |
| Per-query embedding | <$0.000001 |
| Per-query chat (~3k in / 200 out) — `google/gemini-2.5-flash` | ~$0.0014 |
| Per-query chat (~3k in / 200 out) — `anthropic/claude-haiku-4.5` | ~$0.004 |
| Per-query chat (~3k in / 200 out) — `openai/gpt-4o-mini` | ~$0.0006 |
| **Total per query** (`google/gemini-2.5-flash`, excl. ingest) | **~$0.0014** |

Cost and latency are printed for every query via `shared.display.render_query_result` plus a `Latency:` footer line, and a per-run JSON is written to `evaluation/results/costs/tier-1-<timestamp>.json`. To add a model not yet in `shared/pricing.py`, append a row keyed on its OpenRouter slug (e.g. `"meta-llama/llama-3.3-70b-instruct"`) with the verified vendor price.

---

## Persistence

The index lives at **`chroma_db/tier-1-naive/`**. This path is the convention for the entire repo:

- **Tier 5 (Phase 130, Agentic RAG)** reads from `chroma_db/tier-1-naive/` as a read-only baseline and writes its own modifications to `chroma_db/tier-5-agentic/` (Pitfall 8 — collection collisions).
- To start over: `python main.py --ingest --reset`. This deletes the collection and re-embeds the full corpus (~$0.011 again).
- ChromaDB stores SQLite metadata + HNSW binaries on disk; the directory is portable and can be tarred up for sharing.

---

## Known weaknesses (deliberate)

These are not bugs — they are the **baseline**, and each one motivates a specific higher tier. The blog narrative for this repo turns each weakness into a chapter.

### Multi-hop

Questions that require synthesizing information from **more than one paper** typically retrieve only one of the relevant documents. With a single-vector-search hop and `top_k=5`, the second paper's chunks rarely score high enough to surface. See **Tier 3 (Graph RAG / LightRAG)** for the entity-graph fix.

### Multimodal

Questions that depend on a **figure, chart, or image** score near zero. Tier 1 is text-only via `PyMuPDF.get_text("text")` — figures are stripped at ingest time and never embedded. See **Tier 4 (Multimodal RAG-Anything)** for the figure-aware pipeline.

### Citation chain

Questions like "what does Lewis et al. cite as their DPR encoder?" fail when the cited work is not in the corpus, or when the citation is mangled by PDF text extraction. There's no entity resolution, no cross-document linking, no graph traversal. See **Tier 3** again — entity graphs walk citations natively.

A representative weakness probe is `multimodal-010` (or any multi-hop entry) in `evaluation/golden_qa.json`. Run with: `python main.py --query "..."`.

---

## Sample query

The default canned query is **`single-hop-001`** from `evaluation/golden_qa.json` — the canonical "Tier 1 looks great here" demo:

> *"What is the core mechanism Lewis et al. 2020 introduce in the RAG paper for combining parametric and non-parametric memory?"*

This is single-hop, single-paper, text-only. Tier 1 nails it. Run any multi-hop or multimodal entry from `golden_qa.json` and you'll see the failure modes above.

---

## Architecture

The 7-stage pipeline, end-to-end:

```
PDF
 └─> extract_pages          (PyMuPDF, text-only)
      └─> chunk_page        (512 tokens / 64 overlap, tiktoken cl100k_base)
           └─> embed_batch  (OpenRouter -> openai/text-embedding-3-small, 1536-dim, batch=100)
                └─> ChromaDB (cosine HNSW, persistent at chroma_db/tier-1-naive/)
                     └─> retrieve_top_k (k=5)
                          └─> build_prompt (context-stuffed)
                               └─> chat.complete (OpenRouter, --model selectable)
                                    └─> render_query_result + Latency:
```

All stages are direct API calls — **no LangChain, no LlamaIndex, no framework**. The whole tier is ~600 lines across `ingest.py`, `embed_openai.py`, `chat.py`, `store.py`, `retrieve.py`, `prompt.py`, and `main.py`. The OpenAI Python SDK does double duty: it speaks to OpenRouter for both embeddings and chat by overriding `base_url` to `https://openrouter.ai/api/v1`. Higher tiers will introduce frameworks where they earn their keep.

---

## Reused by

**Phase 130 / Tier 5 (Agentic RAG)** treats `chroma_db/tier-1-naive/` as a tool / read-only baseline — the agent loop calls Tier 1's collection as one of several retrievers and decides per-query whether the naive answer is good enough. This is why the persistence path is locked: changing it would break the downstream tier.

If you only run one tier in this repo, run this one. Then read `evaluation/golden_qa.json` and find a question Tier 1 gets wrong — that's your motivation to read the next tier.
