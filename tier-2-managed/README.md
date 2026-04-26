# Tier 2: Managed RAG (Gemini File Search)

> Zero-infrastructure managed RAG. Google does the chunking, embedding, indexing, and retrieval; we just upload PDFs and ask questions.

This tier shows what happens when you outsource the entire retrieval pipeline to a managed service. There is **no local vector store, no chunker, no embedder, and no retrieval call** — just `client.file_search_stores.upload_to_file_search_store(...)` to ingest, and `client.models.generate_content(..., tools=[FileSearch(...)])` to query. The model itself decides when to consult the index, and citations come back already structured in `response.candidates[0].grounding_metadata.grounding_chunks` — no regex over free text, no manually-assembled chunk list.

That's the **managed-RAG win** vs Tier 1: same answer to the same question, with built-in citations as a first-class response field. The trade-off is opacity — you can't tweak chunk size, embedding model, or HNSW parameters. For a corpus where the defaults are good enough, the LOC + ops savings are dramatic.

If you read Tier 1 first (recommended), this tier should feel **smaller**. The whole tier is ~480 LOC across `store.py`, `query.py`, and `main.py` — most of which is 503-retry resilience and synthetic cost accounting. The "real" RAG pipeline is two SDK calls.

---

## Quickstart

```bash
cd tier-2-managed
uv pip install -r requirements.txt          # or: cd .. && uv pip install -e ".[tier-2]"
cp ../.env.example ../.env                  # then edit and set GEMINI_API_KEY
python main.py                              # default: create store + upload 100 papers + run canned query
```

Get a Gemini API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). The first run creates a `FileSearchStore` in Google's cloud (the handle is cached at `tier-2-managed/.store_id`, gitignored), uploads every PDF in `dataset/papers/` sequentially with 30s/60s/120s exponential backoff (Pitfall 2 — 503 storms at full-corpus scale), and answers the canonical demo question. Subsequent runs skip already-uploaded PDFs by consulting `documents.list()` first, so `--ingest` is idempotent.

Note: `[tier-2]` does NOT add new top-level dependencies — `google-genai>=1.73` from `[shared]` already exposes `client.file_search_stores.*` (added in google-genai SDK v1.49). The `[tier-2]` extra exists in `pyproject.toml` as a stub so `uv pip install -e ".[tier-2]"` is a stable command users can rely on if the dep contract changes later.

---

## CLI reference

| Flag | Description | Default |
|------|-------------|---------|
| `--ingest` | Upload PDFs to the FileSearchStore (idempotent — skips already-uploaded `display_name` matches). | off |
| `--query "..."` | Run a single managed-RAG query against the persisted store. | canned demo question |
| `--reset` | Delete the FileSearchStore + clear `.store_id` sidecar before `--ingest`. Required to re-index from scratch. | off |
| `--model <slug>` | Gemini chat model for the answer step. Must be present in `shared/pricing.py` for cost tracking. | `gemini-2.5-flash` |

Invoking `python main.py` with **no flags** auto-runs `--ingest` (idempotent) followed by the default `--query`. This is the canonical "does it work?" demo and the same pattern Tier 1 uses.

---

## Expected cost (vintage 2026-04)

Gemini File Search bundles indexing into per-query latency without exposing a per-token line item, so the indexing column below is a **synthetic estimate** — see the footnote.

| Operation | Tokens | Price | Cost |
|-----------|--------|-------|------|
| One-time indexing (estimated, ~500K tokens for 100 papers) | 500,000 | $0.15 / 1M | **~$0.075** |
| Per-query LLM (`gemini-2.5-flash`, ~3K in / 200 out) | ~3,200 | $0.30 in / $2.50 out | **~$0.0014** |
| Storage (after indexing) | — | $0 | $0 |
| Query-time embedding | — | $0 | $0 |

> **Footnote (Pitfall 7):** Indexing cost is a **synthetic estimate**. The Gemini File Search API does not return per-upload token counts. We approximate via PyMuPDF text extraction + tiktoken `cl100k_base` × $0.15/1M (matching the published `gemini-embedding-001` rate). The first live run prints the actual estimate alongside the LLM-line cost from `response.usage_metadata`, which is exact. The cost JSON written to `evaluation/results/costs/tier-2-{timestamp}.json` records both the synthetic embedding line AND the exact LLM line per the D-13 schema.

To add a chat model not yet in `shared/pricing.py`, append a row keyed on the Gemini slug (e.g. `gemini-2.5-pro`) with the verified vendor price.

---

## Persistence

The store handle is cached at **`tier-2-managed/.store_id`** (gitignored from Plan 129-01 — Pattern 1 store-id caching). The actual store + index live in **Google's cloud** — there is no local DB file like Tier 1's `chroma_db/tier-1-naive/`. The cached handle is just a server-assigned resource name (`fileSearchStores/<id>`) used to re-attach to the same store on subsequent runs.

To start over: `python main.py --ingest --reset` — this calls `client.file_search_stores.delete(name=..., config={"force": True})` to drop the server-side store, removes the local sidecar, and re-creates a fresh store on next ingest.

The `.store_id` sidecar is gitignored because every collaborator gets their own store — the file is per-machine state, not shared infrastructure. Treat it the same way you'd treat a `.env` file: local, machine-specific, never committed.

---

## Known weaknesses (deliberate)

These are not bugs — they are the **managed-RAG envelope**, and several motivate higher tiers. The blog narrative for this repo turns each weakness into a chapter.

### Multi-hop questions

Same as Tier 1 — single-shot retrieval, can't traverse cross-document entities. The model retrieves once via the FileSearch tool, then answers; there is no second hop and no entity graph. See **Tier 3 (Graph RAG / LightRAG)** for the entity-graph fix that walks citations natively.

### 503 storms during ingest at full corpus

Multiple users on `discuss.ai.google.dev` report 503 Service Unavailable on `fileSearchStores.upload_to_file_search_store()` and `importFile()` — failing after ~80s consistently — while the basic `files.upload()` works fine. Root cause is speculated to be Embedding-Model-TPM saturation on Google's side. Mitigation: sequential uploads (never concurrent) with 30s/60s/120s exponential backoff, implemented in `tier_2_managed.store.upload_with_retry`. Resumable ingest via `documents.list()` skip-already-uploaded means you can hit Ctrl-C mid-storm and resume cleanly. Source: `https://discuss.ai.google.dev/t/file-search-store-api-returns-503-for-all-file-sizes-files-upload-works-fine/121691`.

### Indexing cost is a synthetic estimate

There is no per-token indexing line item from Google. Tier 2 records a synthetic `gemini-embedding-001` line in the cost tracker (cl100k_base estimate × $0.15/1M) per Pitfall 7. The LLM-line cost from `response.usage_metadata` is exact. Phase 131's evaluation pass will report the actual delta between estimate and bill once we can measure it.

### Citations are surface-level

`grounding_metadata.grounding_chunks[i].retrieved_context` carries `title` (the document's `display_name`) and `text` (the snippet the model used). The optional `score` field is **not exposed by `gemini-2.5-flash`** in our testing (Open Q3 from the research pass — flash-tier models seem to omit it; the Pro models surface it). `to_display_chunks` defends with `getattr(ctx, "score", 0.0) or 0.0` so the renderer's `f"{score:.3f}"` formatting never raises. If you switch `--model gemini-2.5-pro`, you may see real similarity scores; the live test SUMMARY records the empirical answer.

### Multimodal (image / figure-grounded) questions

Not yet tested with File Search. Google's docs are silent on whether the managed pipeline rasterizes embedded figures or treats PDFs as text-only. Defer to **Tier 4 (Multimodal RAG-Anything)** for the explicit figure-aware pipeline.

A representative weakness probe is any multi-hop entry in `evaluation/golden_qa.json`. Run with `python main.py --query "..."`.

---

## Sample query

The default canned query is **`single-hop-001`** from `evaluation/golden_qa.json` — the same question Tier 1 uses, intentionally:

> *"What is the core mechanism Lewis et al. 2020 introduce in the RAG paper for combining parametric and non-parametric memory?"*

Tier 1 answers this from a manually-assembled chunk list (returned by ChromaDB's similarity search). Tier 2 answers the same question with **built-in citations via `grounding_metadata`** — same answer, but the chunks come back as a structured response field rather than a separate retrieval call. That's the managed-RAG win this tier is designed to demonstrate.

Run any multi-hop entry from `golden_qa.json` and you'll see Tier 2 hit the same single-hop ceiling as Tier 1 — that's the motivation for Tier 3.

---

## Architecture

The whole tier, end-to-end:

```
PDFs
 └─> client.file_search_stores.upload_to_file_search_store
      (long-running operation; poll until done)
      └─> [Google chunks + embeds + indexes — opaque managed pipeline]
           └─> client.models.generate_content(model=..., tools=[FileSearch(...)])
                └─> response.text + response.candidates[0].grounding_metadata.grounding_chunks
                     └─> to_display_chunks (defensive None / score handling)
                          └─> shared.display.render_query_result
                               + Latency: footer
                               + Cost JSON written to evaluation/results/costs/tier-2-<ts>.json
```

The managed pipeline (the bracketed step) is opaque by design — that's the whole point of Tier 2. Compare to Tier 1's 7-stage pipeline (`extract_pages → chunk_page → embed_batch → ChromaDB → retrieve_top_k → build_prompt → chat.complete`) and you see the LOC delta is real: ~480 LOC for Tier 2 vs ~600 LOC for Tier 1, with most of Tier 2's lines being 503-retry resilience and synthetic cost accounting rather than retrieval logic.

---

## Reused by

**Phase 130 / Tier 5 (Agentic RAG)** may use Gemini File Search as **one of multiple retrieval tools** available to the agent — the agent loop will be free to call Tier 1's `chroma_db/tier-1-naive/` collection AND Tier 2's `FileSearchStore` AND any Tier 3 graph endpoint, choosing per-query whichever is most likely to give a good answer. The store handle (cached in `.store_id`) is the contract: Tier 5 reads the resource name from there to re-attach to the same store without re-ingesting.

If you only run one tier in this repo to feel the LOC / ops savings of managed RAG, run this one — then read Tier 3 to see what you give up in exchange for managed simplicity.
