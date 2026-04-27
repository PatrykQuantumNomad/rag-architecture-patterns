> [!WARNING]
> **First run downloads ~3-5 GB of MineRU layout/OCR models (~5-15 min cold start).**
> Full-corpus ingest costs **~$1-2** (vision + LLM + embed across 100 PDFs + 581 images;
> ~30-60 min depending on rate-limit headroom). For most users, **use Docker** — the image
> bakes the model cache in a build stage. See "Docker Quickstart" below.

# Tier 4: Multimodal RAG (RAG-Anything)

> Unified knowledge graph over **images + tables + text**. PDFs go in, MineRU parses every modality, RAG-Anything builds one KG, queries traverse cross-modal entities.

Tier 4 is the **multimodal chapter** of the repo's narrative. Where Tier 1 (vector top-k) and Tier 3 (graph traversal) both strip figures at extract time, Tier 4 keeps them. RAG-Anything wraps LightRAG with a MineRU-driven layout pass that emits a single content list across text chunks, table cells, equation LaTeX, and rasterized figures — then sends every modality through entity-extraction LLM calls so the resulting graph mixes "encoder layer" entities with "Figure 1 of Attention Is All You Need" entities and the relationships between them.

A representative Tier-4 win is an **image-grounded query that Tier 1/2/3 cannot answer**:

> *"What does Figure 1 of the Attention Is All You Need paper depict, and how does the encoder-decoder architecture relate to retrieval-augmented generation?"*

Tier 1's vector top-k cannot retrieve image content (it embeds text only). Tier 2 (Gemini File Search) accepts PDFs but has no per-image vision pass. Tier 3 (graph) strips figures at extract time. Tier 4 indexes the image during ingest, links it to surrounding text via shared entities, and walks both the image entity and the cross-document conceptual link at query time.

If Tier 3 is "expensive once, much smarter on hard questions", Tier 4 is "even more expensive, AND answers a question class the prior tiers can't touch." The cost asymmetry is real (full-corpus ingest ~$1-2 vs Tier 3 ~$1) but so is the capability delta — Tier 4 unlocks image-grounded multimodal Q&A.

---

## Quickstart (pip)

```bash
cd tier-4-multimodal
uv pip install -r requirements.txt   # or: cd .. && uv pip install -e ".[tier-4]"
cp ../.env.example ../.env           # then edit and set OPENROUTER_API_KEY
python main.py --no-images --yes     # PDFs only; first run triggers MineRU model fetch (~5-15 min)
```

> ⚠️ **First run downloads the MineRU models (~3-5 GB) into `~/.mineru/`.** Plan for ~5-15 min of cold-start time even before any LLM call fires. Use the **Docker Quickstart** below to skip the cold start entirely — the image bakes the model cache in a build stage.

`--no-images` keeps the first run cheap (PDFs only, ~$0.50-1.00 across 100 papers). Drop the flag (or pass `--include-images`) on a follow-up run to include the 581-figure standalone bundle (~$0.30-0.50 vision pass). `--yes` skips the cost-surprise confirmation prompts; omit on interactive runs to see the warnings.

Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys); browse models at [openrouter.ai/models](https://openrouter.ai/models). The same key drives Tier 1, Tier 3, Tier 4, and Tier 5 — Tier 2 is the only Gemini-native exception in this repo.

---

## Docker Quickstart (REPO-05 mandatory path)

**Docker is the recommended path for Tier 4** — the multi-stage build bakes the MineRU model cache in stage 1, so first-time `docker run` is fast (no 3-5 GB download). Tier 4 is the only tier where Docker is **mandatory** for a smooth UX (Tiers 1/2/3/5 are pip-only and pleasant; Tier 4 cold start without Docker is a 5-15 min model download).

```bash
# From repo root:
docker build -t rag-tier-4 -f tier-4-multimodal/Dockerfile .
docker run --rm --env-file .env rag-tier-4   # default canned multimodal query

# Inside the orchestrator's SOCKS5 sandbox (Pitfall 11):
docker run --rm --env-file .env \
  -e ALL_PROXY=socks5h://host.docker.internal:61994 \
  -e HTTP_PROXY=socks5h://host.docker.internal:61994 \
  rag-tier-4 --query "What does Figure 1 of the Attention Is All You Need paper depict?"
```

The image is ~4 GB (slim-bookworm Python 3.11 + LibreOffice + MineRU model cache + Tier-4 deps). Build time is ~10 min on first run; subsequent rebuilds are seconds (BuildKit layer cache). The runtime stage's entrypoint is `tier-4-multimodal/main.py`, so any flag from the **CLI reference** below can be appended after the image name.

`.env` is excluded from the image via `.dockerignore` — the secret is **mounted at run time** via `--env-file .env`. The only template baked in is `.env.example` (no secret material).

---

## CLI reference

| Flag | Purpose | Default |
|------|---------|---------|
| `--ingest` | Run ingest pass (PDFs via MineRU + standalone images). Idempotent on re-run thanks to RAG-Anything's content-hash dedup. | off |
| `--query "..."` | Question to answer. | canned multimodal probe |
| `--mode <m>` | LightRAG query mode: `naive` / `local` / `global` / `hybrid` / `mix`. See descriptions below. | `hybrid` |
| `--model <slug>` | OpenRouter LLM slug for entity extraction + answers + vision. Must be in `shared/pricing.py`. | `google/gemini-2.5-flash` |
| `--reset` | Wipe `rag_anything_storage/tier-4-multimodal/` before `--ingest`. **Re-ingest costs ~$1-2.** | off |
| `--include-images` / `--no-images` | Toggle standalone-figure ingest (581 PNGs from `dataset/images/`). | on |
| `--include-pdfs` / `--no-pdfs` | Toggle PDF ingest (100 papers from `dataset/papers/`). | on |
| `--device <d>` | MineRU compute device: `auto` (default), `cpu`, `cuda:0`, `mps`. Autodetects via torch when `auto`. | `auto` |
| `--yes` | Skip cost-surprise confirmation prompts (for non-interactive use, CI, Docker). | off |

### `--mode` choices

| Mode | What it does | When to use |
|------|--------------|-------------|
| `naive` | Vector-only retrieval — no graph traversal. | Sanity check / ablation. |
| `local` | Entity-neighborhood traversal. | Single-paper queries about specific entities. |
| `global` | Community-summary retrieval. | Broad / synthesizing questions. |
| `hybrid` | `local` + `global` blended. **Safe default.** | Multi-hop and multimodal. |
| `mix` | `hybrid` + reranker (requires `bge-reranker-v2-m3`). | Opt-in once a reranker is configured. |

Invoking `python main.py` with **no flags** auto-runs `--ingest` (gated unless `--yes`) followed by the default `--query` against `--mode hybrid`. This is the canonical "does it work?" demo.

---

## Expected cost (vintage 2026-04)

| Operation | Tokens | Price | Cost |
|-----------|--------|-------|------|
| Full ingest: 100 PDFs (LLM + entity extraction) | ~5M | `google/gemini-2.5-flash` $0.30/1M in / $2.50/1M out | **~$0.50-1.00** |
| Full ingest: 581 images (vision pass) | ~1M | $0.30/1M (image-token rate) | **~$0.30-0.50** |
| Embeddings (PDFs + images) | ~6M | `openai/text-embedding-3-small` $0.02/1M | ~$0.12 |
| Per-query (hybrid mode, ~3K in / 200 out) | ~3,200 | $0.30/1M in / $2.50/1M out | ~$0.0015 |
| **Storage (`rag_anything_storage/`)** | — | local disk | $0 |

Numbers are estimates from `130-RESEARCH.md` @ 2026-04 vintage. The first live run prints the actual costs via `evaluation/results/costs/tier-4-<timestamp>.json` — `CostAdapter` (Plan 02) routes both LLM and embedding usage from RAG-Anything's internal `token_tracker` callback into a single `CostTracker` instance, capturing both ingest and per-query spend.

**Comparison:** Tier 1 ingest ≈ $0.011, Tier 3 ingest ≈ $1.00, Tier 4 ingest ≈ $1-2 (~100-180× Tier 1). The trade is the multimodal capability delta — image-grounded queries the prior tiers cannot answer.

---

## Persistence

RAG-Anything persists graph + KV stores under **`rag_anything_storage/tier-4-multimodal/`** (gitignored). Layout mirrors LightRAG's (because RAG-Anything composes a LightRAG instance internally) plus a Tier-4-specific MineRU intermediate cache:

- `graph_chunk_entity_relation.graphml` — the multimodal NetworkX KG (entities span text + image + table modalities).
- `vdb_entities.json` — vector index over entity descriptions (text + image entity descriptions co-indexed).
- `vdb_relationships.json` — vector index over relationship descriptions.
- `vdb_chunks.json` — vector index over the original chunks (text chunks + table cells + equation LaTeX + image captions).
- `kv_store_full_docs.json` — full document texts keyed by paper_id.
- `kv_store_text_chunks.json` — chunked texts keyed by chunk id.
- `kv_store_doc_status.json` — idempotency log (Pattern 5; deduplicates re-ingest).
- `kv_store_llm_response_cache.json` — cache of LLM/vision responses (re-runs near-free).

Plus **`tier-4-multimodal/output/<paper_id>/`** — MineRU intermediate parses (layout JSON, table HTML, OCR'd text, extracted figures). Gitignored. Useful for debugging when an entity is missing from the KG ("did MineRU even see this page?").

**`EMBED_DIMS = 1536` is hardcoded** at the module level in `tier-4-multimodal/rag.py`. Switching embedding models without `--reset` silently corrupts retrieval (Pitfall 4, inherited from Tier 3 / HKUDS issue #2119). To start over: `python main.py --reset --ingest --yes` — re-extracts entities at full cost.

---

## Known weaknesses (deliberate)

These are the trade-offs Tier 4 makes to deliver multimodal RAG.

### First-run MineRU model download (~3-5 GB; Pitfall 1)

The first ingest call downloads layout-detection, OCR, and table-recognition models from HuggingFace into `~/.mineru/`. Plan for ~5-15 min of cold-start time on a fresh checkout. **Mitigated by Docker** — the multi-stage Dockerfile pre-fetches models during `docker build` so subsequent `docker run` invocations are fast.

### Vision pass calls VLM once per image (Pitfall 5)

The 581-image standalone bundle triggers ~581 vision LLM calls during ingest — one per image, sequential. At ~3 sec/call (with Gemini 2.5 Flash multimodal latency), that's ~30 min cumulative even before any rate-limit pushback. Bursty parallel ingest may surface 429s from OpenRouter on `google/gemini-2.5-flash`; default ingest is sequential to avoid this.

### `aquery` returns a string — no structured `chunks` field (Pitfall 7)

RAG-Anything 1.2.10 returns a plain string from `aquery`. There is no per-chunk citation list to thread into `shared.display.render_query_result`. The `to_display_chunks` helper in `query.py` returns `[]` defensively; `render_query_result` prints "No chunks retrieved." which honestly conveys the multimodal-graph-RAG citation pattern (the answer self-cites in body text). A future raganything release that surfaces structured chunks will populate via the dict-branch in `to_display_chunks`.

### Standalone images need ABSOLUTE `img_path` (Pitfall 4)

`insert_content_list` silently accepts relative paths but produces empty image entities in `vdb_entities.json` (RAG-Anything resolves the path against an internal working directory rather than the caller's CWD). `tier-4-multimodal/ingest_images.py` mitigates by calling `Path(...).resolve()` BEFORE composing the content list — both code paths exposed (`ingest_standalone_images` async + `build_image_content_list` pure helper) enforce this.

### Sparse caption coverage (Phase 127 dataset asymmetry)

Only 8 of 581 figures in `dataset/manifests/figures.json` carry captions extracted from source PDFs (Lewis 2020 RAG, GraphRAG x2, LightRAG x2, Transformer, VisRAG x2). The other 573 figures pass through the vision pass with empty captions. RAG-Anything's vision LLM still describes the image content for entity extraction, but multimodal answer quality on uncaptioned figures depends entirely on what the VLM infers from pixels alone. Phase 131 (RAGAS evaluation) will quantify the captioned-vs-uncaptioned answer-quality delta.

### Docker dependency (REPO-05 honest)

For most users, "use pip" is a strict subset of "use Docker" because the MineRU model download is the cold-start bottleneck. Tier 4 is the only tier in this repo where Docker meaningfully improves first-run UX (Tiers 1/2/3/5 are happy on pip alone). REPO-05 tracks Docker as a **mandatory** path for Tier 4, not a nice-to-have.

A representative Tier-4 win is the multimodal default query; representative losses are queries that don't require image grounding (Tier 1 or 3 cheaper) or queries that require crisp per-chunk citations (Tier 2's grounding metadata better).

---

## Sample query

The default canned query is the **multimodal probe** (verbatim in `tier-4-multimodal/main.py` as `DEFAULT_QUERY`):

> *"What does Figure 1 of the Attention Is All You Need paper depict, and how does the encoder-decoder architecture relate to retrieval-augmented generation?"*

This is one of the multimodal-extra slots from `evaluation/golden_qa.json` (Phase 127 Plan 06's 10/10/10/0 split — the Lewis/RAG/encoder-decoder probe is captioned in `figures.json`). It exercises the full Tier 4 win narrative:

1. **Image grounding** — references "Figure 1" of a specific paper. Tier 1's text-only embeddings cannot retrieve the image; Tier 4's vision-pass entity extraction did.
2. **Cross-document conceptual link** — relates encoder-decoder architecture (a Transformer concept) to retrieval-augmented generation (a Lewis 2020 RAG concept). The hybrid graph mode walks both image and text entities to surface the connection.

Phase 131 (evaluation) will benchmark Tier 4's answer quality against the multimodal slots in `golden_qa.json` to quantify the delta vs Tier 1/3.

---

## Architecture

The end-to-end pipeline:

```
PDFs (100)        -> RAGAnything.process_document_complete (parser="mineru")
                  -> [MineRU: layout + OCR + table + equation extraction]
                  -> LightRAG entity extraction (LLM via OpenRouter)
                  -> rag_anything_storage/tier-4-multimodal/

Standalone PNGs   -> ingest_images.build_image_content_list (absolute img_path)
(581)             -> RAGAnything.insert_content_list
                  -> Vision pass (VLM via OpenRouter) -> entity extraction
                  -> SAME working dir; one unified KG

Query             -> RAGAnything.aquery(question, mode="hybrid")
                  -> LightRAG hybrid: keyword extraction + entity search + chunk retrieval
                  -> answer (string; chunks=[] passed to shared.display defensively)
```

The single `asyncio.run(amain)` bridge in `main.py` is the only async/sync boundary (Pitfall 8); every RAG-Anything call inside `amain` stays async. Cost tracking flows from RAG-Anything's internal LightRAG `token_tracker` callback through `tier_4_multimodal.cost_adapter.CostAdapter` into `shared.cost_tracker.CostTracker` — capturing both LLM/vision and embedding spend on a single key (`OPENROUTER_API_KEY`) on a single gateway (OpenRouter). Outcome A from Phase 129 Plan 03 (probe-validated) extends to Tier 4 verbatim because RAG-Anything composes a LightRAG instance internally.

---

## Reused by

**Phase 131 (RAGAS evaluation)** benchmarks Tier 4 against the multimodal-extra slots in `evaluation/golden_qa.json` to quantify the multimodal capability delta vs Tier 1/2/3. The 10 multimodal questions in that set are the Tier-4-specific evaluation gate.

**Tier 5 (Agentic RAG)** could optionally call Tier 4's multimodal KG as a **3rd retrieval tool** alongside Tier 1's ChromaDB collection (and possibly Tier 3's graph). This is **Open Q4** in `130-RESEARCH.md` and was deliberately deferred — Plan 130-04's Tier 5 ships with two tools (Tier 1 vector search + paper metadata lookup) to keep the agent loop tractable and the demo cost-bounded. Wiring Tier 4 in as a 3rd tool is a Phase 132+ follow-on once RAGAS evaluation has measured whether the multimodal slot meaningfully improves agent decision quality.

If you only run one tier in this repo, run **Tier 1** first (cheap, fast). Then run a multimodal query (e.g. anything referencing a specific figure) against both Tier 1 and Tier 4 — that delta is the entire reason Tier 4 exists.
