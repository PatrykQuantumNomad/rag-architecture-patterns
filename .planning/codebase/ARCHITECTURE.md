<!-- refreshed: 2026-05-04 -->
# Architecture

**Analysis Date:** 2026-05-04

## System Overview

The RAG Architecture Patterns repository demonstrates five progressively sophisticated retrieval-augmented generation implementations, each isolated in its own tier directory with independent dependencies. All tiers share a common dataset, evaluation framework, and utility layer.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLI Entry Points (main.py)                         │
│  tier-1-naive/  tier-2-managed/  tier-3-graph/  tier-4-multimodal/  tier-5/ │
│    ChromaDB       Gemini File       LightRAG        RAG-Anything      OpenAI │
│                      Search                          + LightRAG        Agents │
└────────┬──────────────────────────────────┬──────────────────────────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Shared Core Utilities                               │
│  shared/config, llm, embeddings, cost_tracker, pricing, display, loader    │
│  - Settings management (env + lazy validation)                              │
│  - Gemini-backed LLM + Embedding clients (Protocol-based)                   │
│  - Token usage tracking + USD cost computation + JSON persistence           │
│  - Dataset manifest loading (papers, figures, metadata)                     │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Data & Evaluation                                      │
│  dataset/ — 100 PDFs (LFS), 581 figures, manifests (papers.json, etc)       │
│  evaluation/ — 30 golden Q&A, cost results landing zone                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Tier 1 (Naive RAG)** | ChromaDB vector store with OpenRouter embeddings/chat | `tier-1-naive/main.py` |
| **Tier 2 (Managed)** | Gemini File Search (vendor-managed retrieval) | `tier-2-managed/main.py` |
| **Tier 3 (Graph RAG)** | LightRAG entity/relationship extraction + graph traversal | `tier-3-graph/main.py` |
| **Tier 4 (Multimodal)** | RAG-Anything heterogeneous corpus (PDFs + figures) | `tier-4-multimodal/main.py` |
| **Tier 5 (Agentic)** | OpenAI Agents orchestrating FileSearch + function calls | `tier-5-agentic/main.py` |
| **shared/config.py** | Pydantic Settings, lazy `get_settings()` factory, env validation | `shared/config.py` |
| **shared/llm.py** | Gemini-backed chat completions (Protocol-based) | `shared/llm.py` |
| **shared/embeddings.py** | Gemini embeddings client (Protocol-based) | `shared/embeddings.py` |
| **shared/cost_tracker.py** | Token + USD cost recording and JSON persistence | `shared/cost_tracker.py` |
| **shared/loader.py** | Dataset manifest loading (papers, figures, videos, metadata) | `shared/loader.py` |
| **shared/display.py** | Rich-powered query/chunks/answer/cost rendering | `shared/display.py` |
| **scripts/** | One-off CLI tools (corpus curation, figure extraction, metadata build) | `scripts/` |
| **tests/** | Pytest suite (unit tests + live smoke tests) | `tests/` |

## Pattern Overview

**Overall:** Multi-tier comparative architecture demonstrating cost/quality tradeoffs in RAG.

**Key Characteristics:**
- **Tier isolation**: Each tier has its own `main.py` entry point, requirements.txt, and (mostly) independent module structure
- **Shared utilities layer**: All tiers depend on `shared/` for config, LLM, embeddings, cost tracking, dataset loading, and display
- **CLI uniformity**: Each tier follows the same argparse pattern: `--ingest` (with `--reset`), `--query`, and no-flags default (auto-ingest + canned query)
- **Cost-first design**: Every tier tracks token usage via `CostTracker` and persists JSON to `evaluation/results/costs/` for downstream aggregation
- **Protocol-based abstractions**: Shared LLM/embedding clients are Protocols, allowing each tier to substitute providers (e.g., Tier 1 uses OpenRouter while shared defaults to Gemini)

## Layers

**CLI Entry Point:**
- Purpose: Parse flags, orchestrate ingest/query flow, render results
- Location: `tier-{N}-{name}/main.py`
- Contains: argparse setup, cmd_ingest, cmd_query subcommands
- Depends on: Tier-specific modules + shared utilities
- Used by: End users via `python tier-N-{name}/main.py`

**Tier-Specific Logic:**
- Purpose: Implement the retrieval mechanism for that tier
- Location: `tier-{N}-{name}/` (e.g., `retrieve.py`, `rag.py`, `query.py`)
- Contains: Index creation, retrieval, query execution
- Depends on: Shared utilities + external SDKs (chromadb, google-genai, lightrag-hku, etc.)
- Used by: The tier's main.py CLI

**Shared Utilities:**
- Purpose: Reusable infrastructure across all tiers
- Location: `shared/`
- Contains: Config, LLM/embedding clients, cost tracking, pricing, dataset loading, display
- Depends on: External libraries (pydantic, google-genai, rich)
- Used by: All tiers

**Dataset & Evaluation:**
- Purpose: Curated corpus and evaluation harness
- Location: `dataset/` (papers, figures, manifests), `evaluation/` (golden Q&A, cost results)
- Contains: 100 PDFs (LFS), 581 figures (LFS), manifests, 30 hand-authored questions
- Depends on: Git LFS
- Used by: All tiers for ingest and query validation

## Data Flow

### Primary Request Path (Tier 1 example)

1. **CLI Parse** — `main.py` argparse reads flags (`--query`, `--ingest`, `--reset`, `--model`) (`tier-1-naive/main.py:238-240`)
2. **Collection Open** — `store.open_collection()` creates/opens ChromaDB with cosine HNSW (`tier-1-naive/store.py:23-51`)
3. **Ingest (if needed)**:
   - Load papers from `dataset/manifests/papers.json` via `DatasetLoader` (`shared/loader.py:37-39`)
   - Extract PDF pages via `extract_pages()` (`tier-1-naive/ingest.py:21-34`)
   - Chunk pages (512 tokens, 64-token overlap) via `chunk_page()` (`tier-1-naive/ingest.py:37-76`)
   - Embed chunks in batches via OpenRouter `text-embedding-3-small` (`tier-1-naive/main.py:113-127`)
   - Store embeddings + metadata in ChromaDB (`tier-1-naive/main.py:118-123`)
4. **Query Execution**:
   - Embed query via `embed_batch()` → OpenRouter (`tier-1-naive/main.py:160-161`)
   - Retrieve top-k via `retrieve_top_k()` with cosine similarity (`tier-1-naive/retrieve.py:13-47`)
   - Build RAG prompt via `build_prompt()` (context-stuffing pattern) (`tier-1-naive/prompt.py:11-43`)
   - Complete chat via `complete()` → OpenRouter (`tier-1-naive/main.py:165-166`)
5. **Render & Cost** — Display via `render_query_result()`, track cost in `CostTracker`, persist JSON (`tier-1-naive/main.py:181-193`)

### Secondary Path (Tier 3 async flow)

Tier 3 (Graph RAG) flips to async because `lightrag-hku` is async-only:
1. `main()` → `asyncio.run(amain(args, console))` (single bridge point) (`tier-3-graph/main.py:200+`)
2. `amain()` — all ingest/query work runs inside async context
3. `build_rag()` — creates LightRAG instance with `CostAdapter(tracker)` token tracker
4. `ingest_corpus()` — async document insertion with entity/relationship extraction
5. `run_query()` — async query execution with hybrid/mix mode (keyword + graph traversal)
6. Results → shared `render_query_result()`, cost JSON persisted via `CostTracker`

**State Management:**
- **Tier 1**: ChromaDB persistent on-disk, idempotency via collection existence check
- **Tier 2**: Gemini File Store (vendor-managed), store ID cached to `.store_id`
- **Tier 3**: LightRAG working directory (`lightrag_storage/tier-3-graph/`), status tracked in `kv_store_doc_status.json`
- **Tier 4**: RAG-Anything graph + LightRAG output graph
- **Tier 5**: OpenAI Agent state (stateless per query, shared File Search)

## Key Abstractions

**LLMClient (Protocol):**
- Purpose: Pluggable chat completion interface
- Examples: `shared/llm.py` (Gemini), `tier-1-naive/chat.py` (OpenRouter)
- Pattern: `complete(prompt, model=None) -> LLMResponse`
- Result fields: `text`, `input_tokens`, `output_tokens`, `model`

**EmbeddingClient (Protocol):**
- Purpose: Pluggable embedding interface
- Examples: `shared/embeddings.py` (Gemini), `tier-1-naive/embed_openai.py` (OpenRouter)
- Pattern: `embed(text, model=None) -> list[list[float]]`
- Always returns batched vectors for predictable handling

**CostTracker:**
- Purpose: Record token usage + compute USD + persist JSON
- Pattern: `record_llm(model, input_tokens, output_tokens)`, `record_embedding(model, input_tokens)`
- Persistence: `persist(dest_dir) -> Path` writes to `evaluation/results/costs/{tier}-{timestamp}.json`
- Schema: Fixed D-13 structure (tier, timestamp, queries[], totals)

**DatasetLoader:**
- Purpose: Lazy, graceful manifest loading without requiring `.env`
- Pattern: `papers()`, `figures()`, `videos()`, `metadata()` — cached methods
- Fallback: Returns `[]` or `{}` for missing manifests (plan-phased dataset)

## Entry Points

**CLI (user-facing):**
- `python tier-1-naive/main.py` — auto-ingest + canned query
- `python tier-1-naive/main.py --ingest` — ingest only
- `python tier-1-naive/main.py --query "..." --top-k 5` — specific query
- All tiers follow this pattern

**Smoke Test (CI/verification):**
- `pytest tests/smoke_test.py -m live` — validates shared modules + Gemini connectivity
- Tests imports, `get_settings()` validation, one embed + one complete call

**Script (data prep):**
- `python scripts/build_metadata.py` — regenerate `dataset/manifests/metadata.json`
- `python scripts/curate_corpus.py` — populate `papers.json` from arXiv seeds
- `python scripts/extract_figures.py` — extract images from PDFs → `figures.json`

## Architectural Constraints

- **Threading:** Single-threaded event loop (Tier 3 async runs inside `asyncio.run()` bridge). No worker threads.
- **Global state:** Process-wide singletons in `shared/` (LLM client, embedding client, settings via `@lru_cache`). Tier-specific singletons (ChromaDB client, LightRAG rag instance) scoped to tier CLI.
- **Circular imports:** None enforced; tier modules import from shared, not vice versa.
- **Lazy validation:** Settings are validated at `get_settings()` call (not import time) to allow fresh checkouts without `.env`.
- **Collection/store isolation:** Each tier owns its own storage path (e.g., `chroma_db/tier-1-naive/`, `lightrag_storage/tier-3-graph/`, `.store_id` per tier) to prevent cross-tier collisions.
- **Protocol-based dispatch:** LLM/embedding clients are Protocols, so each tier can substitute without modifying shared layer.

## Anti-Patterns

### Module-level Settings Singleton

**What happens:** Importing `from shared.config import settings` causes `Settings()` to instantiate at import time.

**Why it's wrong:** Fresh checkouts without `.env` fail at import, breaking the "read-only imports succeed" contract (breaks tests/smoke_test.py::test_imports).

**Do this instead:** Use the lazy factory `from shared.config import get_settings()` which only instantiates when called, deferring ValidationError to code paths that actually need keys (`tier-1-naive/main.py:243`, `shared/llm.py:53`).

### Tier-Specific Code in Shared

**What happens:** A retrieval method or cost adapter is added to `shared/` instead of tier-specific modules.

**Why it's wrong:** Each tier has different indexing strategies (vector, graph, vendor-managed), cost calculation quirks, and dependencies. Mixing them in shared breaks isolation.

**Do this instead:** Place tier-specific logic in `tier-{N}-{name}/` (e.g., `cost_adapter.py` for Tier 3's LightRAG token tracker mapping, `ingest.py` for Tier 1's chunking strategy). Use the Protocol pattern (LLMClient, EmbeddingClient) when shared layer needs to call tier-provided behavior.

### Embedding Model Inconsistency

**What happens:** Different tiers embed with different models without tracking which model was used.

**Why it's wrong:** Embedding similarity is model-specific. Tier 1 uses text-embedding-3-small (OpenRouter); Tier 3 uses Gemini embeddings. Cross-tier comparisons break if you mix vectors from different models.

**Do this instead:** Store the embedding model name in chunk metadata (Tier 1 stores `paper_id`, `page`, `chunk_idx`; Tier 3 passes to LightRAG's embedding config). The cost tracker records model per call (`record_embedding(model, tokens)`), so audits reveal any drift.

### Hardcoded Paths Outside STRUCTURE

**What happens:** Paths like `dataset/papers/` are assumed scattered throughout tier modules.

**Why it's wrong:** Moving or renaming the dataset directory requires grep across all tiers.

**Do this instead:** Centralize path resolution: use `DatasetLoader(dataset_root=get_settings().dataset_root)` to get the configurable root, then construct paths from it (e.g., `dataset_root / "papers"` in `tier-1-naive/main.py:100`).

## Error Handling

**Strategy:** Fail fast with friendly messages, exit code 2 for config/setup errors (exit code 0 for success, non-zero for operational failure).

**Patterns:**
- **Missing `.env` key**: `pydantic.ValidationError` raised by `get_settings()` call; CLI catches and prints red message with setup instructions (Tier 1 OPENROUTER_API_KEY guard at `main.py:243-250`)
- **Empty collection/index**: Check collection count before query; print "run --ingest first" and exit 2 (`tier-1-naive/main.py:151-156`)
- **Unknown models**: `CostTracker.record_llm()` raises `KeyError` if model not in `shared.pricing.PRICES`; tier CLI bubbles it with "add model to pricing table" guidance
- **PDF extraction**: Graceful skip (log yellow warning, continue) if PDF file missing (`tier-1-naive/main.py:101-102`)
- **API retries**: Tier 2 and Tier 3 implement backoff (30s/60s/120s) for transient 503 errors; idempotent via deduplication checks

## Cross-Cutting Concerns

**Logging:** Console output via `rich.console.Console` and `console.print()` (colored, paneled, tabular). No file logging configured; stderr for errors, stdout for output.

**Validation:** Settings via Pydantic at `get_settings()` time. Model validation (e.g., pricing lookup) at cost-recording time. Dataset manifest validation in `DatasetLoader._read_list()` (type checking JSON structure).

**Authentication:** API keys from `.env` via Pydantic, stored as `SecretStr` (masked in repr). Lazy loading so imports don't fail without keys.

---

*Architecture analysis: 2026-05-04*
