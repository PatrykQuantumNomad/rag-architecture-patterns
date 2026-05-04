# External Integrations

**Analysis Date:** 2026-05-04

## APIs & External Services

**Language Model APIs:**
- OpenRouter (https://openrouter.ai) - Unified gateway routing to OpenAI, Anthropic, Google, Meta, and others
  - SDK/Client: `openai` SDK pointed at `https://openrouter.ai/api/v1` base URL
  - Auth: `OPENROUTER_API_KEY` (env var)
  - Usage: Tier 1 embeddings + chat, Tier 3 graph extraction + chat, Tier 4 multimodal LLM/vision/embed, Tier 5 agent chat
  - Model slugs (OpenRouter format): `openai/text-embedding-3-small` (embeddings), `google/gemini-2.5-flash` (default LLM)

- Google Gemini (native via google-genai SDK) - Direct Gemini API (backup to OpenRouter; Phase 127 smoke test only)
  - SDK/Client: `google-genai>=1.73,<2`
  - Auth: `GEMINI_API_KEY` (env var, REQUIRED for smoke test)
  - Usage: `shared/llm.py` legacy client (Phase 127), default embedding model fallback
  - Endpoint: https://generativelanguage.googleapis.com (implicit in SDK)

**Citation & Knowledge Graph APIs:**
- arXiv (https://arxiv.org) - Paper metadata and PDF downloads
  - SDK/Client: `arxiv>=3.0,<4` (official client)
  - Auth: None (public API; rate limits enforced client-side with 3-second delay per ToU)
  - Usage: `scripts/curate_corpus.py` — paper discovery and PDF retrieval
  - Enforced delay: `ARXIV_DELAY_SECONDS = 3.0` (ToU compliance; no parallelism)

- Semantic Scholar (https://api.semanticscholar.org) - Citation graph traversal
  - SDK/Client: `semanticscholar>=0.12,<1`
  - Auth: `S2_API_KEY` (optional; raises 429s without key; exponential backoff in `scripts/curate_corpus.py`)
  - Usage: `scripts/curate_corpus.py` — reference + citation lookup (1-hop out from seed papers)
  - Rate limiting: `S2_MAX_RETRIES = 5`, `S2_BACKOFF_BASE_SECONDS = 5.0`

**OpenAI Agents Platform (Tier 5):**
- OpenAI Agents SDK (https://github.com/openai/agents-sdk)
  - SDK/Client: `openai-agents[litellm]==0.14.6` (vendor-maintained; includes LiteLLM support)
  - Auth: `OPENROUTER_API_KEY` (routed via LiteLLM, not native OpenAI)
  - Usage: Tier 5 agent orchestration with `FileSearchTool` + `@function_tool` support
  - Tracing: Disabled at module load to suppress stderr warnings (`agents.set_tracing_disabled()` in `tier_5_agentic/agent.py`)

## Data Storage

**Vector Databases:**
- ChromaDB (local persistent client)
  - Type: In-process vector database with HNSW indexing
  - Client: `chromadb>=1.5.8,<2`
  - Storage: `chroma_db/tier-1-naive/` (Tier 1), `chroma_db/tier-5-agentic/` (Tier 5 shared baseline)
  - Configuration: Cosine similarity (`HNSW` space="cosine"); 1536-dim vectors (OpenAI embedding-3-small)
  - Collection: `enterprise_kb_naive` (Tier 1 collection name)
  - Implementation: `tier-1-naive/store.py::open_collection()`

**Graph & Knowledge Stores (Tier 3):**
- LightRAG (bundled with `lightrag-hku==1.4.15`)
  - Storage: `lightrag_storage/tier-3-graph/` (isolated per HKUDS spec)
  - Stores: Graph database + KV store for entity/relationship indices
  - Configuration: Embedding dim = 1536 (LOCKED at first ingest; Pitfall 4 — do NOT change without `--reset`)
  - Implementation: `tier-3-graph/rag.py::build_rag()` constructs LightRAG instance

**Multimodal RAG Storage (Tier 4):**
- RAG-Anything (wraps LightRAG internally)
  - Storage: `rag_anything_storage/tier-4-multimodal/` (distinct from Tier 3; prevents cross-contamination)
  - Type: Multimodal vector + graph store supporting PDFs, images, and vision LLM callbacks
  - Client: `raganything==1.2.10`
  - Configuration: Embedding dim = 1536 (same lock as Tier 3; NEVER expose `--embed-model` as CLI flag)
  - Implementation: `tier-4-multimodal/rag.py::build_rag()`

**File Storage:**
- Local filesystem only
  - Dataset: `dataset/papers/` (100 PDFs via git-lfs), `dataset/images/` (581 figures via git-lfs)
  - Manifests: `dataset/manifests/papers.json`, `dataset/manifests/figures.json`, `dataset/manifests/metadata.json`
  - Per-tier outputs: `evaluation/results/queries/`, `evaluation/results/metrics/`, `evaluation/results/costs/`
  - Log location: `evaluation/results/` (JSON logs per query + per-tier cost records)

**Caching:**
- None (all retrieval is live; no persistent cache layer beyond vector stores)

## Authentication & Identity

**Auth Providers:**
- API-key based authentication (no OAuth/OIDC)
  - OpenRouter: `OPENROUTER_API_KEY` (Bearer token in Authorization header)
  - Google Gemini: `GEMINI_API_KEY` (passed to `google.genai.Client()`)
  - Semantic Scholar: `S2_API_KEY` (optional; query parameter in HTTP requests)
  - OpenAI (fallback): `OPENAI_API_KEY` (optional; not used post-Phase 128 for Tier 1)

**Configuration Source:**
- `.env` file (`shared/config.py`) parsed by pydantic-settings at first call to `get_settings()`
- All keys are `SecretStr` (pydantic type) to prevent accidental logging
- Lazy validation: Keys are checked only when `get_settings()` is called, not at import time

## Monitoring & Observability

**Error Tracking:**
- None (errors are propagated to CLI; pytest tests mark live calls with `@live` marker)

**Logs:**
- Console output only (via `rich` for rich terminal display)
- Cost tracking: `shared/cost_tracker.py` persists cost JSON to `evaluation/results/costs/`
  - Per-tier costs: `{tier}-{timestamp}.json` (D-13 schema; model, input tokens, output tokens, cost, timestamp)
  - Judge costs: `ragas-judge-{tier}-{timestamp}.json` (evaluation harness only)
- No persistent log aggregation; all logs are stdout/stderr

**Token Counting:**
- tiktoken (OpenAI's tokenizer) — used for cost estimation before API calls
- LightRAG cost tracking: `tier-3-graph/cost_adapter.py` wraps LightRAG's native `token_tracker` kwarg
- Tier 4 cost tracking: `tier-4-multimodal/cost_adapter.py` (same pattern as Tier 3; 1.4.15-validated)

## CI/CD & Deployment

**Hosting:**
- User-deployed (batch inference script or interactive CLI)
- No cloud platform integration (all inference runs locally)

**CI Pipeline:**
- None (no GitHub Actions, GitLab CI, or similar)
- Test execution: Manual via pytest (`pytest tests/ -v`)
- Live tests gated: `@live` marker (requires `.env` with API keys; not run by default)

## Environment Configuration

**Required env vars:**
- `OPENROUTER_API_KEY` - For Tier 1+ (embeddings + chat via OpenRouter)
- `GEMINI_API_KEY` - For Phase 127 smoke test + legacy shared.llm

**Optional env vars:**
- `OPENAI_API_KEY` - Legacy direct OpenAI (not needed post-Phase 128)
- `S2_API_KEY` - Semantic Scholar rate limit increase (corpus curation only)
- `DEFAULT_CHAT_MODEL` - Override LLM model slug (default: `google/gemini-2.5-flash`)
- `DEFAULT_EMBEDDING_MODEL` - Override embedding model (default: `openai/text-embedding-3-small`)
- `DATASET_ROOT` - Override corpus path (default: `dataset`)

**Secrets location:**
- `.env` file (gitignored; created by user from `.env.example` template)
- Never committed to repository

**Model Override env vars (per-tier):**
- `TIER3_LLM_MODEL` - Tier 3 LLM model override (read lazily in `tier-3-graph/rag.py`)
- `TIER4_LLM_MODEL` - Tier 4 LLM model override (read lazily in `tier-4-multimodal/rag.py`)

## Webhooks & Callbacks

**Incoming:**
- None (RAG system is stateless; no webhook consumption)

**Outgoing:**
- None (all communication is request-response via REST APIs)

## Integration Patterns

**OpenRouter Unified Gateway (Tiers 1, 3, 4, 5):**
- Single `OPENROUTER_API_KEY` routes embeddings, chat, and vision calls across all tiers
- OpenAI SDK used with `base_url="https://openrouter.ai/api/v1"` override
- Model slugs follow OpenRouter format: `{provider}/{model-name}` (e.g., `openai/text-embedding-3-small`)
- LiteLLM integration (Tier 5, evaluation): Requires `openrouter/` prefix on slugs (Pitfall 10)

**Cost Tracking Cross-Tier:**
- `shared/cost_tracker.py` - Single CostTracker instance per tier invocation
- Records: `(model, input_tokens, output_tokens, timestamp)` → cost computed via `shared/pricing.py`
- Outputs: `evaluation/results/costs/{tier}-{timestamp}.json` (D-13 schema)
- Integration: Passed as `token_tracker` kwarg to `openai_complete_if_cache()` and `openai_embed()` (LightRAG 1.4.15 contract)

**Lazy Environment Read:**
- All API keys read from `os.environ` INSIDE async closures (Tier 3, 4) or factory functions (Tier 1, 5)
- Module imports succeed without `.env` (required by test suite)
- Validation occurs only when called (not at import time)

---

*Integration audit: 2026-05-04*
