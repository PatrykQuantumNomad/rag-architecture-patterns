# Technology Stack

**Analysis Date:** 2026-05-04

## Languages

**Primary:**
- Python 3.10+ - All 5 RAG architecture tiers, shared utilities, scripts, and evaluation harness

## Runtime

**Environment:**
- Python 3.10+ (requirement enforced in `pyproject.toml`)

**Package Manager:**
- uv (referenced in README setup instructions for virtual environment and dependency management)
- pip (used alongside uv; both compatible via `pyproject.toml`)
- setuptools 68+ (build backend)
- wheel (build system)

**Lockfile:**
- No explicit lockfile; dependencies pinned via version ranges in `pyproject.toml`

## Frameworks

**Core LLM/Embedding:**
- google-genai 1.73+ - Primary Gemini client for shared LLM service (`shared/llm.py`), legacy smoke test, and default models
- openai 1.50+ - OpenAI SDK routed through OpenRouter gateway for embeddings (Tier 1+) and chat completions
- litellm 1.0+ - Unified LLM routing layer for evaluation scoring and Tier 5 agent (Pitfall 10: requires `openrouter/` prefix)

**RAG & Knowledge Graph:**
- lightrag-hku 1.4.15 (pinned exactly) - Tier 3 Graph RAG with entity/relationship extraction; LightRAG-native cost tracking
- raganything 1.2.10 - Tier 4 multimodal RAG framework; wraps LightRAG internally
- chromadb 1.5.8+ - Tier 1 vector database (HNSW, cosine similarity); co-located with Tier 5 reuse

**Agents & Orchestration:**
- openai-agents[litellm] 0.14.6 - Tier 5 agentic framework; SDK-native `@function_tool` and `FileSearchTool` support

**Evaluation & Benchmarking:**
- ragas 0.4.3+ - Evaluation harness: `faithfulness`, `answer_relevancy`, `context_precision` metrics; LiteLLM integration for judge LLM
- datasets 4.0+ - Hugging Face dataset format for RAGAS SingleTurnSample integration

**Data Processing:**
- pymupdf 1.27+ - PDF text extraction (Tier 1, Tier 3) and figure extraction (scripts)
- Pillow 10+ - Image handling for multimodal pipelines (Tier 4 figure processing)

**Corpus Curation:**
- arxiv 3.0+ - Official arXiv API client (3-second delay enforced per ToU in `scripts/curate_corpus.py`)
- semanticscholar 0.12+ - Semantic Scholar API for citation-graph traversal

**Configuration & Infrastructure:**
- pydantic 2.10+ - Runtime settings validation (environment variables with `SecretStr` for API keys)
- pydantic-settings 2.10+ - `.env` file parsing and configuration management (`shared/config.py`)
- python-dotenv 1.0+ - Environment variable loading from `.env` files
- rich 14+ - Terminal UI and tables for display (`shared/display.py`)
- tiktoken 0.10+ - Token counting for cost tracking

**Testing:**
- pytest 8.4+ - Test framework and runner
- pytest-live marker (custom) - Gate live API tests with `@live` marker

**Build/Dev:**
- ruff 0.6+ - Linting and code quality (included in dev group)
- setuptools 68+ - Python packaging and module discovery

## Configuration

**Environment:**
- `.env` file (gitignored; template: `.env.example`) containing:
  - `OPENROUTER_API_KEY` (REQUIRED for Tier 1+) - Routes OpenAI embeddings and LLM calls through https://openrouter.ai/api/v1
  - `GEMINI_API_KEY` (REQUIRED for Phase 127 smoke test and legacy shared.llm) - Native Gemini access via google-genai SDK
  - `OPENAI_API_KEY` (OPTIONAL) - Direct OpenAI; Tier 1 routes through OpenRouter post-Phase 128
  - `S2_API_KEY` (OPTIONAL) - Semantic Scholar API key; improves corpus curation rate limits
  - `DEFAULT_CHAT_MODEL` (default: `google/gemini-2.5-flash`) - Model slug for LLM calls
  - `DEFAULT_EMBEDDING_MODEL` (default: `openai/text-embedding-3-small`) - OpenRouter/OpenAI embedding model
  - `DATASET_ROOT` (default: `dataset`) - Path to corpus directory

**Pydantic Settings:**
- `shared/config.py` - Lazy factory pattern (`get_settings()` returns process-wide singleton) validates REQUIRED keys only at call time, not import time
- Environment config is isolated from initialization to allow `from shared import config` without `.env`

**Build:**
- `pyproject.toml` - Single source of truth for all dependencies, optional extras per tier, and pytest markers
- `setup.py` implicit via setuptools backend with explicit package list (handles hyphens in directory names: `tier-1-naive` → `tier_1_naive`)

## Platform Requirements

**Development:**
- Python 3.10+ (verified in `pyproject.toml` and README)
- Virtual environment (venv or equivalent) — all tiers tested with `uv venv`
- git-lfs for dataset pull (290 MB; optional; skip with `GIT_LFS_SKIP_SMUDGE=1`)
- ~5 MB disk (code-only; ~290 MB with dataset via git-lfs)

**Production/Inference:**
- Python 3.10+
- Network connectivity to:
  - OpenRouter (https://openrouter.ai) for Tier 1, 3, 4, 5
  - OpenAI embeddings (via OpenRouter) for all tiers requiring vectors
  - Gemini API (for legacy Phase 127 smoke test and optional embedding fallback)
- Storage:
  - `chroma_db/tier-1-naive/` - Tier 1 ChromaDB vector store (persistent, on-disk)
  - `lightrag_storage/tier-3-graph/` - Tier 3 LightRAG graph + KV indices (persistent)
  - `rag_anything_storage/tier-4-multimodal/` - Tier 4 RAG-Anything storage (persistent)
  - `evaluation/results/` - Cost and scoring JSON outputs (transient, per-run)

**Evaluation:**
- Same network/storage requirements plus temporary space for evaluation JSON logs and metrics
- RAGAS scoring incurs additional LLM calls (~$0.20-0.50 per full 30-question suite with Gemini 2.5 Flash judge)

---

*Stack analysis: 2026-05-04*
