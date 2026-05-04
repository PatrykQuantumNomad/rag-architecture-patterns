# Codebase Structure

**Analysis Date:** 2026-05-04

## Directory Layout

```
rag-architecture-patterns/
├── .claude/                    # GSD (get-shit-done) config + templates
├── .planning/                  # GSD planning artifacts
│   └── codebase/              # Architecture + structure docs (you are here)
├── shared/                     # Cross-tier utilities (8 modules)
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings + lazy get_settings()
│   ├── llm.py                 # Gemini LLMClient Protocol + factory
│   ├── embeddings.py          # Gemini EmbeddingClient Protocol + factory
│   ├── cost_tracker.py        # Token usage + USD tracking + JSON persistence
│   ├── pricing.py             # Model → USD/1M-token lookup table
│   ├── loader.py              # DatasetLoader for manifests
│   └── display.py             # render_query_result() → rich Panel/Table
├── dataset/                    # Corpus (LFS: ~290 MB)
│   ├── papers/                # 100 PDFs (LFS-tracked)
│   ├── images/                # 581 extracted figures (LFS-tracked)
│   ├── videos/                # (empty; deferred to Phase 132)
│   ├── manifests/             # JSON indexes
│   │   ├── papers.json        # 100 entries: arxiv_id, title, year, abstract, license, filename
│   │   ├── figures.json       # 581 entries: figure_id, paper_id, bbox, caption
│   │   ├── metadata.json      # Top-level index (regenerable)
│   │   └── videos.json        # (empty; deferred)
│   └── README.md              # Dataset provenance + per-asset licensing
├── evaluation/                 # Golden Q&A + results landing zone
│   ├── golden_qa.json         # 30 hand-authored questions
│   │                           # - 10 single-hop text
│   │                           # - 10 multi-hop citation-chain text
│   │                           # - 10 multimodal (mixed)
│   ├── results/
│   │   └── costs/             # CostTracker JSON outputs ({tier}-{timestamp}.json)
│   └── README.md              # Entry schema + evaluation methodology
├── tier-1-naive/              # Tier 1: Naive ChromaDB RAG (Phase 128)
│   ├── __init__.py
│   ├── main.py                # CLI: argparse + cmd_ingest + cmd_query
│   ├── store.py               # ChromaDB collection factory (cosine HNSW)
│   ├── ingest.py              # PDF extraction + page-aware chunking (512 tokens, 64-token overlap)
│   ├── retrieve.py            # Top-k with cosine-similarity normalization
│   ├── prompt.py              # RAG prompt builder (context-stuffing)
│   ├── chat.py                # OpenRouter chat completion client
│   ├── embed_openai.py        # OpenRouter embedding client (text-embedding-3-small)
│   ├── requirements.txt        # Single-line: -e ..[tier-1]
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_chunker.py
│   │   ├── test_store.py
│   │   └── test_main_live.py
│   └── README.md              # Tier 1 design rationale + research references
├── tier-2-managed/            # Tier 2: Gemini File Search (Phase 128)
│   ├── __init__.py
│   ├── main.py                # CLI (same pattern as Tier 1)
│   ├── store.py               # File Store creation + upload_with_retry
│   ├── query.py               # FileSearch tool wiring
│   ├── requirements.txt
│   ├── tests/
│   ├── .store_id              # Cached store ID (idempotency marker)
│   └── README.md
├── tier-3-graph/              # Tier 3: LightRAG (Phase 129)
│   ├── __init__.py
│   ├── main.py                # Async CLI: asyncio.run(amain(...))
│   ├── rag.py                 # build_rag() factory + LightRAG config
│   ├── ingest.py              # Async document insertion with extraction
│   ├── query.py               # Async query with hybrid/mix modes
│   ├── cost_adapter.py        # Token tracker → CostTracker bridge
│   ├── requirements.txt
│   ├── tests/
│   └── README.md
├── tier-4-multimodal/         # Tier 4: RAG-Anything (Phase 130)
│   ├── __init__.py
│   ├── main.py                # Async CLI
│   ├── rag.py                 # RAG-Anything + LightRAG orchestration
│   ├── ingest_pdfs.py         # PDF ingest pipeline
│   ├── ingest_images.py       # Image ingest pipeline
│   ├── query.py               # Multimodal query execution
│   ├── cost_adapter.py        # Token tracking (RAGAS/LiteLLMEmbeddings quirks)
│   ├── Dockerfile             # Containerized ingest (for reproducibility)
│   ├── expected_output.md     # Reference outputs
│   ├── output/                # Generated outputs (multimodal indices)
│   ├── scripts/               # Tier 4-specific utilities
│   ├── requirements.txt
│   ├── tests/
│   └── README.md
├── tier-5-agentic/            # Tier 5: OpenAI Agents (Phase 130)
│   ├── __init__.py
│   ├── main.py                # CLI (same pattern)
│   ├── agent.py               # Agent + tools setup
│   ├── tools.py               # FileSearchTool + @function_tool wiring
│   ├── expected_output.md
│   ├── requirements.txt
│   ├── tests/
│   └── README.md
├── scripts/                   # One-off CLI tools for data prep
│   ├── __init__.py
│   ├── build_metadata.py      # Regenerate metadata.json from papers/figures
│   ├── curate_corpus.py       # Fetch arXiv papers → papers.json
│   ├── extract_figures.py     # Extract images from PDFs → figures.json
│   ├── cut_video_clips.py     # (Phase 132: video curation)
│   ├── probe_lightrag_token_tracker.py  # Verify LightRAG token tracker integration
│   ├── seed_papers.json       # arXiv seed IDs for corpus curation
│   ├── video_sources.json     # (deferred)
│   └── README.md              # Script documentation
├── tests/                     # Shared test suite (pytest)
│   ├── conftest.py            # Pytest fixtures (tmp_path, mock settings)
│   ├── smoke_test.py          # Import validation + Gemini connectivity (live)
│   ├── test_cost_tracker.py   # CostTracker.record_llm/record_embedding/persist
│   ├── test_dataset.py        # DatasetLoader.papers/figures/metadata
│   ├── test_display.py        # render_query_result output shape
│   ├── test_env_example.py    # .env.example completeness check
│   ├── test_golden_qa.py      # golden_qa.json schema validation
│   ├── test_loader.py         # DatasetLoader edge cases
│   ├── test_pricing.py        # PRICES table completeness for known models
│   ├── test_repo_metadata.py  # Metadata.json regenerability
│   └── test_tier_requirements.txt  # Tier extras are installable
├── chroma_db/                 # ChromaDB persistent storage (generated)
│   └── tier-1-naive/
│       └── [.chroma/ collection data]
├── lightrag_storage/          # LightRAG working directory (generated)
│   └── tier-3-graph/
│       ├── graph_chunk_entity_relation.graphml
│       └── kv_store_doc_status.json
├── rag_anything_storage/      # RAG-Anything output (generated)
│   └── [indices, embeddings]
├── .env                       # (user-created, not committed)
│                              # Required: GEMINI_API_KEY (for smoke test)
│                              # Optional: OPENROUTER_API_KEY (for Tier 1, 3, 4)
│                              # Optional: OPENAI_API_KEY (legacy)
│                              # Optional: S2_API_KEY (corpus curation)
├── .env.example               # Template with all keys documented
├── .gitignore                 # Includes: .env, chroma_db/, lightrag_storage/, __pycache__/
├── pyproject.toml             # Package config + extras (shared, tier-1..5, curation)
├── uv.lock                    # Pinned dependencies (uv package manager)
├── LICENSE                    # Apache-2.0
├── README.md                  # Project overview + setup
└── .gitattributes             # LFS filter for .pdf, .png

[Generated at runtime]
├── chroma_db/tier-1-naive/.chroma/      # ChromaDB persistent index
├── lightrag_storage/tier-3-graph/       # LightRAG graph + status
├── rag_anything_storage/                # RAG-Anything indices
└── evaluation/results/costs/*.json      # CostTracker outputs
```

## Directory Purposes

**`shared/`:**
- Purpose: Cross-tier utilities (config, LLM/embedding clients, cost tracking, dataset loading, display)
- Contains: 8 Python modules, each a focused responsibility
- Key files: `config.py` (settings factory), `cost_tracker.py` (USD tracking), `loader.py` (manifest I/O)

**`dataset/`:**
- Purpose: Curated corpus (100 PDFs, 581 figures, manifests)
- Contains: LFS-tracked papers/ and images/, JSON manifests, per-asset licensing docs
- Key files: `papers.json` (100 entries), `figures.json` (581 entries), `metadata.json` (top-level index)
- Committed: Manifests (small JSON), `.gitattributes` rules; PDFs/images via LFS

**`evaluation/`:**
- Purpose: Evaluation harness (golden Q&A, cost result landing zone)
- Contains: `golden_qa.json` (30 questions), `results/costs/` directory for per-run JSON
- Key files: `golden_qa.json` (fixed, hand-authored), `results/costs/{tier}-{timestamp}.json` (auto-generated)

**`tier-{N}-{name}/`:**
- Purpose: Isolated tier implementation
- Contains: CLI, ingest logic, retrieval logic, tier-specific utilities, tests, requirements.txt
- Key files: `main.py` (entry point), `requirements.txt` (single-line: `-e ..[tier-N]`)
- Pattern: All tiers follow same CLI structure (argparse, cmd_ingest, cmd_query)

**`scripts/`:**
- Purpose: One-off data preparation tools
- Contains: Corpus curation, figure extraction, metadata building, LightRAG probing
- Key files: `curate_corpus.py` (fetch papers), `extract_figures.py` (extract images)

**`tests/`:**
- Purpose: Shared test suite covering common utilities and cross-tier concerns
- Contains: Pytest tests, fixtures (conftest.py), both unit and live (API-calling) tests
- Key files: `smoke_test.py` (CI gate), `test_cost_tracker.py`, `test_loader.py`

**`.claude/` & `.planning/`:**
- Purpose: GSD (get-shit-done) tooling configuration and generated planning docs
- Contains: Agent skills, templates, phase planning (outside this repo)

## Key File Locations

**Entry Points:**
- `tier-1-naive/main.py`: Naive RAG CLI
- `tier-2-managed/main.py`: Managed RAG CLI
- `tier-3-graph/main.py`: Graph RAG CLI (async)
- `tier-4-multimodal/main.py`: Multimodal RAG CLI (async)
- `tier-5-agentic/main.py`: Agentic RAG CLI

**Configuration:**
- `shared/config.py`: Pydantic Settings, lazy factory pattern
- `.env`: User-populated (GEMINI_API_KEY required, others optional)
- `.env.example`: Template with all keys documented
- `pyproject.toml`: Package definition, extras (shared, tier-1..5, curation)

**Core Logic:**
- `tier-1-naive/store.py`: ChromaDB collection management
- `tier-1-naive/ingest.py`: PDF extraction + chunking
- `tier-1-naive/retrieve.py`: Top-k retrieval + cosine normalization
- `tier-1-naive/prompt.py`: RAG prompt builder
- `shared/cost_tracker.py`: Token usage + USD computation
- `shared/loader.py`: Dataset manifest loading

**Testing:**
- `tests/smoke_test.py`: Import validation + Gemini connectivity (live)
- `tests/conftest.py`: Pytest fixtures
- `tier-1-naive/tests/`: Tier 1-specific tests
- `tier-3-graph/tests/`: Tier 3-specific tests (async)

**Data:**
- `dataset/manifests/papers.json`: 100 papers index
- `dataset/manifests/figures.json`: 581 figures index
- `evaluation/golden_qa.json`: 30 hand-authored questions

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `cost_tracker.py`, `embed_openai.py`)
- Test files: `test_*.py` (pytest convention)
- Data files: lowercase with underscores (e.g., `papers.json`, `golden_qa.json`)
- Storage directories: `lowercase-with-hyphens` for tier names (`tier-1-naive`, `tier-2-managed`)
- Generated outputs: Tier-specific subdirs under storage roots (e.g., `chroma_db/tier-1-naive/`)

**Directories:**
- Tier directories: `tier-{number}-{name}` (e.g., `tier-1-naive`, `tier-3-graph`)
- Utility directories: lowercase (`shared`, `scripts`, `dataset`, `evaluation`, `tests`)
- Python packages: underscores for non-hyphenated shims (e.g., `tier_1_naive` importable as `from tier_1_naive`)

**Functions:**
- Public API: `verb_noun()` (e.g., `build_prompt()`, `embed_batch()`, `render_query_result()`)
- Internal/private: `_leading_underscore()` (e.g., `_lookup_price()`, `_GeminiLLMClient`)
- Factory methods: `get_*()` or `build_*()` (e.g., `get_settings()`, `get_llm_client()`, `build_rag()`)

**Variables:**
- Chunk IDs: `{paper_id}_p{page:03d}_c{chunk_idx:03d}` (e.g., `2004.07159_p001_c000`)
- Metadata keys: lowercase with underscores (e.g., `paper_id`, `chunk_idx`)
- Model slugs: OpenRouter format (e.g., `google/gemini-2.5-flash`, `anthropic/claude-haiku-4.5`)

## Where to Add New Code

**New Tier Implementation:**
1. Create directory `tier-{N}-{name}/`
2. Add `__init__.py` (empty or re-export key classes)
3. Add `main.py` following the argparse + cmd_ingest + cmd_query pattern
4. Add tier-specific modules (e.g., `rag.py`, `ingest.py`, `query.py`, `retrieve.py`)
5. Add `requirements.txt` with single line: `-e ..[tier-N]`
6. Add `pyproject.toml` entry under `[project.optional-dependencies]`: `tier-N = [...]`
7. Add tests in `tier-{N}-{name}/tests/`
8. Update `README.md` and `pyproject.toml` table

**New Shared Utility Module:**
- File: `shared/new_module.py`
- Pattern: Exportable function/class, lazy initialization if accessing API keys
- Document: Add docstring to `shared/__init__.py` Modules section
- Update imports in tiers that need it

**New Dataset Asset:**
1. Add files to `dataset/papers/`, `dataset/images/`, or `dataset/videos/`
2. Track via Git LFS (`.gitattributes` already configured for `.pdf`, `.png`)
3. Update manifest: `dataset/manifests/papers.json` or `dataset/manifests/figures.json`
4. Regenerate top-level: `python scripts/build_metadata.py`
5. Commit manifest JSON + update LFS tracking

**New Test:**
- File: `tests/test_new_concern.py` (shared) or `tier-{N}-{name}/tests/test_new_concern.py` (tier-specific)
- Pattern: Use pytest fixtures from `conftest.py` (e.g., `tmp_path`, `mock_settings`)
- Marker: Use `@pytest.mark.live` for tests that call APIs, `@pytest.mark.skip` for deferred work

**New Script:**
- File: `scripts/new_tool.py`
- Pattern: argparse for CLI, main() entry point, docstring with usage
- Document: Add entry to `scripts/README.md`

## Special Directories

**`chroma_db/`:**
- Purpose: ChromaDB persistent storage
- Generated: Yes (auto-created by Tier 1)
- Committed: No (`.gitignore` excludes `chroma_db/`)
- Cleanup: Delete to reset Tier 1 index

**`lightrag_storage/`:**
- Purpose: LightRAG working directory (graph, status, embeddings)
- Generated: Yes (auto-created by Tier 3 on first ingest)
- Committed: No (`.gitignore` excludes `lightrag_storage/`)
- Cleanup: Delete to reset Tier 3 graph

**`rag_anything_storage/`:**
- Purpose: RAG-Anything indices and embeddings
- Generated: Yes (auto-created by Tier 4)
- Committed: No (`.gitignore` excludes)
- Cleanup: Delete to reset Tier 4 multimodal indices

**`evaluation/results/costs/`:**
- Purpose: CostTracker JSON outputs (one file per tier run)
- Generated: Yes (auto-created by `CostTracker.persist()`)
- Committed: No (`.gitignore` excludes `results/`)
- Pattern: `{tier}-{timestamp}.json` (e.g., `tier-1-20260425T120000Z.json`)

**`.pytest_cache/`:**
- Purpose: Pytest cache
- Generated: Yes
- Committed: No

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No

**`.claude/` & `.planning/`:**
- Purpose: GSD tooling (agent skills, phase planning)
- Generated: Partially (agent writes to `.planning/codebase/`)
- Committed: Yes (skills + templates)

---

*Structure analysis: 2026-05-04*
