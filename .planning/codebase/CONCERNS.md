# Codebase Concerns

**Analysis Date:** 2026-05-04

## Tech Debt

### Upstream Library Incompatibilities — RAGAS 0.4.3

**Issue:** RAGAS 0.4.3 exposes incompatible API surfaces across metrics that cause silent failures and metric loss.

**Files:** `evaluation/harness/score.py`

**Impact:**
- `context_precision` metric calls `embed_query()` (legacy LangChain API) but RAGAS's `LiteLLMEmbeddings` only exposes `embed_text()` → AttributeError mid-evaluation
- `answer_relevancy` metric calls `embed_documents()` but LiteLLMEmbeddings only exposes `embed_texts()` → NaN scores silently persist
- Token usage extraction via `result.total_tokens()` raises IndexError when internal usage_data list is empty (no judge calls piped through token parser)
- Fallback behavior is swallowed by RAGAS scoring loop, causing entire metric columns to become NaN without traceable error

**Workaround Applied:** Lines 125-132 manually alias missing methods on embedding instance after `embedding_factory()` returns it; lines 248-251 wrap token-usage extraction in try/except to prevent IndexError from aborting the full tier score.

**Fix Approach:** Track RAGAS upstream releases. When 0.4.4+ lands, test if the aliases are still needed and remove if the upstream API stabilizes. Consider switching to an alternative metrics framework if RAGAS continues to expose breaking changes across patch versions.

---

### SecretStr Unwrapping Inconsistency

**Issue:** Pydantic SecretStr requires `.get_secret_value()` unwrapping before passing to client constructors, but this requirement is inconsistently applied across the codebase.

**Files:**
- `tier-2-managed/main.py:289` ✓ Correct
- `evaluation/harness/adapters/tier_2.py:89` ✓ Fixed in ce5c2ad
- `shared/config.py` — stores all API keys as SecretStr, but some call sites forgot unwrapping

**Impact:** Client instantiation fails with AttributeError on SecretStr methods (e.g., `.strip()`) that assume string type. This blocks entire tier evaluations before any query data is captured.

**Fix Approach:** Audit all `get_settings()` usage to ensure SecretStr fields are unwrapped via `.get_secret_value()` before passing to external SDKs. Consider creating a helper in `shared/config.py` that returns unwrapped credentials to prevent future misses.

---

### RAGAnything Initialization Pitfall

**Issue:** RAGAnything's `aquery()` method does NOT automatically initialize LightRAG. Only ingest paths (`process_file`, `process_folder`, `aquery_with_multimodal`, `aquery_vlm_enhanced`) trigger `_ensure_lightrag_initialized()` internally.

**Files:** `tier-4-multimodal/scripts/eval_capture.py` (fixed in f0cd134)

**Impact:** Query-only workflows against pre-existing RAGAnything storage raise `ValueError: No LightRAG instance available` on every call. Records persist as empty answers with zero cost, masking the failure.

**Fix Approach:** All query-only paths must explicitly call `rag._ensure_lightrag_initialized()` before first query. Document this as a non-obvious RAGAnything API contract in Tier 4's README.

---

## Known Bugs

### Device Autodetection Fallback May Fail Silently

**Issue:** `_detect_device()` in `tier-4-multimodal/main.py:116-135` catches all exceptions and falls back to `"cpu"`, masking import or runtime errors in torch availability checks.

**Symptoms:** User's GPU might be available but detection silently falls back to CPU due to unrelated exception (e.g., broken torch installation, missing MPS backend). Query runs on wrong device with no warning.

**Files:** `tier-4-multimodal/main.py:116-135`

**Trigger:** `import torch` succeeds but `torch.cuda.is_available()` or `torch.backends.mps` raises exception.

**Workaround:** Check console output for MineRU device line to confirm autodetected device. Explicitly pass `--device cuda:0` or `--device mps` if needed.

---

### Missing Tier 4 Validation Before Cost Gate

**Issue:** First-ingest cost gate (`main.py:331-333`) checks if `graph_chunk_entity_relation.graphml > 1024 bytes` to detect "already built" state. However, RAGAnything creates this file as an empty placeholder during constructor initialization.

**Symptoms:** Fresh checkout with empty storage still shows "already built = True" if the file exists, suppressing the cost confirmation gate.

**Files:** `tier-4-multimodal/main.py:331-333`

**Trigger:** Constructor call at line 320 creates the graphml file before ingest. If user re-runs with `--ingest` without `--reset`, gate silently allows re-ingest even though storage exists.

**Workaround:** Always use `--reset` when wiping storage; do not rely on the heuristic check alone for cost safety.

---

### Async/Sync Boundary in Tier 2 Adapter

**Issue:** `run_tier2()` in `evaluation/harness/adapters/tier_2.py` wraps sync `tier2_query()` call in `asyncio.to_thread()`, but if the underlying tier2_query call has internal async operations, the thread boundary may deadlock or lose context.

**Files:** `evaluation/harness/adapters/tier_2.py:93`

**Impact:** Rare deadlock risk if Gemini SDK's internal implementation shifts to async patterns in future versions.

**Workaround:** Monitor google-genai SDK releases. Current 1.73+ versions are sync-only, so this is safe for now.

---

## Security Considerations

### Env Var Leakage via Forward-to-Process-Env Pattern

**Issue:** Tier 3 and Tier 4 forward SecretStr credentials to process environment (`os.environ["OPENROUTER_API_KEY"]`) so LightRAG closures see the key on first invocation.

**Files:**
- `tier-3-graph/main.py:275-278`
- `tier-4-multimodal/main.py:287-289`

**Risk:** Process environment variables are visible to any subprocess or debugger attached to the process. This is safer than passing the key inline to every LightRAG call, but still exposes the credential in memory longer than necessary.

**Current Mitigation:**
- Keys are only forwarded immediately before RAG instantiation (not at startup)
- Only in CLI paths where the user invoked the tier directly
- The alternative (threading through LightRAG's entire public API) is infeasible given the complexity

**Recommendations:**
- Prefer `.get_secret_value()` at the call site over process env forwarding where possible
- Consider using LightRAG's upcoming async credentials API (if available) to pass credentials scoped to a single call
- Document this pattern in security section of main README

---

### API Key Validation Too Late in Initialization

**Issue:** API key existence checks happen AFTER creating expensive objects (CostTracker, RAG instance builders), delaying fast-fail for missing credentials.

**Files:**
- `tier-3-graph/main.py:265-278` — Key check happens after `_build_parser()` but before object creation ✓ Correct
- `tier-4-multimodal/main.py:262-289` — Key check happens early; subsequent forwarding is safe ✓ Correct
- `tier-2-managed/main.py` — Key check is post-setup ⚠️ Risk

**Impact:** On fresh checkout with missing .env, user sees verbose import errors instead of a friendly "OPENROUTER_API_KEY required" message.

**Fix Approach:** Move all API key validation to the very top of `amain()` before any SDK imports or object instantiation.

---

## Performance Bottlenecks

### Sequential PDF Upload in Tier 2

**Issue:** `cmd_ingest()` in `tier-2-managed/main.py:140-200` uploads PDFs sequentially with no parallelism, even though Gemini File Search API supports concurrent uploads.

**Files:** `tier-2-managed/main.py:180` (single loop with `upload_with_retry()`)

**Cause:** Conservative pattern chosen to avoid quota exhaustion and 503 storms (Pattern 2 from 129-RESEARCH.md). Gemini File Search has aggressive rate limiting.

**Improvement Path:**
- Benchmark parallel upload with batch_size=2-3 to find safe concurrency ceiling
- Implement async concurrent uploads with semaphore to cap parallelism
- Expected gain: 3-5x ingest speedup for 100-PDF corpus (currently ~5-10 min)

---

### Re-Tokenization of PDFs on Every Ingest

**Issue:** `_count_pdf_tokens()` in `tier-2-managed/main.py:100-132` re-tokenizes full PDF text on every ingest run to estimate synthetic indexing cost, even when PDFs have not changed.

**Files:** `tier-2-managed/main.py:100-132`, called from `cmd_ingest()` per PDF

**Cause:** No caching of token counts across runs. File-size proxy fallback (when PyMuPDF unavailable) is fast, but full extraction + tokenization takes ~100ms/PDF.

**Improvement Path:**
- Cache estimated token counts in a per-PDF metadata file (e.g., `.tier2_token_cache.json`)
- Only re-tokenize if file mtime has changed
- Expected gain: 10-15s per re-ingest (negligible for one-time setup; matters for CI cycles)

---

### Graph Construction Latency in Tier 3 on Large Corpora

**Issue:** LightRAG's entity extraction + graph construction is synchronous and single-threaded in `tier-3-graph/rag.py` despite the corpus being >100 PDFs.

**Files:** `tier-3-graph/rag.py`, calls via `tier-3-graph/main.py:181`

**Cause:** LightRAG's public API does not expose batch processing or incremental graph updates. Each PDF extraction is a full round-trip to LLM.

**Improvement Path:**
- Profile entity extraction latency to identify the LLM-call bottleneck (suspected: OpenRouter batch inference latency)
- Consider caching entity extraction results per PDF by content hash to avoid re-processing on retries
- Expected gain: 2-3x with caching; 5-10x if LightRAG upstream adds batch entity extraction in future releases

---

### RAGAS Scoring Batch Serialization

**Issue:** `score_query_log()` in `evaluation/harness/score.py` builds the full EvaluationDataset in memory before calling `evaluate()`, even with `batch_size=10` partitioning applied to the judge LLM calls.

**Files:** `evaluation/harness/score.py:208-218`

**Cause:** RAGAS requires full dataset upfront to shuffle and batch. With 30 questions per tier × 3 metrics, the in-memory graph of sample objects is large.

**Improvement Path:**
- Profile memory usage on full corpus (100+ PDFs × 30 questions = 3K+ samples)
- If memory-constrained, implement streaming batch scoring (load N samples, score, write results, repeat)
- Current 10-sample batching is conservative and safe

---

## Fragile Areas

### Tier 5 Agent Truncation Not Surfaced to User

**Component/Module:** `tier-5-agentic/agent.py` and `tier-5-agentic/main.py`

**Files:** `tier-5-agentic/agent.py`, `tier-5-agentic/main.py`

**Why Fragile:** OpenAI Agents SDK enforces a `max_turns` limit. When an agent exceeds this, the response is truncated with an internal flag set, but the CLI does not surface a warning to the user. The query appears to complete normally but with a partial answer.

**Safe Modification:** Add an explicit check in `main.py` after agent execution to detect `max_turns_exceeded` via the response metadata and print a yellow warning to console. Log the exact turn count vs. limit.

**Test Coverage:** No explicit test for agent truncation behavior. Add a test in `tier-5-agentic/tests/test_agent.py` that mocks OpenAI SDK to return a max-turns-exceeded response and verifies the warning is surfaced.

---

### Gemini File Search Grounding Metadata Volatility

**Component/Module:** `tier-2-managed/query.py` and `evaluation/harness/adapters/tier_2.py`

**Files:** `tier-2-managed/query.py:40-60`, `evaluation/harness/adapters/tier_2.py:104-113`

**Why Fragile:** Gemini's `grounding_metadata` field is optional and may be `None` if the model decides the corpus does not contain relevant information. Current code defends with `getattr` fallbacks, but this is brittle when the response structure changes in SDK updates.

**Safe Modification:** Extract grounding parsing into a dedicated function with explicit type checks:
```python
def _extract_contexts_from_grounding(resp, safe=True):
    """Extract context list from Gemini response.
    
    Args:
        resp: GenerateContentResponse object
        safe: If True, return [] on any structure mismatch
    """
    # Defensive parsing with explicit checks
```

**Test Coverage:** Add unit tests in `tier-2-managed/tests/` that mock responses with missing/malformed grounding_metadata and verify the parser handles all cases.

---

### Loose Exception Handling in Evaluation Harness

**Component/Module:** `evaluation/harness/adapters/` (all tiers)

**Files:** `evaluation/harness/adapters/tier_1.py:132`, `evaluation/harness/adapters/tier_3.py:140`, `evaluation/harness/adapters/tier_5.py:145`

**Why Fragile:** Broad `except Exception` clauses swallow unexpected errors without logging. When a tier fails in production evaluation, the record persists with a generic error message and no traceback.

**Safe Modification:** Replace broad exception handlers with specific exception types and add structured logging:
```python
try:
    # tier-specific query logic
except ImportError as e:
    console.print(f"[red]Import error (check extras installed): {e}[/red]")
    return EvalRecord(..., error=f"import:{type(e).__name__}:{str(e)[:100]}")
except ValueError as e:
    console.print(f"[red]Value error (check config): {e}[/red]}")
    return EvalRecord(..., error=f"value:{str(e)[:100]}")
except Exception as e:
    console.print(f"[red]Unexpected error: {type(e).__name__}: {e}[/red]")
    import traceback
    traceback.print_exc()
    return EvalRecord(..., error=f"unexpected:{type(e).__name__}")
```

**Test Coverage:** Mock each error path and verify correct error messages are recorded in EvalRecord.

---

## Scaling Limits

### Memory Usage in LightRAG Graph Construction (Tier 3)

**Current Capacity:** Tested on 100-PDF corpus (~25MB text). Graph construction completes in ~2-3 min with moderate memory overhead.

**Limit:** At 500+ PDFs or 100MB+ text, entity extraction + graph building will require significantly more memory for the KG structure. LightRAG's in-memory graph representation may hit memory limits on typical dev machines (8GB RAM).

**Scaling Path:**
- Profile memory growth as corpus size increases
- Consider persistent graph storage backend (LightRAG supports alternative backends beyond in-memory)
- Implement incremental graph updates (currently full rebuild required)
- Expected support: 500-1K PDFs with 16GB RAM

---

### Concurrent Tier Evaluation Limits

**Current Capacity:** Evaluation harness runs tiers sequentially (run.py → score.py). Per-tier query execution is async but tier loop is serial.

**Limit:** With 30 questions × 4 tiers × ~10s/query (average), full eval takes ~20 minutes. Running tiers in parallel would require careful CostTracker isolation to avoid collision.

**Scaling Path:**
- Implement parallel tier execution with separate CostTracker per tier
- Test CostTracker thread safety (currently assumes single-threaded access)
- Expected improvement: 4-5x faster evaluation (4-6 min instead of 20 min)

---

### Token Usage Tracking Under High Concurrency

**Current Capacity:** CostTracker is a simple in-memory dict that accumulates token counts. Single-threaded async workloads are safe.

**Limit:** If RAGAS evaluation shifts to true parallelism (multiple judge LLM calls per batch), concurrent tracker updates may lose data due to race conditions.

**Scaling Path:**
- Add thread-safe locks to CostTracker.record_llm() and record_embedding()
- Add tests for concurrent updates
- Consider atomic operations or a queue-based approach for distributed evaluation

---

## Dependencies at Risk

### LightRAG Version Lock at 1.4.15

**Risk:** Locked to `lightrag-hku==1.4.15` in `pyproject.toml:47`. Future patches may break the internal `_ensure_lightrag_initialized()` API that Tier 3 and Tier 4 depend on.

**Impact:** Tier 3 and Tier 4 will fail to initialize if LightRAG removes or refactors this private API.

**Migration Plan:**
- File upstream issue with LightRAG maintainers requesting public initialization API
- If unavailable, fork and maintain a local copy of LightRAG with the necessary API surface
- Alternatively, implement a wrapper that abstracts initialization details

---

### RAGAS Rapid Iteration Risk

**Risk:** `ragas>=0.4.3,<0.5` allows any 0.4.x patch. The three bugs found in 0.4.3 suggest rapid churn in the upstream project.

**Impact:** Future patches may introduce new breaking changes. The current workarounds (alias methods, exception guards) may not apply to 0.4.4+.

**Migration Plan:**
- Pin to `ragas==0.4.3` if the patch release is stable (once confirmed after production use)
- Set up automated testing against 0.4.4-rc when available to catch regressions early
- Consider alternative eval frameworks (e.g., DeepEval, custom metrics) if RAGAS continues to break

---

### google-genai SDK Updates

**Risk:** `google-genai>=1.73,<2` allows any 1.x patch. Tier 2 eval adapter heavily depends on internal response structure (`grounding_metadata`, `usage_metadata`).

**Impact:** SDK update could change response fields or remove optional attributes.

**Migration Plan:**
- Test against newer 1.x patches immediately after release
- Implement defensive field accessors with fallbacks (already in place, but could be strengthened)
- Monitor google-genai changelog for breaking changes to response types

---

## Missing Critical Features

### No Retry Logic for LLM Calls

**Problem:** Evaluation harness and tier queries make direct LLM calls without exponential backoff. Transient 429/503 errors cause immediate failure instead of retry.

**Blocks:** Full evaluation runs on unreliable networks or when LLM providers have brief outages.

**Impact:** 30-question eval that hits a 503 error on question #27 must be re-run from scratch, losing 27 queries' worth of cost.

**Fix Approach:** Implement a decorator `@with_exponential_backoff(max_retries=3)` in `shared/llm.py` and apply to all LLM-call sites. Cap retries at 3 to avoid runaway costs on permanent errors.

---

### No Incremental Evaluation State

**Problem:** Evaluation harness does not checkpoint progress. If a multi-tier eval run fails partway through, re-running starts from the beginning.

**Blocks:** Production evaluation runs where interrupts/crashes are possible.

**Impact:** Large eval runs (500+ questions × 5 tiers) lose all progress on failure.

**Fix Approach:**
- Persist per-tier checkpoint file (e.g., `evaluation/results/.tier_N_checkpoint.json`) with completed question IDs
- On restart, skip already-completed records
- Clear checkpoint on successful full run

---

### No Validation Schema for Golden Q&A

**Problem:** `evaluation/golden_qa.json` is loaded and indexed without schema validation. Invalid entries silently skip or produce confusing errors.

**Blocks:** Early detection of malformed golden Q&A during harness init.

**Impact:** If a golden Q&A entry is missing `expected_answer`, RAGAS scoring silently records empty reference and produces meaningless metrics.

**Fix Approach:** Add a Pydantic schema for golden Q&A entries and validate on load in `score.py:_load_golden_qa_index()`.

---

## Test Coverage Gaps

### Tier 4 Dual-Mode Adapter Not Fully Tested

**Untested Area:** `evaluation/harness/adapters/tier_4.py` dual-mode logic (cached vs. library fallback).

**Files:** `evaluation/harness/adapters/tier_4.py:47-146`

**Risk:** Cached mode is primary (requires user pre-capture). Library fallback is rarely exercised in test environments. If cached mode fails, fallback path may have silent errors.

**Priority:** High

**Fix Approach:**
- Add unit test that mocks cached QueryLog and verifies record lookup
- Add unit test for CachedTier4Miss exception
- Add integration test for library mode with mocked RAGAnything

---

### Agent Truncation Path

**Untested Area:** Max-turns-exceeded behavior in Tier 5 agent.

**Files:** `tier-5-agentic/agent.py`, `tier-5-agentic/main.py`

**Risk:** If agent truncates, the CLI does not surface a warning. User thinks they got a full answer.

**Priority:** Medium

**Fix Approach:** Add test that mocks OpenAI agent response with max_turns_exceeded flag and verifies CLI outputs a warning.

---

### Gemini Grounding Metadata Variations

**Untested Area:** All possible grounding_metadata structures from Gemini API.

**Files:** `tier-2-managed/query.py:40-60`, test suite in `tier-2-managed/tests/`

**Risk:** If grounding structure changes in SDK update, `to_display_chunks()` may fail to extract contexts but will silently return empty list.

**Priority:** Medium

**Fix Approach:**
- Add unit tests with mocked Gemini responses covering:
  - grounding_metadata = None
  - grounding_metadata with empty document list
  - grounding_metadata with missing/malformed field names
- Verify contexts are extracted correctly or empty list is returned gracefully

---

### End-to-End Evaluation Without Live API Calls

**Untested Area:** Full evaluation pipeline with mocked tier adapters.

**Files:** `evaluation/harness/run.py`, `evaluation/harness/score.py`

**Risk:** Integration between run.py and score.py is only tested live. Offline testing would catch breaking changes.

**Priority:** Medium

**Fix Approach:**
- Create fixtures that mock all tier adapters to return canned EvalRecord objects
- Add test that runs harness.run() → harness.score() → harness.compare() end-to-end with mocks
- Verify output files are created and metrics are computed

---

*Concerns audit: 2026-05-04*
