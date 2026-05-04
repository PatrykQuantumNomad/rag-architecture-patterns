# Stack Research

**Domain:** RAG eval pipeline — fix-only milestone (Tier 4 graphml regen + Tier 5 empty_contexts diagnosis)
**Researched:** 2026-05-04
**Confidence:** HIGH (Context7-verified for openai-agents, lightrag, openinference; official MinerU docs; PyPI release dates checked)

---

## Recommended Stack — Additions Only

The existing pyproject.toml stack is sound. The two failures do **not** need new RAG frameworks, new providers, new vector stores, or a different agent SDK. They need (a) one CLI tool that already exists in the optional install path (`mineru`), and (b) a five-line code change inside the eval adapter that uses APIs already present in the pinned `openai-agents` version.

### Core Technologies (Existing — DO NOT change)

| Technology | Pinned Version | Purpose | Why keep |
|------------|----------------|---------|----------|
| `lightrag-hku` | `==1.4.15` | Tier 3/4 KG store | Pinned to validated graphml format; bumping risks index-format break. Context7 docs match this version's `initialize_storages()` contract. |
| `raganything` | `==1.2.10` | Tier 4 multimodal wrapper | Locked invariants in `tier-4-multimodal/rag.py` (EMBED_DIMS=1536, working_dir, parse_method "auto") all assume this version. |
| `openai-agents[litellm]` | `==0.14.6` | Tier 5 agent runtime | `RunResult.new_items` / `ToolCallOutputItem` API (the Tier 5 fix vehicle) is stable in 0.14.x per Context7 (`/websites/openai_github_io_openai-agents-python`). 0.15.x adds `ModelRefusalError` and is non-breaking but unnecessary. |
| `ragas` | `>=0.4.3,<0.5` | Eval scoring | Existing pin; the patches in `f0cd134`/`186a9c2`/`fb397f0` are 0.4.3-specific. Bumping invalidates those fixes. |
| `chromadb` | `>=1.5.8,<2` | Tier 1/5 vector store | Tier 5 reads through Tier 1's collection (Pitfall 9 invariant). |

### New Additions — Tier 4 Path (Outside-Sandbox Ingest)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `mineru[core]` | `>=2.5,<4` (current 3.1.6 on PyPI 2026-04-28; pin floor at 2.x because `MINERU_PARSE_METHOD` was renamed to `PARSE_METHOD` in 2.0) | PDF → structured JSON parser | Required by `raganything==1.2.10`'s `MineruParser._run_mineru_command`. Already declared as a transitive of `raganything[all]` but pulled in explicitly so the **outside-sandbox** ingest script can call the `mineru` CLI directly without round-tripping through the Python API. |
| `paddlepaddle` | `>=2.6` (CPU build) — only if forcing legacy backend | OCR backend for `--backend pipeline` | Optional. **NOT needed** with MinerU 2.x default (`vlm-auto-engine`). Skip unless ingest fails on scanned-PDF figures. Confidence MEDIUM — verify after first ingest run. |

**Confidence:** HIGH for `mineru` pin (verified PyPI 2026-04-28 release; CLI flags in `opendatalab.github.io/MinerU/usage/cli_tools/`). MEDIUM for `paddlepaddle` (only conditional; do not preinstall).

### New Additions — Tier 5 Path (Empty-Contexts Diagnosis)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `openinference-instrumentation-openai-agents` | `==1.4.2` (latest on PyPI 2026-04-29) | OpenTelemetry tracing of every Runner step incl. ToolCallItem / ToolCallOutputItem with their JSON payloads | **Diagnostic-only.** Install as a dev-only extra `evaluation-debug`. Not a runtime dep — exporting OTLP to Phoenix locally is the fastest way to confirm "did the agent call the tool, did the tool return something, what did the tool return?" without rebuilding the harness. |
| `arize-phoenix` | `>=6,<8` | Local OTLP collector + UI to render the spans the instrumentor emits | Same as above — dev-only. Run `phoenix serve` → spans visible at `http://127.0.0.1:6006`. |
| `opentelemetry-sdk` | `>=1.30,<2` | Standard tracer provider | Pulled in transitively but pin explicitly so version drift doesn't break the instrumentor. |
| `opentelemetry-exporter-otlp-proto-http` | `>=1.30,<2` | OTLP/HTTP exporter to Phoenix | Required by the OpenInference quickstart pattern. |

**Why instrumentation, not a new framework:** The Tier 5 root cause is unknown — it could be (1) `retrieved_contexts=[]` is *intentional* in `evaluation/harness/adapters/tier_5.py:125` (Pitfall 9 design choice — agent self-cites), or (2) the agent isn't calling tools at all because of a known LiteLLM+Gemini compat issue, or (3) tools are called but return empty results from ChromaDB. OTel spans distinguish these in one run. **Adding telemetry beats swapping the SDK** because the SDK is fine; the question is *what is happening*.

**Confidence:** HIGH (Context7 `/arize-ai/openinference` confirms install + import paths; openai-agents docs confirm `add_trace_processor` is the integration point; `set_tracing_disabled(disabled=True)` in `tier-5-agentic/agent.py:43` will need to be **conditional** so OTel can capture spans when debugging).

### Supporting Tools — Already Present, Used Differently

| Tool | Purpose | Notes |
|------|---------|-------|
| `litellm` | LiteLLM debug logging | Call `litellm._turn_on_debug()` from a `--debug` flag in eval CLI to see raw OpenRouter request/response payloads. No version bump needed — already `>=1.0,<2`. |
| `pyvis` | Optional graphml visual sanity-check | `pip install pyvis>=0.3` — generates an HTML visualization of `graph_chunk_entity_relation.graphml` post-ingest so you can confirm the regen worked before running RAGAS. Confidence MEDIUM — nice-to-have, skip if disk space tight. |

---

## Installation

```bash
# Tier 4 — outside-sandbox ingest host (one-time)
# Run on a machine with >=8GB free RAM, ~5GB disk for MinerU model cache.
uv pip install -e '.[tier-4]'                  # existing extras (raganything==1.2.10, lightrag-hku==1.4.15)
uv pip install 'mineru[core]>=2.5,<4'          # CLI binary on PATH
export MINERU_MODEL_SOURCE=huggingface         # or 'modelscope' if HF blocked
mineru --help                                  # verify

# Tier 5 — debug-only extras (developer machines, not CI)
uv pip install \
    'openinference-instrumentation-openai-agents==1.4.2' \
    'arize-phoenix>=6,<8' \
    'opentelemetry-sdk>=1.30,<2' \
    'opentelemetry-exporter-otlp-proto-http>=1.30,<2'
```

**Suggested pyproject.toml addition** (one new optional-dependency group, no changes to existing groups):

```toml
[project.optional-dependencies]
# … existing tier-1 through tier-5 unchanged …

debug-tier5 = [
  "rag-architecture-patterns[tier-5,evaluation]",
  "openinference-instrumentation-openai-agents==1.4.2",
  "arize-phoenix>=6,<8",
  "opentelemetry-sdk>=1.30,<2",
  "opentelemetry-exporter-otlp-proto-http>=1.30,<2",
]

ingest-tier4 = [
  "rag-architecture-patterns[tier-4]",
  "mineru[core]>=2.5,<4",
]
```

---

## MineRU Outside-Sandbox CLI Pattern (Tier 4 Fix)

The sandbox blocks `OMP_NUM_THREADS` shmem (Phase 138/139 evidence in blog repo). Run **two-stage**: parse PDFs with `mineru` CLI on the host, then load the parsed JSON into RAG-Anything.

### Stage A — Parse 100 PDFs to disk (host machine)

```bash
# From repo root, on a host where MinerU works (no sandbox shmem block).
# CPU-only, default 'auto' parse method (handles digital + scanned PDFs).
mineru \
  -p dataset/papers/ \
  -o tier-4-multimodal/output/mineru-raw/ \
  -m auto \
  -b pipeline           # 'pipeline' = CPU-friendly; omit on GPU host for vlm-auto-engine

# Verify per-paper output:
#   tier-4-multimodal/output/mineru-raw/<paper_id>/<paper_id>_content_list.json
#   tier-4-multimodal/output/mineru-raw/<paper_id>/images/*.{png,jpg}
ls tier-4-multimodal/output/mineru-raw/ | wc -l   # expect 100
```

**CLI flags verified against `opendatalab.github.io/MinerU/usage/cli_tools/`:**
- `-p / --path PATH` — input file or directory (required) — accepts a directory of PDFs
- `-o / --output PATH` — output directory (required)
- `-m / --method` — `auto` (default), `txt`, or `ocr`
- `-b / --backend` — `pipeline` (CPU-friendly) | `vlm-auto-engine` (default, needs GPU for speed)
- `-l / --lang` — language hint, defaults to auto-detect; pass `en` for ML papers

**Environment variables (verified from official docs):**
- `MINERU_MODEL_SOURCE=huggingface|modelscope` — switch model registry on first run (~3-5GB download)
- `MINERU_PDF_RENDER_THREADS` — bound CPU concurrency (set to 4 on a 16-core host to avoid OOM on 100 PDFs)
- `MINERU_PROCESSING_WINDOW_SIZE` — memory/throughput tuning

### Stage B — Wipe corrupt graphml + re-ingest from MinerU output

The current corruption lives at `rag_anything_storage/tier-4-multimodal/` — that directory is empty in the working tree (verified 2026-05-04). RAG-Anything will rebuild on first ingest, but the issue is the **previous** corrupt `graph_chunk_entity_relation.graphml` was checked in or cached on the local box.

```bash
# 1. Hard-reset the storage directory.
rm -rf rag_anything_storage/tier-4-multimodal/

# 2. Re-ingest from the parsed MinerU JSON (NOT from raw PDFs again — saves a second
#    MinerU pass + LLM extraction cost). The existing tier-4-multimodal/main.py
#    --ingest path can be extended with a --from-mineru-output flag, OR you run
#    eval_capture.py which already calls process_document_complete with doc_id.
python tier-4-multimodal/main.py --ingest --yes

# 3. Sanity-check graphml exists and is non-empty.
test -s rag_anything_storage/tier-4-multimodal/graph_chunk_entity_relation.graphml \
  && echo "OK: graphml regenerated" \
  || echo "FAIL: graphml missing or empty"

# 4. THEN run eval_capture.py — the _ensure_lightrag_initialized() call at line 99
#    will now find a clean graph.
python tier-4-multimodal/scripts/eval_capture.py --yes
```

**Storage files LightRAG/RAG-Anything maintain (verified Context7 `/hkuds/lightrag`):**
- `graph_chunk_entity_relation.graphml` — NetworkX graph (the file that was corrupt)
- `kv_store_doc_status.json` — per-document ingest status (delete to force re-ingest of all)
- `kv_store_full_docs.json` — raw chunked docs
- `kv_store_text_chunks.json` — chunk → entity index
- `vdb_*.json` — NanoVectorDB sidecars (1536-dim — must match `EMBED_DIMS` constant)

**Confidence:** HIGH. Recovery procedure (delete + re-ingest) verified by multiple sources including LightRAG docs and community examples.

---

## Tier 5 Empty-Contexts Diagnosis Pattern

### What the code currently does

```python
# evaluation/harness/adapters/tier_5.py:125
return EvalRecord(
    …,
    retrieved_contexts=[],  # Pitfall 9 honest empty — agent self-cites
    …,
)
```

**This is the bug.** It is *intentional* by Pitfall 9 design, but RAGAS treats empty contexts as `nan_reason='empty_contexts'` (per `evaluation/harness/score.py` Pitfall 2 short-circuit), so 30/30 NaN is the **expected** output of the current adapter, not a runtime failure.

### The fix — extract tool outputs from RunResult

The OpenAI Agents SDK exposes the full sequence of tool calls and their outputs via `result.new_items`. Verified pattern from Context7 `/websites/openai_github_io_openai-agents-python`:

```python
# evaluation/harness/adapters/tier_5.py — proposed change

from agents.items import ToolCallOutputItem  # NEW import

# … inside run_tier5, after Runner.run(...) succeeds:
result = await Runner.run(agent, question, max_turns=max_turns)
answer = result.final_output or ""
usage = result.context_wrapper.usage

# Lift tool outputs into retrieved_contexts. Each ToolCallOutputItem.output is
# whatever the @function_tool returned — for search_text_chunks that's
# list[dict] with 'snippet' keys. Stringify into the RAGAS context shape.
contexts: list[str] = []
for item in result.new_items:
    if isinstance(item, ToolCallOutputItem):
        out = item.output
        if isinstance(out, list):  # search_text_chunks returns list[dict]
            for hit in out:
                snippet = (hit or {}).get("snippet") if isinstance(hit, dict) else None
                if snippet:
                    contexts.append(snippet)
        elif isinstance(out, dict):  # lookup_paper_metadata
            abstract = out.get("abstract")
            if abstract:
                contexts.append(abstract)
        elif isinstance(out, str) and out:
            contexts.append(out)

return EvalRecord(
    …,
    retrieved_contexts=contexts,  # Pitfall 9 RESOLVED — agent retrieval surfaced
    …,
)
```

**Verified API surface (Context7 — confidence HIGH):**
- `RunResult.new_items: list[RunItem]` — present in 0.14.x
- `ToolCallOutputItem(raw_item, output, type='tool_call_output_item', tool_origin, agent)` — `output` field carries whatever the tool callable returned
- `ItemHelpers.text_message_output(item)` — only needed for `MessageOutputItem`; not used here

### Backstop diagnosis — if even after the fix `contexts` is empty

That means the agent *isn't calling tools*. Run with OpenInference instrumentation:

```python
# evaluation/harness/score.py or a new evaluation/harness/debug_tier5.py
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Re-enable tracing (tier-5-agentic/agent.py disables it by default for prod)
from agents import set_tracing_disabled
set_tracing_disabled(disabled=False)

provider = trace_sdk.TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(
    OTLPSpanExporter("http://127.0.0.1:6006/v1/traces")  # Phoenix local
))
OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

# … then run one or two questions and inspect spans in Phoenix UI.
```

**Spans will show:**
- Was `search_text_chunks` invoked? (if not → LLM/model issue, see below)
- What args did it receive? (if `query=""` → prompt issue)
- What did it return? (if `[]` → ChromaDB collection issue, check `open_collection(reset=False)` matches Tier 1's path)
- Did the LLM make additional turns or stop after the first tool call?

### Known LiteLLM + Gemini 2.5 Flash compatibility caveat

Verified via `BerriAI/litellm#16651` and `openai/openai-agents-python#2257`: `gemini/gemini-2.5-flash` (and `openrouter/google/gemini-2.5-flash`) sometimes returns `finish_reason: "MALFORMED_FUNCTION_CALL"` silently when the JSON schema generated by `@function_tool` includes `Annotated[T, Field(...)]` defaults. **`tier-5-agentic/tools.py:108-121` uses exactly this pattern** with `k: Annotated[int, Field(default=5, ge=1, le=20, ...)] = 5`.

If OpenInference confirms tools aren't being invoked, the fix priority order is:

1. **Simplify the schema first** (cheapest): drop `Field(default=...)` from `k` and rely on Python's `= 5` default only. Confidence MEDIUM — fixes some but not all `MALFORMED_FUNCTION_CALL` reports.
2. **Switch model slug** (still cheap): try `openrouter/openai/gpt-4o-mini` or `openrouter/anthropic/claude-haiku-4.5` — both have proven tool-calling on OpenRouter per `openrouter.ai/docs/guides/features/tool-calling`. The `DEFAULT_MODEL` constant in `tier-5-agentic/agent.py:49` is the only swap point. Confidence HIGH.
3. **Bump openai-agents** (last resort): 0.14.7 has "fix: remove unset fields from calls to Responses API" and 0.15.0 introduces `ModelRefusalError` so refusals don't masquerade as `MaxTurnsExceeded`. Bump only if 1+2 don't fix and you have rerun budget. Confidence MEDIUM.

**Cost note:** swapping `gemini-2.5-flash` → `gpt-4o-mini` raises per-question cost ~3-4×. With 30 questions × ~$0.001 → ~$0.003 per Q at gpt-4o-mini, full re-run still under $0.20. Within the $1-3 budget.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `mineru` CLI outside sandbox | `unstructured`, `pymupdf`-only | Skip MineRU entirely if we drop multimodal — but then Tier 4 collapses into Tier 3 (graph-only LightRAG). Don't do this; the blog post premise is multimodal. |
| `openinference-instrumentation-openai-agents` for diagnosis | `logfire` (Pydantic) | logfire is a managed SaaS — adds an account dependency and external API key the constraint forbids. OpenInference + Phoenix is fully local. |
| `openinference-instrumentation-openai-agents` for diagnosis | Custom `add_trace_processor(...)` callback that pickles spans to JSON | Cheaper (no Phoenix install) but ~50 lines of hand-rolled code we'd need to test. OpenInference is the standard. Use only if Phoenix install fails. |
| Walking `RunResult.new_items` for contexts | `result.context_wrapper.run_data` introspection | The SDK doesn't expose tool-output collection on `context_wrapper` — `new_items` is the documented path. |
| Keep `openai-agents==0.14.6` | Bump to `==0.15.1` | 0.15.x adds `ModelRefusalError` and `error_handlers={"max_turns": ...}` hooks. Useful but neither is required for the empty_contexts fix. Defer. |
| Keep `gemini-2.5-flash` slug | Switch to `gpt-4o-mini` | Switch only if OpenInference traces show `MALFORMED_FUNCTION_CALL`. Default to keeping Gemini because the existing CostTracker pricing entries assume it. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `unstructured` / `langchain` document loaders for Tier 4 | They produce a different chunk schema than RAG-Anything 1.2.10 expects (`type/text/page_idx/img_path` shape per Context7). Plumbing a converter is more work than just running `mineru`. | `mineru` CLI |
| `pip install mineru` (no extras) | Default pulls only the model loader, not the parser pipeline. Causes `ImportError` on first `process_document_complete`. | `mineru[core]` (or `[all]` if disk allows) |
| `RAGAnything().check_parser_installation()` as a CI gate | The check requires the model cache to be downloaded, which takes 5+ minutes and ~5GB. Don't gate CI on it. | Run the check **once** post-install on the host machine; trust it thereafter. |
| `result.final_output` as the only context source for RAGAS | The agent self-cites in prose; `final_output` is the ANSWER, not the context. Using it as both makes `faithfulness` trivially 1.0 (answer is grounded in itself). | `result.new_items` walk for `ToolCallOutputItem.output` (the actual evidence). |
| Re-introducing `OPENAI_TRACING_KEY` to fix Tier 5 | Pitfall 8 already disabled platform tracing and we don't want OpenAI to receive traces of an OpenRouter run. | Local OpenInference + Phoenix. |
| Bumping `lightrag-hku` past 1.4.15 | Index dim/format break risk; HKUDS issue #2119 cited in `tier-4-multimodal/rag.py`. | Stay pinned. |
| Bumping `raganything` past 1.2.10 | Locked invariants in `rag.py` (parse_method "auto", working_dir, EMBED_DIMS=1536) tied to this version. | Stay pinned. |
| Adding `paddlepaddle` preemptively | ~2GB install, 90% of papers don't need OCR. | Add only if MinerU `auto` parse fails on a specific scanned PDF. |

---

## Stack Patterns by Variant

**If running Tier 4 ingest on a host WITH a CUDA GPU:**
- Use `mineru -p dataset/papers/ -o … -m auto` (omit `-b pipeline`)
- Default backend `vlm-auto-engine` is 5-10× faster
- Disk: still ~5GB for model cache
- RAM: ~6GB peak per PDF
- Confidence HIGH (verified `opendatalab.github.io/MinerU/`)

**If running Tier 4 ingest on a CPU-only host (most users):**
- Use `mineru -p … -o … -m auto -b pipeline`
- Set `MINERU_PDF_RENDER_THREADS=4` to bound RAM
- Expect ~30-60s per PDF on a modern laptop → ~50min for 100 PDFs (one-time cost)
- Confidence MEDIUM (community-reported timings; not benchmarked here)

**If running Tier 5 diagnosis without a Phoenix UI:**
- Skip `arize-phoenix`, use `ConsoleSpanExporter` only
- Spans print to stderr as JSON — grep for `tool_call` and `tool_call_output`
- Cheaper but harder to read for >5 questions
- Confidence HIGH (Context7-verified)

**If the OpenInference instrumentor reveals tools ARE being called and returning data, but contexts are still empty after the adapter fix:**
- Bug is in the `isinstance(item, ToolCallOutputItem)` walk — likely an SDK type-name change in 0.14.6 vs. 0.14.x docs
- Fallback: walk `result.new_items` and check `getattr(item, "type", None) == "tool_call_output_item"` instead of `isinstance`
- Confidence MEDIUM (defensive coding pattern; not yet observed)

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `openai-agents==0.14.6` | `litellm>=1.0,<2`, `openinference-instrumentation-openai-agents==1.4.2` | Verified — both pinned versions in PyPI metadata declare `openai-agents` compat range. |
| `mineru>=2.5` | `raganything==1.2.10` | RAG-Anything's `MineruParser._run_mineru_command` uses MinerU 2.x CLI flags (`-p`, `-o`, `--device`); pre-2.0 used a config-file pattern that's now removed. Pin floor at 2.x. |
| `lightrag-hku==1.4.15` | `raganything==1.2.10` | Locked invariant — RAG-Anything 1.2.10 imports `lightrag.QueryParam` and depends on `LightRAG.aquery(only_need_context=True)`. Both stable in 1.4.15. |
| `arize-phoenix>=6` | `opentelemetry-sdk>=1.30,<2` | Phoenix 6.x bumped its OTel floor; pre-6 versions cap at otel-sdk 1.27. |
| `openinference-instrumentation-openai-agents==1.4.2` | `opentelemetry-sdk>=1.30,<2`, `openai-agents>=0.10` | PyPI metadata lower bound; 0.14.6 well within range. |
| `ragas==0.4.3` | (unchanged) | Already patched for two upstream quirks (`fb397f0`, `186a9c2`). Do not touch. |

---

## Sources

**Context7 (HIGH confidence):**
- `/websites/openai_github_io_openai-agents-python` — `RunResult.new_items`, `ToolCallOutputItem` dataclass, `add_trace_processor`, `set_tracing_disabled`, `MaxTurnsExceeded`, `error_handlers={"max_turns":...}` (0.15+ only)
- `/openai/openai-agents-python` — same SDK, code-level snippets confirming `items.py` exports
- `/hkuds/lightrag` — `initialize_storages`, `NetworkXStorage` graphml file layout, `JsonDocStatusStorage`, working_dir isolation
- `/arize-ai/openinference` — `OpenAIAgentsInstrumentor`, install + tracer-provider wiring

**Official docs (HIGH confidence):**
- `https://opendatalab.github.io/MinerU/usage/cli_tools/` — full CLI flag list (`-p`, `-o`, `-b`, `-m`, `-l`, `-s`, `-e`, `-f`, `-t`); env vars `MINERU_MODEL_SOURCE`, `MINERU_PDF_RENDER_THREADS`, `MINERU_PROCESSING_WINDOW_SIZE`
- `https://opendatalab.github.io/MinerU/usage/quick_usage/` — `MINERU_MODEL_SOURCE=huggingface|modelscope`
- `https://github.com/HKUDS/RAG-Anything` — README confirms MinerU 2.x flag-based config replaces config files
- `https://pypi.org/project/openai-agents/` — 0.15.1 latest as of 2026-05-02; `[litellm]` extra confirmed
- `https://pypi.org/project/openinference-instrumentation-openai-agents/` — 1.4.2 latest as of 2026-04-29
- `https://github.com/HKUDS/RAG-Anything/releases` — 1.2.10 release date 2026-03-24

**GitHub issues (MEDIUM confidence — confirms known LiteLLM+Gemini issues but NOT the root cause of THIS empty_contexts bug):**
- `openai/openai-agents-python#2257` — Gemini structured output + tools incompat (does not apply — Tier 5 has no `output_type`)
- `openai/openai-agents-python#1575` — Gemini structured output + tools incompat (same as above, predecessor)
- `BerriAI/litellm#16651` — `MALFORMED_FUNCTION_CALL` silent failures on `gemini-2.5-flash` (POSSIBLY applies — instrument to confirm)
- `openai/openai-agents-python#1846` — `tool_choice='mytool'` + LiteLLM streaming pydantic error (does not apply — Tier 5 doesn't force tool_choice)

**Web search (LOW confidence — used only as breadcrumbs to authoritative sources, not cited as fact):**
- "openai-agents-python LitellmModel openrouter gemini function tool not called debug"
- "lightrag graphml NetworkXStorage corrupt rebuild"

---
*Stack research for: RAG eval pipeline fix-only milestone (Tier 4 graphml + Tier 5 empty_contexts)*
*Researched: 2026-05-04*
