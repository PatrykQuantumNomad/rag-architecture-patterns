# Tier 5: Agentic RAG (OpenAI Agents SDK + LiteLLM)

> Autonomous tool use over the enterprise KB. The LLM picks tools (text search via Tier 1's ChromaDB, paper metadata lookup), iterates up to `max_turns=10`, and self-cites paper IDs in the answer.

This is the **agentic chapter** of the repo. Tiers 1–4 are deterministic single-shot pipelines: embed → retrieve → LLM-answer. Tier 5 hands the steering wheel to the model. The agent decides *which* tool to call, *with what arguments*, and *whether* to call another tool after seeing the first result. The loop is bounded by `max_turns=10` (ROADMAP-locked safety cap) and by the LiteLLM cost ceiling for each turn.

The win narrative is **cross-tool reasoning**: questions like *"Cite Lewis 2020 with the full author list"* require the agent to call `search_text_chunks("retrieval-augmented generation")` to find relevant chunks (which surface a `paper_id`), then call `lookup_paper_metadata(paper_id)` to verify the canonical title + authors. A single-shot Tier 1 query cannot compose those two retrieval surfaces; the planner LLM does it for free.

If you only run one tier in this repo, run **Tier 1** first. Tier 5 is the *meta-tier* that reuses Tier 1's index — it does not (yet) reach into Tier 3's graph or Tier 4's multimodal index. See [Reused by](#reused-by) below for the deferred enhancement.

---

## ⚠️ Pre-requisite: Tier 1 must be ingested first

Tier 5 reads Tier 1's ChromaDB index at `chroma_db/tier-1-naive/`. If you haven't run Tier 1 yet, do that first:

```bash
cd ../tier-1-naive
python main.py --ingest
```

**Tier 5 ONLY READS Tier 1's index — it never writes to it.** This is a hard code invariant: `tier-5-agentic/tools.py::_get_collection` calls `open_collection(reset=False)` always. Mutating Tier 1's path from Tier 5 would corrupt the index for both tiers and silently break Tier 1's retrieval (Pitfall 9 from `130-RESEARCH.md`). The fast-fail in `main.py` exits with code 2 and a friendly red error if `chroma_db/tier-1-naive/` is missing — there is no auto-ingest fallback by design (file ownership: Tier 1 owns its index).

---

## Quickstart

```bash
cd tier-5-agentic
uv pip install -r requirements.txt          # or: cd .. && uv pip install -e ".[tier-5]"
cp ../.env.example ../.env                  # then edit and set OPENROUTER_API_KEY
python main.py                               # default: canned multi-tool probe (DPR vs RAG)
```

Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys); browse models at [openrouter.ai/models](https://openrouter.ai/models). The first `python main.py` runs the canonical multi-tool demo question through the agent — typically 2–4 turns, ~$0.005–0.015 cost, and a paper-id-cited answer. The agent ALWAYS prefers `search_text_chunks` first (per its `INSTRUCTIONS` system prompt) and `lookup_paper_metadata` second when verifying author lists.

---

## CLI reference

| Flag | Purpose | Default |
|------|---------|---------|
| `--query "..."` | Question for the agent. | canned multi-tool probe (DPR vs RAG) |
| `--max-turns N` | Hard cap on agent iterations. **ROADMAP-locked at 10.** Raising this is allowed but documented as out-of-spec. | 10 |
| `--model <slug>` | LiteLLM model slug for the agent's planner LLM. **Must start with `openrouter/`** (Pitfall 10). | `openrouter/google/gemini-2.5-flash` |

Invoking `python main.py` with **no flags** runs the canned multi-tool probe against `--max-turns 10` on `--model openrouter/google/gemini-2.5-flash`. Exit codes: `0` success, `2` fast-fail (missing key OR missing Tier 1 index), `3` truncation (`MaxTurnsExceeded` — answer is partial, cost is still recorded).

---

## Expected cost (vintage 2026-04)

| Operation | Tokens | Price | Cost |
|-----------|--------|-------|------|
| Per-query LLM (`gemini-2.5-flash`, ~5K in / 500 out across 2-4 turns) | ~5,500 | $0.30/1M in / $2.50/1M out | ~$0.0030 |
| Tool-call overhead (each tool roundtrip ≈ ~500 tokens) | ~2,000 | (LLM rate) | ~$0.0010 |
| Embeddings (only if a tool re-embeds — `search_text_chunks` does once per call) | minimal | $0.02/1M | ~$0.00001 |
| **Total per query (typical)** | — | — | **~$0.005–0.015** |
| **Worst case (10 turns, planner stuck in a tool-call loop)** | — | — | ~$0.05 |

Cost scales roughly linearly with the number of agent turns. The ROADMAP-locked cap (`max_turns=10`) is the worst-case ceiling: the agent CAN spend $0.05 if it iterates the full budget, but in practice the canned multi-tool probe finishes in 2–4 turns. Cost and latency are printed for every query via `shared.display.render_query_result` plus a `Latency:` footer line, and a per-run JSON is written to `evaluation/results/costs/tier-5-<timestamp>.json`. The cost is recorded per-query (single `record_llm` call against `result.context_wrapper.usage`), NOT per-turn — the OpenAI Agents SDK aggregates usage across turns into a single `Usage` object. To attribute cost per tool, see Phase 131's evaluation harness.

---

## Persistence

Tier 5 has **NO persistence directory of its own.** It reads:

- `chroma_db/tier-1-naive/` — Tier 1's ChromaDB collection (read-only; `reset=False` invariant from Pitfall 9).
- `dataset/manifests/papers.json` — shared paper metadata loaded via `shared.loader.DatasetLoader`.

It writes:

- `evaluation/results/costs/tier-5-<timestamp>.json` — per-query cost JSON via `shared.cost_tracker.CostTracker`.

That's it. No graph, no vdb, no working_dir. If you `rm -rf` everything Tier 5-shaped from disk, the only loss is the cost JSON history.

---

## Known weaknesses (deliberate)

These are not bugs — each is a deliberate scope cut motivating a future tier or future plan.

### Single-shot CLI (no conversation memory)

Each `python main.py --query "..."` invocation is independent. There is no thread-id / session-id / conversational memory. Multi-turn dialogue, follow-up clarifications, and persistent context windows are **Phase 131+ territory**. The Agents SDK supports threading natively; we deliberately did not wire it here to keep the Tier 5 code surface ≈300 LOC.

### No guardrails (input/output filters)

The Agents SDK ships an `InputGuardrail` / `OutputGuardrail` system. Tier 5 does NOT use it. Adversarial prompts (prompt-injection in retrieved chunks, jailbreak attempts) are handled by the planner LLM's native safety training only. **TIER-05 spec did not call for guardrails**, and adding them to the README's CLI surface would be misleading.

### Two tools only — no figure search, no graph traversal

Tier 5 exposes ONLY `search_text_chunks` (Tier 1's index) and `lookup_paper_metadata` (paper manifest). It does NOT call Tier 3's graph (`lightrag_storage/tier-3-graph/`) or Tier 4's multimodal index. **Open Q4 (deferred enhancement):** add a third tool that proxies Tier 4's `aquery` for image-grounded questions. That would let the agent route by question shape (text → Tier 1, multi-hop → Tier 3, figure-grounded → Tier 4) — true polyglot RAG. Out of scope for Phase 130; tracked as a Phase 131+ candidate.

### LitellmModel `openrouter/` prefix is mandatory (Pitfall 10)

`--model google/gemini-2.5-flash` (without the prefix) silently routes to native Vertex/PaLM and fails with a misleading authentication error. The `assert chosen.startswith("openrouter/")` guard in `build_agent` catches this at construction time. **Always pass the full `openrouter/<provider>/<model>` slug.**

### `max_turns=10` is a HARD cap

When the agent exceeds the cap, the Agents SDK raises `agents.exceptions.MaxTurnsExceeded`. `main.py` catches it, marks the run as truncated, and returns the agent's partial answer prefixed with `[truncated — agent exceeded max_turns=10]`. Cost is still recorded (best-effort: `getattr(exc, "usage", None)` per Pitfall 6 — accumulated `usage` may be `None` on some 0.x patch versions). Exit code `3` is distinct from the fast-fail `2` so CI can distinguish "configuration error" from "agent ran out of turns".

### Cost is recorded per-query, NOT per-turn

The SDK aggregates `Usage` (input_tokens / output_tokens) across all turns into a single `result.context_wrapper.usage` object. We `record_llm` once with that aggregate. Per-tool cost attribution would require hooking each tool's invocation; out of scope for TIER-05. Phase 131's evaluation harness can split by reading the model trace if it surfaces.

---

## Sample query

The default canned query is the **multi-hop DPR ↔ RAG probe** (verbatim in `tier-5-agentic/main.py` as `DEFAULT_QUERY`):

> *"Compare the dense retrieval approach in DPR (Karpukhin 2020) with the retrieval-augmented generation approach in RAG (Lewis 2020). What is the key architectural difference? Cite paper_ids."*

This is a known multi-hop case requiring tool composition:

1. Agent reads the question, recognizes "DPR" + "Karpukhin 2020" → calls `search_text_chunks("DPR dense retrieval Karpukhin")` → gets chunks referencing `paper_id=2004.04906`.
2. Agent calls `lookup_paper_metadata("2004.04906")` to verify the canonical title + author list.
3. Agent repeats for "RAG" / "Lewis 2020" → `paper_id=2005.11401`.
4. Agent synthesizes the answer: *"DPR (Karpukhin et al. 2020, paper_id 2004.04906) is the dense passage retriever; RAG (Lewis et al. 2020, paper_id 2005.11401) composes a DPR-style retriever with a generator. The key architectural difference is …"*

Single-shot Tier 1 cannot do this — its `top_k=5` typically picks chunks from one paper, not both. Tier 3's graph mode can synthesize via the entity edge, but does not verify author lists (no metadata tool). Tier 5 wins specifically when **the question asks for citations the planner LLM has to verify, not just retrieved**.

See [`tier-5-agentic/expected_output.md`](./expected_output.md) for a verbatim CLI snapshot of an actual agent run including the tool-call trace.

---

## Architecture

```
User question  -> Agent.run (LitellmModel: openrouter/google/gemini-2.5-flash)
               -> [LLM decides which tool to call]
               -> tool: search_text_chunks(query, k=5)
                  -> _embed_query(query)                   (OpenRouter openai/text-embedding-3-small)
                  -> open_collection(reset=False)           (chroma_db/tier-1-naive — read-only)
                  -> retrieve_top_k                         (ChromaDB cosine HNSW)
                  -> returns [{paper_id, page, snippet, similarity}, ...]
               -> tool: lookup_paper_metadata(paper_id)
                  -> DatasetLoader().papers() lookup
                  -> returns {paper_id, title, authors, year, abstract}
               -> [LLM iterates: more tool calls or final answer]
               -> max_turns=10 cap (raises MaxTurnsExceeded if hit)
               -> result.context_wrapper.usage -> CostTracker.record_llm
               -> shared.display.render_query_result(chunks=[])   (agent self-cites in answer text)
               -> tracker.persist() -> evaluation/results/costs/tier-5-<timestamp>.json
```

Tier 5's only async/sync boundary is the single `asyncio.run(amain)` at the bottom of `main.py`; every `Runner.run` and tool invocation stays async (Pitfall 8). The two tools are pure functions decorated with `@function_tool` — the SDK derives the JSON schema the planner sees from the type hints and `pydantic.Field` metadata, and the docstring becomes the `description` field. Module-level singletons (`_collection`, `_loader`, `_oai_client`) are lazy-initialized so non-live unit tests can import the module without an OpenRouter key.

---

## Reused by

**Phase 131 (RAGAS evaluation)** scores Tier 5's outputs against the 30-question `evaluation/golden_qa.json` set, comparing answer quality (faithfulness, relevance) and cost-per-correct-answer against Tiers 1–4. The agentic loop typically wins on multi-hop + citation-heavy questions and ties on single-hop lookups (where the second turn is wasted overhead).

**Open Q4 (deferred enhancement):** Tier 5 could optionally call Tier 4's multimodal `aquery` as a third tool, letting the agent route figure-grounded questions to the multimodal index and text-only questions to Tier 1's ChromaDB. This is the natural Phase 131+ extension — it requires no changes to Tier 4's surface, just a new `@function_tool` wrapping `tier-4-multimodal/query.py::ask`. Tracked but explicitly OUT OF SCOPE for Phase 130's `TIER-05` ROADMAP gate.

If you only run one tier in this repo, run **Tier 1**. If you want the agentic UX, run Tier 1 first (so `chroma_db/tier-1-naive/` is populated), then Tier 5. The cost delta vs Tier 1 (~$0.005 vs ~$0.0014, ~3.5×) is the price of autonomy + citation verification.

---

## Docker

**Docker is OPTIONAL for Tier 5.** Tier 5's runtime dependencies are all pip-installable (`openai-agents[litellm]==0.14.6` + `chromadb>=1.5.8` + `openai>=1.50`) and have no system-level binary requirements (no MineRU, no LibreOffice, no torch download). Phase 130 ships a Tier 4 `Dockerfile` only — Tier 4 needs containerization because of MineRU's 3–5 GB model download (REPO-05 mandatory path). Tier 5 does not. If you want Tier 5 in a container anyway, point a `python:3.11-slim` base at the repo's `pyproject.toml [tier-5]` extras — no special Dockerfile needed.

This is the **honest REPO-05 contract**: containers are mandatory ONLY where pip-only is genuinely impractical (Tier 4). Forcing Tier 5 into a container would be ceremony, not value.
