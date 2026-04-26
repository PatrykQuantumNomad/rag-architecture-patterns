> [!WARNING]
> **One-time ingest cost: ~$1.00 (~10 min) via `google/gemini-2.5-flash` on OpenRouter.**
> LightRAG runs an LLM call per chunk to extract entities + relationships. Cost is dominated
> by the chunk count (~420 for 100 papers). Re-runs are near-free thanks to LightRAG's
> `kv_store_doc_status.json` deduplication. `--reset --ingest` re-extracts at full cost.
> `python main.py --yes` skips the confirmation prompt for non-interactive use.

# Tier 3: Graph RAG (LightRAG)

> Knowledge-graph RAG that traverses entities + relationships across documents. Answers **multi-hop questions Tier 1 cannot.**

LightRAG indexes the corpus as a knowledge graph: each chunk is sent to an LLM that extracts named entities and the relationships between them, building a NetworkX graph alongside the usual vector index. At query time, LightRAG combines (a) keyword extraction → entity neighborhood traversal (local mode) and (b) community-summary retrieval (global mode) to surface chunks that single-hop vector search would miss. The on-disk artifacts are exclusive to this tier and live under `lightrag_storage/tier-3-graph/`.

This tier is the **multi-hop chapter** of the repo's narrative. The default demo question explicitly references two papers (Lewis et al. 2020 RAG and Karpukhin et al. 2020 DPR) and asks how their concepts relate — Tier 1's vector-only top-k typically retrieves chunks from only one of them, while Tier 3's hybrid mode walks the shared "non-parametric memory" entity edge and surfaces both.

If Tier 1 is "good enough most of the time, cheap to run", Tier 3 is "expensive once, much smarter on hard questions". The cost asymmetry (Tier 1 ingest ≈ $0.011 vs Tier 3 ≈ **$1.00**, ~90×) is the explicit trade.

---

## Quickstart

```bash
cd tier-3-graph
uv pip install -r requirements.txt          # or: cd .. && uv pip install -e ".[tier-3]"
cp ../.env.example ../.env                  # then edit and set OPENROUTER_API_KEY
python main.py --yes                        # default: extract graph (~$1) + run multi-hop demo query
```

Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys); browse models at [openrouter.ai/models](https://openrouter.ai/models). The first `python main.py` ingests the full corpus into `lightrag_storage/tier-3-graph/` (one-time cost ~$1) and then runs the canonical multi-hop probe query. `--yes` short-circuits the cost-surprise prompt — omit it on interactive runs to see the warning + confirmation gate. Subsequent runs are near-free: LightRAG's `kv_store_doc_status.json` deduplicates by `paper_id`, so `--ingest` is idempotent.

---

## CLI reference

| Flag | Purpose | Default |
|------|---------|---------|
| `--ingest` | Build the knowledge graph (PDFs → entity/relationship extract → graphml + vdb). Idempotent on re-run. | off |
| `--query "..."` | Run a graph-aware retrieval query against the persisted graph. | canned multi-hop demo |
| `--mode <m>` | Retrieval strategy: `naive` / `local` / `global` / `hybrid` / `mix`. See descriptions below. | `hybrid` |
| `--reset` | Wipe `lightrag_storage/tier-3-graph/` before `--ingest`. **Re-ingest costs ~$1.** | off |
| `--model <slug>` | OpenRouter LLM slug for entity extraction + answers. Must be in `shared/pricing.py`. | `google/gemini-2.5-flash` |
| `--yes` | Skip cost-surprise confirmation prompts (for non-interactive use). | off |

### `--mode` choices

| Mode | What it does | When to use |
|------|--------------|-------------|
| `naive` | Vector-only retrieval (single-hop, no graph traversal) — Tier-1-equivalent. | Sanity check / ablation. |
| `local` | Entity neighborhood traversal — finds chunks linked to entities mentioned in the question. | Single-paper questions about specific entities. |
| `global` | Community-summary retrieval — high-level themes across the corpus. | Broad / synthesizing questions. |
| `hybrid` | `local` + `global` blended. **Safe default** — works on every install. | Multi-hop questions. |
| `mix` | `hybrid` + reranker. May silently degrade without `bge-reranker-v2-m3` configured (Open Q4). | Opt-in once a reranker is installed. |

Invoking `python main.py` with **no flags** auto-runs `--ingest` (gated unless `--yes`) followed by the default `--query` against `--mode hybrid`. This is the canonical "does it work?" demo.

---

## Expected cost (vintage 2026-04)

| Operation | Calls | Avg tokens | Cost |
|-----------|-------|-----------|------|
| One-time entity extraction (~1 LLM call per chunk, ~420 chunks) | 420 | 3K in + 600 out | **~$1.00** |
| One-time embedding (~420 chunks × ~1.2K tokens) | — | 500,000 | ~$0.01 |
| Per-query LLM (graph traversal + answer) | 1-3 | 5,400 | ~$0.005 |
| Per-query embedding (keyword extraction) | 1 | 30 | <$0.000001 |

**Comparison:** Tier 1 ingest ≈ $0.011; Tier 3 ingest ≈ **90× more** ($1.00). The trade is: per-query reasoning quality on multi-hop questions Tier 1 misses entirely.

Cost and latency are printed for every query via `shared.display.render_query_result` plus a `Latency:` footer line, and a per-run JSON is written to `evaluation/results/costs/tier-3-<timestamp>.json`. The `CostAdapter` (Plan 03) routes both LLM and embedding usage from LightRAG's `token_tracker` callback into a single `CostTracker` instance, so the JSON captures BOTH ingest and per-query spend.

---

## Persistence

LightRAG persists the graph + KV stores under **`lightrag_storage/tier-3-graph/`** (gitignored). Seven artifacts:

- `graph_chunk_entity_relation.graphml` — the NetworkX knowledge graph (entities + relationships).
- `vdb_entities.json` — vector index over entity descriptions.
- `vdb_relationships.json` — vector index over relationship descriptions.
- `vdb_chunks.json` — vector index over the original chunks.
- `kv_store_full_docs.json` — full document texts keyed by paper_id.
- `kv_store_text_chunks.json` — chunked texts keyed by chunk id.
- `kv_store_doc_status.json` — idempotency log (Pattern 5; deduplicates re-ingest).
- `kv_store_llm_response_cache.json` — cache of LLM responses per chunk (re-runs near-free).

**`EMBED_DIMS = 1536` is hardcoded** at the module level in `tier-3-graph/rag.py`. LightRAG indexes vectors at first ingest; switching embedding models without `--reset` silently corrupts retrieval (HKUDS issue #2119; Pitfall 4). To change the embedding model: `python main.py --reset --ingest --yes` (re-extracts entities at full cost).

---

## Known weaknesses (deliberate)

These are not bugs — they are the trade-offs Tier 3 makes to deliver multi-hop reasoning, and each one motivates a future tier or future plan.

### Cost asymmetry vs Tier 1

$1 vs $0.01 ingest (~90×). Justified ONLY when multi-hop reasoning matters for your workload. If most of your queries are single-paper, single-hop, lookup-style — Tier 1 is the right answer and Tier 3 is overspend. Use `evaluation/golden_qa.json` (Phase 131) to quantify.

### LightRAG version churn (Pitfall 9)

LightRAG releases every 2–7 days, and the `token_tracker` protocol shape is not yet covered by their CI. `[tier-3]` extras pin `lightrag-hku==1.4.15` **EXACTLY**. Bumping the pin requires re-running `scripts/probe_lightrag_token_tracker.py` to confirm `add_usage(...)` is still called with the same payload shape, and possibly updating `tier_3_graph.cost_adapter`.

### `mix` mode may degrade without a reranker (Open Q4)

`mix` is documented in LightRAG as `hybrid + reranker`. Without `bge-reranker-v2-m3` (or equivalent) configured, it can silently produce worse results than `hybrid` alone. We default to `hybrid` for that reason. If you install a reranker, switch to `--mode mix` for marginal recall gains on long-tail entity edges.

### No image / figure handling

LightRAG is text-only by default — figures, charts, diagrams are stripped at extract time and never embedded. Multimodal questions (e.g., "what does Figure 3 show?") will score near zero. See **Tier 4 (Phase 130, RAG-Anything)** for the figure-aware pipeline that builds on top of LightRAG.

### Per-chunk citation surfacing is weaker than Tier 2's

LightRAG returns a synthesized prose answer that blends multiple graph sources; it does not expose a clean per-chunk citation list the way Tier 2 (Gemini File Search grounding metadata) does. `render_query_result` is called with `chunks=[]` for graph queries and prints "No chunks retrieved." — that's an honest representation of "the answer comes from a graph traversal, not a top-k chunk list." Phase 131 evaluation may extract citations from the answer text via heuristics.

A representative Tier-3 win is the default multi-hop probe; representative Tier-3 losses are figure-grounded questions (Tier 4 territory) or single-paper lookup questions (Tier 1 territory).

---

## Sample query

The default canned query is the **multi-hop DPR ↔ RAG probe** (verbatim in `tier-3-graph/main.py` as `DEFAULT_QUERY`):

> *"Comparing Lewis et al. 2020's RAG and Karpukhin et al. 2020's DPR, how does dense passage retrieval relate to RAG's non-parametric memory?"*

This question requires synthesizing two papers via cross-document entity edges. Tier 1's vector-only top-k often retrieves chunks from only one of the two papers — usually the RAG paper, because "non-parametric memory" is a phrase used heavily in Lewis 2020 and weakly in Karpukhin 2020. Tier 3's `hybrid` mode traverses entities (DPR → RAG via the shared "non-parametric memory" entity in the graph) and surfaces both, producing a synthesized comparison. Phase 131 (evaluation) will quantify the delta against `evaluation/golden_qa.json`'s **multi-hop** entries.

---

## Architecture

The end-to-end pipeline:

```
PDFs -> tier-3-graph/ingest.py:extract_full_text   (PyMuPDF — Tier-3-owned per Pitfall 11)
     -> rag.ainsert(text, ids=[paper_id])          (idempotent via kv_store_doc_status)
     -> [LightRAG: chunk -> LLM entity/relationship extract -> NetworkX graph + nano-vectordb]
     -> rag.aquery(question, param=QueryParam(mode='hybrid'))
     -> [LightRAG: keyword extract -> entity neighborhood + global community summary -> answer]
     -> shared.display.render_query_result(chunks=[]; answer-only is honest for graph RAG)
```

Tier 3 owns its `import fitz` (PyMuPDF) so it can be reproduced from `tier-3-graph/requirements.txt` alone — zero cross-tier coupling (Pitfall 11). The single `asyncio.run(amain)` bridge at the bottom of `main.py` is the only async/sync boundary (Pitfall 8); every LightRAG call inside `amain` stays async. Cost tracking flows from LightRAG's `token_tracker` callback through `tier_3_graph.cost_adapter.CostAdapter` into `shared.cost_tracker.CostTracker` — capturing both LLM and embedding spend on a single key (`OPENROUTER_API_KEY`) on a single gateway (OpenRouter).

---

## Reused by

**Phase 130 / Tier 4 (RAG-Anything)** builds on top of LightRAG, extending it to handle figures + tables via the RAG-Anything multimodal wrapper. The graph schema and storage layout are inherited; Tier 4 adds image-aware extraction stages on top.

**Tier 5 (Agentic)** may use Tier 3's persisted graph as ONE retrieval tool alongside Tier 1's ChromaDB collection, letting the agent loop pick the right retriever per query — single-hop questions go to Tier 1, multi-hop to Tier 3, multimodal to Tier 4. This is why the persistence path (`lightrag_storage/tier-3-graph/`) is locked: changing it would break the downstream agent.

If you only run one tier in this repo, run **Tier 1** first (cheap, fast). Then run a multi-hop entry from `evaluation/golden_qa.json` against both Tier 1 and Tier 3 — that delta is the entire reason Tier 3 exists.
