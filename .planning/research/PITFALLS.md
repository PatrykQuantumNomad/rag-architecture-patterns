# Pitfalls Research — Shipping Credible RAG Comparison Numbers

**Domain:** RAG comparison eval ship-readiness for external blog post (v1.0)
**Researched:** 2026-05-04
**Confidence:** MEDIUM-HIGH (RAGAS / openai-agents / judge-bias findings cross-verified across GitHub issues, official docs, and 2024-2025 papers; LightRAG/RAG-Anything ship-state findings rest on fewer issue threads → MEDIUM)

**Scope of this file:** failure modes that would invalidate eval numbers between "ran the harness" and "the blog reader trusts them." Each pitfall is tagged **SHIP-BLOCKER** (must fix or pull from blog) or **DISCLOSE-AND-SHIP** (disclose honestly in the frozen handoff doc, ship anyway).

The blog deadline is real. Cost budget is 2-3 full re-runs (~$1-3 each). The goal is *not* a perfect benchmark — it is "numbers that survive a hostile re-read on Hacker News."

---

## Critical Pitfalls

### Pitfall 1: Tier 5 NaN ships as zero (or worse, as a real score)

**Tag:** SHIP-BLOCKER

**What goes wrong:**
Tier 5 (openai-agents + FileSearchTool) currently returns `30/30 empty_contexts` — every row produces NaN for faithfulness, answer_relevancy, context_precision. If the rollup script averages NaN as zero, or if a downstream consumer reads the dash and assumes "score not measured = score is zero," Tier 5 looks catastrophically bad in the blog when in fact the harness never captured contexts.

**Why it happens:**
Three known root causes for `file_search` returning empty results (any of which match the symptom):

1. **Vector store status lies.** OpenAI's vector store API returns `status: "completed"` before the index is searchable. Immediate `file_search` calls return empty silently — no error, no warning. ([community thread](https://community.openai.com/t/bug-vector-store-status-completed-does-not-guarantee-searchability-file-search-returns-empty-results-silently/1374471))
2. **Files uploaded but not linked.** Uploading a file and creating a vector store can succeed independently — the file may not be attached to the store even though both API calls returned 200. ([community thread](https://community.openai.com/t/uploaded-file-is-not-linked-to-vector-store/1336290))
3. **Harness reads RunResult wrong.** The `RunResult.new_items` list contains `tool_call_item` and `tool_call_output_item` events for `file_search_call` — if the eval harness only inspects `final_output` (the agent's text answer) and never walks `new_items` to extract retrieved context, every row will record `contexts=[]` regardless of whether the search succeeded. ([Tools docs](https://openai.github.io/openai-agents-python/tools/), [Results docs](https://openai.github.io/openai-agents-python/results/))

The shape of "30/30 empty_contexts" — every single row, not a subset — strongly points at root cause #3 (harness extraction bug) rather than #1/#2 (which would produce intermittent failures).

**Consequences:**
- Blog ships Tier 5 with dashes → reader assumes the architecture failed → unfair to openai-agents
- Or worse: someone "fixes" the rollup to treat NaN as 0 → Tier 5 ships with faithfulness=0 → blog claims "agentic RAG performs worst" with completely fabricated data

**Prevention:**
- **Code-level:** before re-running, write a 5-question smoke test that calls `Runner.run()` and prints the full `RunResult.new_items` list. Confirm `file_search_call` items contain populated `results` arrays. If they don't, the bug is upstream of harness extraction (vector store / agent config). If they do, the bug is in how the harness flattens `new_items` → `contexts`.
- **Harness-level:** in the eval capture path, set `FileSearchTool(include_search_results=True)` so retrieved chunks are returned in the trace, then walk `new_items` filtering `type == "tool_call_output_item"` and concatenate the `results[].content[]` text. Do not rely on `final_output` alone.
- **Disclosure-level:** if root cause turns out to be vector-store-not-ready (post-fix latency), document the wait pattern in the frozen handoff and add a `time.sleep` or polling loop in `tier-5-agentic/main.py`.

**Warning signs:**
- All-or-nothing failure pattern (30/30 not 5/30) → harness bug, not flake
- `tool_call_item` events present in trace but `tool_call_output_item` events missing or with empty `results`
- Final agent text *answers* the question correctly but `contexts=[]` → agent hallucinated; file search returned nothing

**Phase to address:** eval-rerun (ship-blocker — cannot publish Tier 5 numbers without fixing this)

---

### Pitfall 2: Tier 4 graphml regenerated mid-run, scores reflect partial graph

**Tag:** SHIP-BLOCKER

**What goes wrong:**
Tier 4's MineRU PDF ingest is brittle (Phase 139 evidence: previous graphml cache corrupted, ingest must run outside the sandbox). If the re-run captures Tier 4 numbers while the graph is still incrementally building — or if a previous run's stale `kv_store_*.json` / `graph_chunk_entity_relation.graphml` artifacts leak in — Tier 4's faithfulness/context_precision will reflect a graph that never existed in any single coherent state.

**Why it happens:**
- LightRAG (which RAG-Anything sits on) supports incremental updates that *seamlessly* merge new data into the existing graph. "Seamless" cuts both ways: there is no clean "graph version N" boundary. If ingest is interrupted, the on-disk graphml is a partial state that queries silently. ([LightRAG repo](https://github.com/HKUDS/LightRAG))
- MineRU has documented version-compatibility issues with PaddleOCR and other deps; legacy `*_content_list.json` formats can leak into the current ingest path and amplify noise. ([RAG-Anything paper](https://arxiv.org/html/2510.12323v1))
- The `tier-4-multimodal/` storage directory contains both LightRAG's KV stores AND MineRU's intermediate JSONs. A "clean" rerun must clear both.

**Consequences:**
- Blog publishes Tier 4 numbers that aren't reproducible — re-running on the same corpus gives different scores because the graph state was non-deterministic
- A reader who tries to reproduce gets different numbers, files an issue, and the credibility hit is binary

**Prevention:**
- **Harness-level:** before Tier 4 re-run, *delete* (don't move, don't archive) `rag_anything_storage/tier-4-multimodal/` entirely. Re-ingest from scratch.
- **Code-level:** after ingest, before query phase, log: `(graphml node count, graphml edge count, kv_store entry count, graphml file mtime)`. Capture this in the provenance block of `eval-numbers-v1.X.md`. A re-reader can then verify graph state matched the captured numbers.
- **Code-level:** capture the `MineRU_version`, `lightrag_version`, `rag_anything_version` (pip freeze) in provenance — version mismatch is a known-bad failure mode and disclosing it lets readers reproduce.
- **Disclosure-level:** explicitly state in the frozen doc: "Tier 4 ingest is the only tier that requires running outside the eval sandbox; graphml hash captured at $SHA."

**Warning signs:**
- Graphml file size differs between two consecutive ingests of the same corpus → non-deterministic ingest
- `kv_store_full_docs.json` count != source paper count → partial ingest
- MineRU stderr contains "PaddleOCR version mismatch" or "legacy content_list format" → upstream brittleness, scores are suspect

**Phase to address:** eval-rerun (ship-blocker — Tier 4 is the multimodal differentiator; bad graph = the blog's central narrative breaks)

---

### Pitfall 3: Self-grading bias inflates Tiers 2-4 relative to Tier 1

**Tag:** DISCLOSE-AND-SHIP

**What goes wrong:**
The judge LLM is `google/gemini-2.5-flash`. Tiers 2-4 use Gemini-family models for generation (Tier 2 directly, Tier 3 via `shared.llm`, Tier 4 via RAG-Anything → LightRAG). Tier 1 uses `google/gemini-2.5-flash` via OpenRouter, Tier 5 uses OpenAI for retrieval but Gemini for the final agent. The judge "recognizes" outputs from its own family and rates them higher.

**Why it happens:**
Multiple 2024-2025 papers (Panickssery et al. NAACL 2024, "Self-Preference Bias in LLM-as-a-Judge" Wataoka et al. 2024) document that LLM judges assign systematically higher scores to outputs from their own family. The mechanism: judges prefer text with lower perplexity under their own distribution, and same-family outputs have lower perplexity. ([arxiv 2410.21819](https://arxiv.org/abs/2410.21819), [Justice or Prejudice](https://llm-judge-bias.github.io/))

For this eval specifically: any tier whose generator is Gemini-family enjoys a systematic boost on faithfulness (Gemini judging Gemini's claim decomposition) and answer_relevancy (Gemini's question regeneration matches Gemini's natural phrasing).

**Consequences:**
- The blog's "Tier 3 Graph RAG wins on faithfulness" headline could be partially an artifact of judge family-bias, not a real architectural advantage
- Hostile re-read: "you used Gemini to grade Gemini's outputs and called it a benchmark"

**Prevention:**
- **Disclosure-level (recommended for v1.0):** add an explicit disclosure block to the frozen handoff. Suggested language:

  > **Self-grading bias disclosure:** Faithfulness, answer_relevancy, and context_precision were graded by `google/gemini-2.5-flash`. Tiers 2, 3, and 4 use Gemini-family models for generation; Tier 1 uses Gemini via OpenRouter; Tier 5 uses OpenAI for retrieval and Gemini for the agent loop. Published research (Panickssery 2024, Wataoka 2024) documents that LLM judges assign 5-15% higher scores to outputs from their own family. The numbers below should be read as *relative within this benchmark*, not absolute quality estimates. A cross-judge run (e.g. Claude or GPT-4 as judge) is out of scope for v1.0 — see open question #4 for follow-up.

- **Cost-aware mitigation (out of scope per project constraints):** a single re-run with a cross-family judge (e.g. `claude-sonnet-4` or `gpt-4o`) on a 10-question subset would expose the bias magnitude. ~$0.30 cost. Listed as v1.1 follow-up, not v1.0 blocker.
- **Anti-pattern to avoid:** ensemble-of-judges majority vote — sounds rigorous but adds new biases (each judge has its own family preferences) and explodes cost. For 30 questions, just disclose.

**Warning signs:**
- Tier ranking is suspiciously aligned with "which generator family matches the judge family"
- Hand-spot-check: pick 5 high-scoring rows and 5 low-scoring rows; do the scores match human judgment? If not, judge bias is plausible.

**Phase to address:** frozen-doc (disclosure-level only for v1.0; cross-judge sanity check is v1.1 candidate)

---

### Pitfall 4: RAGAS faithfulness NaN — silent answer-extraction failure

**Tag:** SHIP-BLOCKER (for any tier where it shows up post-rerun)

**What goes wrong:**
RAGAS faithfulness works in two LLM-mediated steps: (1) decompose the answer into atomic statements; (2) verify each statement against the retrieved context. If step (1) returns zero statements — because the answer was empty, the JSON parse failed, or the LLM refused — the metric returns NaN. This is *distinct* from the `empty_contexts` NaN (no contexts to verify against). Both produce NaN in the same column.

**Why it happens:**
- RAGAS 0.4.x with Gemini has documented JSON-output incompatibility. Pydantic validation fails on missing fields, retry logic gives up, returns NaN. ([Issue #2073](https://github.com/explodinggradients/ragas/issues/2073), [Issue #1150](https://github.com/explodinggradients/ragas/issues/1150))
- Recent commits in this repo (`fb397f0`, `186a9c2`) already patch around two RAGAS 0.4.3 quirks — there are likely more. ([Issue #1651: "No statements were generated"](https://github.com/explodinggradients/ragas/issues/1651))
- The current `comparison.md` shows Tier 2 multi-hop has `Faithfulness: —` (NaN) while Answer Relevancy is `0.782`. This is the symptom: claim decomposition failed, similarity scoring still ran. It is *not* called out in the disclaimers.

**Consequences:**
- Aggregate means computed over rows with hidden NaNs are arithmetic lies (depending on `np.nanmean` vs `np.mean` behavior)
- A reader who notices "n=10, faithfulness=—" but Answer Relevancy is populated will rightly ask "what's going on?" and trust drops

**Prevention:**
- **Code-level:** in the eval harness, when faithfulness NaN is detected, log the *reason* — was it (a) `len(retrieved_contexts) == 0`, (b) `len(extracted_statements) == 0`, or (c) `JSONDecodeError`? Capture per-row in the metrics CSV. The current `n NaN` column is opaque; a `nan_reason` breakdown is a 30-minute change.
- **Code-level:** wrap RAGAS calls in retry-with-backoff specifically for `JSONDecodeError`. RAGAS 0.4.3 has internal retry but it is known to give up too fast on Gemini.
- **Disclosure-level:** in the frozen handoff, the NaN breakdown table must distinguish "empty_contexts" from "empty_statements" from "json_parse_failure" — they have very different blog implications.

**Warning signs:**
- Faithfulness NaN with Answer Relevancy populated for the same row → claim decomposition failed
- Faithfulness NaN AND Answer Relevancy NaN for the same row → empty contexts (real retrieval failure)
- Spike in NaN count in multi-hop class only → judge LLM struggling with longer answers, structured-output limit

**Phase to address:** eval-rerun (ship-blocker if NaN rate >5% per tier; disclose if 1-5%)

---

### Pitfall 5: Apples-to-oranges embedding models across tiers

**Tag:** DISCLOSE-AND-SHIP

**What goes wrong:**
Each tier ships its own embedding model:
- Tier 1: ChromaDB default or OpenRouter-served (per `tier-1-naive/`)
- Tier 2: Gemini File Search uses Google's embedding (`text-embedding-004` or successor)
- Tier 3: LightRAG defaults to OpenAI `text-embedding-3-small` unless overridden
- Tier 4: RAG-Anything inherits LightRAG's embedding
- Tier 5: OpenAI vector store uses OpenAI's hosted embedding (proprietary, not user-controlled)
- Judge: `text-embedding-3-small` for answer_relevancy cosine similarity

When tiers use different embedders, "Tier X has higher context_precision" conflates retrieval architecture with embedding-model quality. This is the most common methodological critique of multi-system RAG benchmarks.

**Why it happens:**
- Forcing a single embedding across all tiers defeats the point of the comparison (Tier 2's value prop *is* Google's hosted embedding)
- But not disclosing it lets readers assume "Tier X's architecture is better" when "Tier X's embedder is better" might be the real story
- Public benchmarks like MTEB are heavily fine-tuned on by embedding model authors, inflating absolute scores. ([arxiv 2407.08275](https://arxiv.org/html/2407.08275v1))

**Consequences:**
- Hostile re-read: "did you control for embedding model? No? Then this isn't a tier comparison, it's an embedding comparison."
- The blog's narrative about *architecture* tradeoffs gets undermined by an unmentioned variable.

**Prevention:**
- **Disclosure-level:** every tier's STACK row in the frozen doc must list its embedder by name and version. Suggested template:

  | Tier | Architecture | Embedder | Embedder source |
  |------|--------------|----------|------------------|
  | 1 | Naive Chroma | text-embedding-3-small | OpenAI direct |
  | 2 | Gemini File Search | text-embedding-004 (managed) | Google hosted |
  | 3 | LightRAG | text-embedding-3-small | OpenAI via LightRAG |
  | 4 | RAG-Anything | text-embedding-3-small | OpenAI via LightRAG |
  | 5 | openai-agents | OpenAI hosted (managed) | OpenAI hosted |

  Then a note: "Tiers 2 and 5 use *managed* embeddings — vendor controls the model and may swap it without our knowledge. Numbers reflect the embedder available on $eval_date."

- **Code-level (optional):** capture the embedder name/version in `evaluation/results/metrics/*.json` provenance per row. Free win for reproducibility.
- **Anti-pattern to avoid:** forcing all tiers onto OpenAI embeddings to "level the playing field" — that breaks Tier 2 entirely (Gemini File Search is a vertically-integrated product) and is a worse benchmark.

**Warning signs:**
- Reader asks "is this comparing architectures or embedding models?" → disclosure is unclear
- Two tiers with the same embedder show very different context_precision (e.g. Tier 3 vs Tier 4 — both `text-embedding-3-small`) → the architecture difference is real
- Two tiers with different embedders show similar scores → architecture parity holds across embedders

**Phase to address:** frozen-doc (disclosure-level; this is a writing-the-disclaimer task, not an engineering task)

---

## Moderate Pitfalls

### Pitfall 6: Sample size n=30 ships without confidence intervals

**Tag:** DISCLOSE-AND-SHIP

**What goes wrong:**
30 questions × 5 tiers is below the threshold where simple bootstrap confidence intervals are reliable (Efron-Tibshirani rule of thumb: bootstrap *starts* working at n≥30; CIs need n≥100). The current PROJECT.md explicitly puts statistical-significance testing out of scope, but the blog will still show numbers like "Tier 1: 0.799, Tier 3: 0.875" without any "±" — readers will mentally treat them as point estimates with implied precision.

**Prevention:**
- **Disclosure-level:** lead the frozen doc with a "Read this first" block that says explicitly: *"n=30 per tier. We report magnitudes, not significance. A 0.05-0.10 gap between tiers is within plausible noise for this sample size. Treat ranking, not exact values, as the signal."* Existing comparison.md already has a version of this — make it more prominent and put it adjacent to the rollup table, not at the bottom.
- **Disclosure-level:** report per-question-class breakdowns (already done) — n=10 per class is even more brittle, must be flagged.
- **Cost-aware option (out of scope):** a single bootstrap pass over the existing 30 results gives at least *some* CI estimate. Cheap (no new LLM calls). Listed as v1.1 candidate.

**Warning signs:**
- Blog draft has language like "Tier 3 is 9.5% better than Tier 1 on faithfulness" — that's a magnitude claim and is fine. "Tier 3 significantly outperforms Tier 1" is a significance claim and would be wrong.

**Phase to address:** frozen-doc

---

### Pitfall 7: openai-agents tool_choice and hosted-tool routing

**Tag:** SHIP-BLOCKER (for Tier 5)

**What goes wrong:**
Hosted tools (FileSearchTool, WebSearchTool) are processed by the LLM, not by the SDK. The SDK *cannot* force a hosted tool to be used. If the agent's instructions or `tool_choice` are wrong, the model may answer from its prior knowledge without calling `file_search` at all — every row would show empty contexts (because no search was performed), even though the tool is technically attached. ([Tools docs](https://openai.github.io/openai-agents-python/tools/))

**Why it happens:**
- Default `tool_choice="auto"` lets the model decide — for a question like "what is RAG?" the model may answer from training data and skip retrieval
- `tool_choice="required"` forces *some* tool call but the model can pick any tool and may not pick FileSearchTool specifically for hosted tools (named tool choice for hosted tools has limitations)
- Even when `file_search` is called, if the vector store is empty or the query phrasing doesn't match indexed content, results come back empty silently

**Prevention:**
- **Code-level:** for the eval re-run, set the agent's instructions to *explicitly* require retrieval ("Always call file_search before answering. Quote retrieved chunks verbatim.") and inspect traces to confirm `file_search_call` is present in `new_items` for every question.
- **Code-level:** add `FileSearchTool(include_search_results=True)` so the trace contains the actual chunks (default may not).
- **Smoke test:** before full re-run, dry-run 5 questions with full trace dump. Verify: (a) `file_search_call` present, (b) `results[]` non-empty, (c) final answer references retrieved content.

**Warning signs:**
- Tier 5 final answers are coherent but `contexts=[]` → model answered without retrieval
- Tier 5 answers are inconsistent across reruns of same question → model is sampling from training data

**Phase to address:** eval-rerun (this is the most likely root cause of Tier 5's 30/30 NaN — fix this and Tier 5 should populate)

---

### Pitfall 8: Provenance hash mismatch — "ce5c2ad" doesn't match the rerun

**Tag:** DISCLOSE-AND-SHIP

**What goes wrong:**
Current comparison.md captures git SHA `ce5c2ad`. Recent commits (`fb397f0`, `186a9c2`, `f0cd134`, `f801916`) post-date that capture and changed eval behavior. If the blog references "captured at ce5c2ad" but the actual rerun happens on `HEAD` post-fixes, the SHA in the doc lies.

**Why it happens:**
Provenance gets captured automatically at run time but the *rollup doc* is hand-edited and can drift.

**Prevention:**
- **Code-level:** the rollup script (`evaluation.harness.compare`) should overwrite the provenance section automatically — don't allow the `comparison.md` to hold stale SHAs.
- **Disclosure-level:** the frozen handoff doc must capture, at minimum: `(eval_date_utc, generation_model_full_id, judge_model_full_id, judge_embedder_id, git_sha, ragas_version, openai_agents_version, lightrag_version, rag_anything_version)` per tier. The current per-tier provenance block has model + SHA but is missing library versions.

**Warning signs:**
- A reader checks out the cited SHA and the harness behaves differently than the doc claims → provenance lied → credibility gone
- Two tiers in the same comparison block reference different SHAs (mid-rerun snapshot) → the "single eval-date" promise broke

**Phase to address:** frozen-doc + eval-rerun (mechanical: rerun all 5 tiers on a single SHA, capture provenance atomically)

---

### Pitfall 9: Latency / cost numbers measure cold cache, not steady state

**Tag:** DISCLOSE-AND-SHIP

**What goes wrong:**
Latency captured during the eval includes any cold-start, retry, or API rate-limit-backoff time. Tier 3 (LightRAG) at 9.09s mean latency may be cold cache; Tier 5 at 3.49s may be retry-throttled. Cost per query for Tier 3 is 10× the others (0.011 vs 0.001) — is that real or did one query trigger a runaway agent loop?

**Why it happens:**
The harness records wall-clock time per query and total LLM cost via `CostTracker`, but doesn't distinguish "this query had 1 LLM call" from "this query had 8 LLM calls because the agent looped." Outliers dominate means at n=30.

**Prevention:**
- **Disclosure-level:** report median + p95 alongside mean for latency. One outlier on 30 dominates the mean.
- **Code-level (cheap):** add `n_llm_calls` and `n_retries` to per-row metrics. Already partially captured in cost data — just surface it in the rollup.
- **Disclosure-level:** explicit caveat: "Latency includes API roundtrip and any internal retries. First-query cold-start is not separated. Numbers are end-user-visible latency for a single query against a warm system."

**Warning signs:**
- Tier 3 cost/query ≈ 10× Tier 1 with no architectural justification → likely an LLM-call-count difference (graph traversal does multiple LLM calls), worth a one-line note in the blog
- Single-query max latency >> p95 → a query had a retry storm; consider rerunning that row

**Phase to address:** frozen-doc (disclosure); optional eval-rerun improvement if median/p95 reporting can be added cheaply

---

### Pitfall 10: Multimodal scores for text-only tiers are reported as if comparable

**Tag:** DISCLOSE-AND-SHIP (already partially disclosed)

**What goes wrong:**
The current comparison.md per-question-class table shows multimodal scores for Tiers 1, 2, 3 — which are text-only architectures. Tier 1 multimodal faithfulness is 0.649, answer_relevancy 0.221. A reader scanning the table without reading the disclaimer below sees those numbers and thinks "Tier 1 sort of handles multimodal." It doesn't — those scores reflect the LLM bullshitting from text-only retrieved chunks about questions that need image understanding.

**Why it happens:**
The table is the visual centerpiece. Disclaimers are below. Eye-flow reads numbers first, caveats last (or never).

**Prevention:**
- **Disclosure-level:** in the frozen handoff, *strike through* (or use em-dash) the multimodal column for Tiers 1-3 in the headline rollup, and put the actual numbers in a separate "for reference only" table below. Make the visual clearly say "this tier didn't compete on multimodal."
- **Disclosure-level:** the existing disclaimer ("Tiers 1-3 are text-only...") is correct but buried. Promote it to a prominent note above the table.

**Warning signs:**
- Blog reader cites "Tier 1 multimodal: 0.649" out of context → table layout failed

**Phase to address:** frozen-doc

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip cross-judge sanity-check for v1.0 | Saves ~$0.30 + 1 hour | Self-grading bias remains an open question; future critique vector | Acceptable for v1.0 with explicit disclosure; mandatory for v1.1 if blog gets traction |
| Don't separate "empty_contexts" from "empty_statements" NaN reasons | 30-min coding saved | Aggregate NaN counts are opaque, can't diagnose post-publication critique | Never — this is a 30-min change, do it before the rerun |
| Report mean latency only (no median/p95) | Fits in the existing column | Outliers dominate; misleads readers who think "mean" = "typical" | Acceptable if disclosure is explicit; better to add median |
| Single eval run for the blog (no re-run for stability) | Saves $1-3 | Numbers are point estimates; one-shot run could be unlucky | Acceptable per project budget — disclose as one-shot |
| Use the same Gemini for generator and judge | Already in place; switching adds cost | Self-grading bias | Acceptable with disclosure (see Pitfall 3) |
| Hand-edit comparison.md provenance | Faster than auto-generation | SHAs/models drift between rerun and doc | Never for the frozen handoff doc — must be machine-generated |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenAI vector store (Tier 5) | Querying immediately after `status=completed` | Poll `vector_store.file_counts.in_progress == 0` AND wait additional ~5s before query |
| openai-agents FileSearchTool | Reading `final_output` to extract retrieved chunks | Walk `RunResult.new_items`, filter `tool_call_output_item`, extract `results[].content[]` |
| openai-agents `tool_choice` | Setting `tool_choice="required"` and assuming FileSearchTool is called | Hosted tools cannot be force-targeted; instruction tuning is the right knob |
| RAGAS 0.4.x with Gemini | Trusting RAGAS internal retry on JSONDecodeError | Wrap calls externally with retry; capture nan_reason per row |
| RAGAS context_recall | Computing without `reference` field | RAGAS silently returns NaN; verify `reference` is populated for every golden row |
| LightRAG storage cleanup | Deleting graphml only | Must also clear `kv_store_*.json`, `vdb_*.json`, and any cache dir — otherwise zombie state |
| MineRU PDF ingest | Running inside the same sandbox as the eval | Run outside per Phase 139; capture extracted markdown commit-by-commit so reruns are deterministic |
| Cost tracker across providers | Summing costs assuming uniform pricing table | Pricing table is per-model; verify `pricing.py` was updated for any model change since 2026-05-02 |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Tier 3 graph traversal LLM-call explosion | Cost/query >> peer tiers, latency outliers | Cap traversal depth in LightRAG config; log `n_llm_calls` per query | At n=30 it's tolerable cost-wise; at n=300 it becomes the dominant cost |
| Tier 5 agent loop runaway | Single query takes 30s+, cost spike | Set `max_turns` on `Runner.run()`; alert if any query crosses threshold | Already a risk at n=30 — one runaway = $0.50+ on a single row |
| RAGAS evaluation cost bloat | Eval cost > generation cost | Use cheaper judge for sanity, more expensive judge for final | Already current state; Gemini Flash judge is fine for v1.0 |
| Gemini structured-output rate limits | NaN spike on multi-hop class | Add per-call retry with exponential backoff; reduce concurrency | Has happened in this repo (commits `fb397f0`, `186a9c2`) — patches in place but new RAGAS version may regress |

---

## "Looks Done But Isn't" Checklist

- [ ] **Tier 5 rerun:** Verify `RunResult.new_items` contains `tool_call_output_item` with non-empty `results[]` for at least one smoke-test question before kicking off the full 30-question run
- [ ] **Tier 4 rerun:** Verify `rag_anything_storage/tier-4-multimodal/` was deleted (not moved) before re-ingest, and post-ingest graphml node count matches expected (~581 figures + paper entities)
- [ ] **Provenance:** Frozen handoff doc captures (eval_date, generation_model_full_id, judge_model_full_id, judge_embedder_id, git_sha, ragas_version, openai_agents_version, lightrag_version, rag_anything_version) per tier
- [ ] **NaN breakdown:** Frozen doc distinguishes `empty_contexts` from `empty_statements` from `json_parse_failure` — current comparison.md does not
- [ ] **Embedder disclosure:** Every tier's embedder is named in the rollup, not just buried in code
- [ ] **Multimodal disclaimer:** Tiers 1-3 multimodal scores are visually de-emphasized (em-dash, "for reference only" sub-table, or strikethrough), not just disclaimer-footnoted
- [ ] **Self-grading disclosure:** Top-of-doc disclosure block names the family-bias paper and quantifies the expected magnitude
- [ ] **Sample-size disclaimer:** "n=30, magnitudes-only" is in the headline area, not the appendix
- [ ] **Reproducibility:** Someone with the SHA + corpus + golden_qa.json can rerun and get within ±0.05 of every reported number
- [ ] **Comparison.md ↔ frozen-doc consistency:** SHAs and timestamps match; if not, the frozen doc supersedes
- [ ] **Latency reporting:** Median + p95 reported, or mean is explicitly disclaimed as outlier-sensitive
- [ ] **Cost tracker pricing:** `shared/pricing.py` was last updated for current model versions (check commit date vs eval date)

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Tier 5 NaN persists after rerun | MEDIUM (~1 hour debug + $0.30 rerun) | (1) Smoke-test 5 questions with full trace dump (2) Confirm `file_search_call` results populated (3) If empty: check vector store population; if populated: fix harness extraction (4) Re-run Tier 5 only |
| Tier 4 graphml partial state | LOW (~30 min + $0.30 rerun) | Wipe `rag_anything_storage/tier-4-multimodal/` entirely; re-ingest from corpus; verify graph stats; rerun Tier 4 only |
| Self-grading bias surfaces in critique | LOW (disclosure) / MEDIUM (cross-judge run, $0.30) | Disclosure language ready in frozen doc; if hostile feedback, run cross-judge on 10 questions as v1.1 |
| Sample-size critique | LOW (disclosure) | Lead the frozen doc with the n=30 caveat; bootstrap CI is a v1.1 add |
| Embedder-confound critique | LOW (disclosure) | Per-tier embedder table in frozen doc |
| RAGAS NaN spike on rerun | MEDIUM (depends on root cause) | Distinguish empty_contexts / empty_statements / json_parse — first two are real, third is a retry-with-backoff fix |
| Provenance SHA mismatch | LOW (rerun rollup script) | Run `python -m evaluation.harness.compare` after final eval; commit the regenerated comparison.md atomically with the frozen doc |
| Latency outlier dominates mean | LOW (disclosure or recompute) | Report median + p95; or filter the >p99 row and disclose |

---

## Pitfall-to-Phase Mapping

| Pitfall | Tag | Phase | Verification |
|---------|-----|-------|--------------|
| 1. Tier 5 NaN | SHIP-BLOCKER | eval-rerun | Smoke test confirms `file_search_call` returns non-empty `results[]`; full rerun produces <5/30 NaN |
| 2. Tier 4 graphml partial | SHIP-BLOCKER | eval-rerun | Storage dir wiped; graphml stats logged; full rerun produces <5/30 NaN |
| 3. Self-grading bias | DISCLOSE-AND-SHIP | frozen-doc | Disclosure block present in handoff; cites Panickssery 2024 + Wataoka 2024 |
| 4. RAGAS faithfulness NaN | SHIP-BLOCKER (if rate >5%) | eval-rerun | nan_reason column present in metrics CSV; per-tier NaN rate <5% post-rerun |
| 5. Embedder mismatch | DISCLOSE-AND-SHIP | frozen-doc | Per-tier embedder table in frozen doc; disclosure note above rollup |
| 6. Sample size n=30 | DISCLOSE-AND-SHIP | frozen-doc | "n=30, magnitudes-only" disclaimer in lead position |
| 7. openai-agents tool routing | SHIP-BLOCKER | eval-rerun | Agent instructions force retrieval; trace verifies `file_search_call` for all 30 |
| 8. Provenance SHA drift | DISCLOSE-AND-SHIP | frozen-doc | Rollup auto-generated; library versions captured; single SHA across all tiers |
| 9. Latency cold-cache | DISCLOSE-AND-SHIP | frozen-doc | Median + p95 reported OR mean is disclaimed |
| 10. Multimodal text-only scores | DISCLOSE-AND-SHIP | frozen-doc | Visual de-emphasis (em-dash / sub-table) for Tiers 1-3 multimodal column |

---

## Disclosure Language Templates

For copy-paste into the frozen handoff doc.

**Self-grading bias (Pitfall 3):**

> All RAGAS metrics in this report were graded by `google/gemini-2.5-flash`. Tiers 2-4 use Gemini-family models for generation. Published research (Panickssery et al. NAACL 2024; Wataoka et al. 2024 — *Self-Preference Bias in LLM-as-a-Judge*) documents that LLM judges assign systematically higher scores to outputs from their own model family, with the effect attributed to lower perplexity on familiar text rather than quality recognition. The numbers below should therefore be read as *relative within this benchmark* and not as absolute quality estimates. A cross-family judge sanity-check (Claude or GPT-4 grading the same outputs) is listed as a v1.1 follow-up.

**Sample size (Pitfall 6):**

> n=30 questions × 5 tiers. This sample size is below the threshold where bootstrap confidence intervals are reliable (Efron & Tibshirani: n≥30 is the floor; n≥100 is recommended for CI work). We report magnitudes — relative tier ordering and gap sizes — not statistical significance. A 0.05-0.10 gap between two tiers' faithfulness scores is within plausible noise for n=30. Treat the *ranking* as the signal; treat exact values as approximate.

**Embedder mismatch (Pitfall 5):**

> Each tier ships with the embedding model native to its architecture (see per-tier table). Tier 1 uses `text-embedding-3-small`; Tier 2 uses Google's managed embedding for File Search; Tier 3-4 use `text-embedding-3-small` via LightRAG; Tier 5 uses OpenAI's hosted embedding inside the vector store. Forcing a single embedder across tiers would defeat the purpose of comparing vertically-integrated products (Tier 2, Tier 5) against composable stacks (Tier 1, 3, 4). A consequence is that observed score differences confound retrieval architecture with embedding-model quality. Where the same embedder is used across tiers (Tiers 1, 3, 4 — `text-embedding-3-small`), score differences can be more confidently attributed to architecture.

**Multimodal disclaimer (Pitfall 10):**

> Tiers 1, 2, 3 are text-only architectures. Their reported scores on multimodal questions reflect what happens when text-only retrieval is asked to answer image-grounded questions — the LLM generates plausible-sounding answers from incomplete context. These scores are reported for reference but should not be read as evidence that text-only RAG handles multimodal queries. Tier 4 (RAG-Anything) is the only architecture in this comparison that ingests image content; Tier 5 (openai-agents) inherits whatever multimodal capability OpenAI's hosted file search provides as of $eval_date.

---

## Sources

**RAGAS:**
- [Issue #1651 — "No statements were generated from the answer"](https://github.com/explodinggradients/ragas/issues/1651)
- [Issue #1150 — "Failed to parse output. Returning None" on faithfulness](https://github.com/explodinggradients/ragas/issues/1150)
- [Issue #2073 — Gemini 2.0-flash structured output](https://github.com/explodinggradients/ragas/issues/2073)
- [Issue #1162 — Problem with answer_relevancy metric](https://github.com/explodinggradients/ragas/issues/1162)
- [Issue #1192 — Answer Relevancy giving same questions every time](https://github.com/explodinggradients/ragas/issues/1192)
- [Issue #1829 — answer_relevancy worse in non-English](https://github.com/vibrantlabsai/ragas/issues/1829)
- [Issue #2566 — 0.4.3 only scores, no explanations](https://github.com/vibrantlabsai/ragas/issues/2566)
- [Faithfulness docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Response Relevancy docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)
- [Context Precision docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Context Recall docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [RAGAS Metrics Explained — Saulius blog](https://saulius.io/blog/ragas-rag-evaluation-metrics-llm-judge)

**openai-agents SDK:**
- [Tools docs (FileSearchTool, hosted vs function tools)](https://openai.github.io/openai-agents-python/tools/)
- [Results docs (RunResult.new_items, final_output)](https://openai.github.io/openai-agents-python/results/)
- [Context management docs (RunContextWrapper)](https://openai.github.io/openai-agents-python/context/)
- [Running agents docs (Runner.run, tracing)](https://openai.github.io/openai-agents-python/running_agents/)
- [Vector store status race condition (community)](https://community.openai.com/t/bug-vector-store-status-completed-does-not-guarantee-searchability-file-search-returns-empty-results-silently/1374471)
- [Vector store file linking failure (community)](https://community.openai.com/t/uploaded-file-is-not-linked-to-vector-store/1336290)
- [FileSearchTool intermittent 500s (community)](https://community.openai.com/t/intermittent-500-error-on-the-new-agents-sdk-retrieving-metadata-with-filesearchtool/1154789)

**LightRAG / RAG-Anything:**
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [RAG-Anything paper (arxiv 2510.12323)](https://arxiv.org/html/2510.12323v1)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)

**Judge bias literature:**
- [Self-Preference Bias in LLM-as-a-Judge (Wataoka et al. 2024, arxiv 2410.21819)](https://arxiv.org/abs/2410.21819)
- [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (Ye et al. 2024)](https://llm-judge-bias.github.io/)
- [LLM Evaluators Recognize and Favor Their Own Generations (Panickssery et al. NAACL 2024)](https://www.researchgate.net/publication/397200002_LLM_Evaluators_Recognize_and_Favor_Their_Own_Generations)
- [A Systematic Study of Position Bias in LLM-as-a-Judge (IJCNLP 2025)](https://aclanthology.org/2025.ijcnlp-long.18.pdf)
- [Play Favorites: Statistical Method to Measure Self-Bias (arxiv 2508.06709)](https://arxiv.org/html/2508.06709v1)

**Embedder / benchmarking methodology:**
- [Beyond Benchmarks: Evaluating Embedding Model Similarity for RAG (arxiv 2407.08275)](https://arxiv.org/html/2407.08275v1)
- [RAG vs GraphRAG: Systematic Evaluation (arxiv 2502.11371)](https://arxiv.org/html/2502.11371v3)

**Statistical methodology:**
- [Bootstrapping (statistics) — Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Bootstrap small-sample limitations (MIT OCW)](https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/3894e9e72ac0f3d1c98d43323a104fec_MIT15_450F10_lec09.pdf)

---
*Pitfalls research for: RAG comparison eval ship-readiness*
*Researched: 2026-05-04*
