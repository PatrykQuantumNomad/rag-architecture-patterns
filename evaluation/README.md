> [!WARNING]
> **Full-corpus evaluation costs ~$0.30-1.00** (Tier inference + RAGAS judge LLM combined; ~30 questions × ≤5 tiers × 3 metrics × ~3 internal LLM calls per metric).
> **Tier 4 is deferred to the user** (Phase 130 SC-1 — sandbox kernel-level OMP shmem block on MineRU; not solvable here). Run Tier 4 locally via `python tier-4-multimodal/scripts/eval_capture.py`.
> **Judge LLM cost is auditable separately** in `evaluation/results/costs/ragas-judge-*.json` (D-13 schema; same `shared.cost_tracker.CostTracker` infra as the tier costs).

# Evaluation Harness — Phase 131

3-stage RAG benchmarking pipeline over the 30-question golden Q&A in `evaluation/golden_qa.json`:

```
Stage 1 (capture)   →  evaluation/results/queries/{tier}-{ts}.json
Stage 2 (score)     →  evaluation/results/metrics/{tier}-{ts}.json
Stage 3 (compare)   →  evaluation/results/comparison.md
```

Each stage is independently re-runnable. Re-running Stage 2 with a different judge LLM doesn't re-pay Stage 1 (tier inference cost). Re-running Stage 3 doesn't re-pay either. This decoupling is the point.

The harness compares five RAG architectures answering the **same 30 questions** against the same corpus, using **RAGAS** (`faithfulness`, `answer_relevancy`, `context_precision`) plus a captured-cost ledger (D-13 schema, same `shared.cost_tracker.CostTracker` as every per-tier `main.py`). The output `comparison.md` is committed to git and Phase 133's `BLOG-04` imports it verbatim.

## Quickstart

From repo root:

```bash
# Once: install evaluation deps + the tiers you want to drive live
uv pip install -e ".[evaluation,tier-1,tier-2,tier-3,tier-5]"   # Tier 4 separately if you have MineRU

# Stage 1 — capture per-tier query logs (live; ~$0.20-0.50 for tiers 1+2+3+5)
python -m evaluation.harness.run --tiers 1,2,3,5 --yes

# Stage 2 — score the query logs with RAGAS (judge LLM ~$0.20-0.50; offline-replayable)
python -m evaluation.harness.score --tiers 1,2,3,5 --yes

# Stage 3 — emit comparison.md (free; pure file I/O)
python -m evaluation.harness.compare --tiers 1,2,3,5

# Read the result
cat evaluation/results/comparison.md
```

### Tier 4 (deferred to user)

Phase 130 SC-1 deferred Tier 4's live test — the sandbox can't run MineRU (kernel-level OMP shmem block). Drive Tier 4 locally:

```bash
# 1. Local: drive Tier 4 over the 30 questions (writes a QueryLog JSON)
python tier-4-multimodal/scripts/eval_capture.py --yes

# 2. Local: re-run Stages 1-3 including the cached Tier 4 log
python -m evaluation.harness.run --tiers 4 \
    --tier-4-from-cache evaluation/results/queries/tier-4-{TIMESTAMP}.json --yes
python -m evaluation.harness.score --tiers 4 --yes
python -m evaluation.harness.compare --tiers 1,2,3,4,5
```

The `eval_capture.py` helper writes a `QueryLog` matching the harness schema; `run.py --tier-4-from-cache PATH` simply re-emits the cached records under the canonical filename. No live Tier 4 calls happen inside `run.py` — the deferral is enforced at the orchestrator level.

## CLI Reference

| Command                                                              | Purpose                                                  |
|----------------------------------------------------------------------|----------------------------------------------------------|
| `python -m evaluation.harness.run --tiers 1,2,3,5 --yes`             | Stage 1 capture (full sweep)                             |
| `python -m evaluation.harness.run --tiers 1 --limit 5 --yes`         | Stage 1 smoke (5 questions on Tier 1)                    |
| `python -m evaluation.harness.run --tiers 4 --tier-4-from-cache PATH`| Tier 4 cached pass-through                               |
| `python -m evaluation.harness.score --tiers 1,2,3,5 --yes`           | Stage 2 RAGAS scoring (judge: gemini-2.5-flash via OpenRouter) |
| `python -m evaluation.harness.score --judge-model openrouter/anthropic/claude-haiku-4.5 --yes` | Alt judge model |
| `python -m evaluation.harness.compare --tiers 1,2,3,4,5`             | Stage 3 emit comparison.md                               |
| `python tier-4-multimodal/scripts/eval_capture.py --yes`             | User's local Tier 4 capture (Phase 130 SC-1 deferral)    |

`--help` on any of the three CLIs prints the full flag list. All three accept `--yes` to bypass cost-surprise prompts.

## Expected Cost (~30 questions, full sweep)

| Stage         | Cost                                  | Note                                                      |
|---------------|---------------------------------------|-----------------------------------------------------------|
| Tier 1 inference | ~$0.005-0.01                       | OpenAI `text-embedding-3-small` + `gemini-2.5-flash` via OpenRouter |
| Tier 2 inference | ~$0.0003                           | Gemini 2.5 Flash with FileSearch (managed; tiny)          |
| Tier 3 inference | ~$0.30                             | LightRAG hybrid mode; ~25 LLM calls / question over the graph |
| Tier 5 inference | ~$0.03-0.05                        | Multi-tool agent with `max_turns=10`                      |
| Tier 4 inference (user-local) | ~$0.05-0.10              | RAG-Anything hybrid mode (cost on top of one-time MineRU ingest ~$1-2) |
| RAGAS judge LLM | ~$0.20-0.50                        | `gemini-2.5-flash` via OpenRouter; `faithfulness × answer_relevancy × context_precision` × ~3 internal calls each |
| **Full sweep total** | **~$0.55-1.00**                | Includes judge cost; excludes Tier 4 first-time MineRU ingest |

Numbers are estimates from Phase 128-06 / 129-06 / 129-07 / 130-06 SUMMARY actuals + 131-RESEARCH §A1. The first live run prints actual costs via `evaluation/results/costs/`.

## Persistence

```
evaluation/results/
├── queries/   ← Plan 04 outputs (gitignored — regenerable)
│   ├── tier-1-2026-04-27T12_00_00Z.json
│   ├── tier-2-2026-04-27T12_00_00Z.json
│   └── ...
├── costs/     ← D-13 frozen schema (committed; Phase 128 precedent)
│   ├── tier-1-eval-2026-04-27T12_00_00Z.json
│   ├── ragas-judge-tier-1-2026-04-27T12_00_00Z.json
│   └── ...
├── metrics/   ← Plan 05 outputs (gitignored — regenerable)
│   └── tier-{N}-2026-04-27T12_00_00Z.json
└── comparison.md   ← Plan 06 output (committed; Phase 133 BLOG-04 imports)
```

`queries/` and `metrics/` are gitignored because they're regenerable from inputs (golden_qa.json + cached costs). `costs/` is committed per the D-13 frozen schema decision (Phase 128 precedent — auditable cost ledger). `comparison.md` is committed because Phase 133 BLOG-04 imports it verbatim into the essay.

## Sandbox SOCKS5 + socksio recipe

When running inside Claude Code's sandbox, OpenRouter HTTP traffic is routed through a SOCKS5 proxy. The same recipe Phase 128/129/130 use applies to RAGAS (LiteLLM under the hood is `httpx`; `httpx` auto-detects `ALL_PROXY` via `trust_env=True`):

```bash
# Already exported by the sandbox; just confirm:
echo "$ALL_PROXY"   # socks5h://localhost:61994

# Sandbox-only patch (NOT in pyproject.toml):
uv pip install socksio
```

If `socksio` is missing, every OpenRouter call surfaces `httpx.UnsupportedProtocol: Request URL has an unsupported protocol 'socks5h://'`. The fix is documented per Pattern 11 in `.planning/phases/131-evaluation-harness/131-RESEARCH.md`.

Outside the sandbox (your laptop, CI), no `socksio` install is needed and `ALL_PROXY` should be unset.

## Known Limitations

- **Tier 4 is deferred to user** (Phase 130 SC-1 — sandbox OMP shmem block). Tier 4 needs MineRU's ~3-5 GB layout/OCR models which the sandbox can't load. Run locally.
- **30 questions × ≤5 tiers is too small for statistical-significance testing.** `comparison.md` emits raw means + `n` + `n_NaN`; the magnitude (not p-values) is what the blog post discusses.
- **Multi-hop ≡ cross-document for our corpus.** All 10 multi-hop entries cite ≥2 source papers; no separate cross-document bucket. Verified in Phase 127-06 SUMMARY.
- **Tiers 1-3 are text-only.** Their multimodal-question scores reflect the modality limitation; Tier 4 is the multimodal-RAG win.
- **Judge LLM = `google/gemini-2.5-flash`** (same family as Tiers 2-4 generators). Self-grading bias is acknowledged; documented in `comparison.md` footer + 131-RESEARCH Open Q4.
- **Empty contexts → NaN, never zero.** Tier 5 may legitimately decline retrieval (Pitfall 9 in 131-RESEARCH); Tier 5's `MaxTurnsExceeded` truncations also become NaN. `np.nanmean` aggregates honestly; `n_NaN` per tier surfaces in `comparison.md`.

## Sample comparison.md (Tier 1 smoke)

```
| Tier | Faithfulness | Answer Relevancy | Context Precision | Mean Latency (s) | Total Cost (USD) | Cost / Query (USD) | n | n NaN |
|------|--------------|------------------|-------------------|------------------|------------------|--------------------|---|-------|
| tier-1 | 0.850 | 0.900 | 0.700 | 1.50 | 0.001000 | 0.001000 | 1 | 0 |
| tier-2 | — | — | — | — | — | — | 0 | 0 |
| ...
```

(See `evaluation/results/comparison.md` for the live result, when present.)

## Architecture map

```
golden_qa.json (30 q)
        |
        v
Stage 1: harness.run --tiers 1,2,3,4,5
        |
   per (tier × question) → adapter.run_tierN(question, tracker)
        |
        v
results/queries/{tier}-{ts}.json    +    results/costs/tier-{N}-eval-{ts}.json
        |
        v
Stage 2: harness.score
        |
   per tier → ragas.evaluate(dataset, [F, AR, CP], judge_llm)
        |
        v
results/metrics/{tier}-{ts}.json    +    results/costs/ragas-judge-{tier}-{ts}.json
        |
        v
Stage 3: harness.compare (queries + costs + metrics → comparison.md)
        |
        v
evaluation/results/comparison.md  ← Phase 133 BLOG-04 imports verbatim
```

## Reused by

- **Phase 132** — Source verification + diagrams (uses `comparison.md` numbers in `StatHighlight` components per BLOG-03).
- **Phase 133** — Blog post (BLOG-04 imports the tier rollup table verbatim into the essay).

For research backing, see `.planning/phases/131-evaluation-harness/131-RESEARCH.md`. For the per-pitfall mitigations, see § Common Pitfalls in that file.
