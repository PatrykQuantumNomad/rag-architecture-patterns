# Evaluation

Every architecture tier answers the **same 30 questions** against the same corpus. The questions live in `golden_qa.json`; the harness that runs each tier and scores the answers is built in **Phase 131 (Evaluation Harness)** — Phase 127 only commits the schema and the question split.

## Golden Q&A

`golden_qa.json` will hold 30 hand-authored entries with the locked split:

| Bucket | Count | What it tests |
|--------|-------|---------------|
| Single-hop (text) | 10 | One paper has the answer; retrieval-precision benchmark |
| Multi-hop (text) | 10 | Answer requires synthesis across 2+ papers; favors graph/agentic tiers |
| Multimodal | 7 | Answer requires reading a figure or chart; only Tier 4 should ace it |
| Video | 3 | Answer comes from the video transcript; tests Tier 4's heterogeneous indexing |

The split is fixed (per CONTEXT decision D-04) so cross-tier comparisons remain stable as the corpus evolves.

### Entry schema

```jsonc
{
  "id": "qa-001",
  "question": "...",
  "expected_answer": "...",
  "source_papers": ["paper_2003.07101", "paper_2104.08663"],
  "modality_tag": "text",          // "text" | "multimodal" | "video"
  "hop_count_tag": "single-hop"    // "single-hop" | "multi-hop"
}
```

`source_papers` is the human-curated ground truth — the harness uses it to compute retrieval precision/recall independently of answer quality. `modality_tag` and `hop_count_tag` let the harness slice cost/quality by question class.

The actual entries are authored in Plan 06.

## Cost tracking

`results/costs/` holds per-run cost JSON, one file per `(tier, run_id)`. The schema is **stable** (per CONTEXT decision D-13) so the blog Phase 133 can ingest it directly without translation.

```jsonc
{
  "tier": "tier-1-naive",
  "run_id": "2026-04-25T14:30:00Z",
  "model_calls": [
    { "model": "gemini-2.5-flash", "input_tokens": 1234, "output_tokens": 567, "cost_usd": 0.0012 }
  ],
  "totals": {
    "input_tokens": 12340,
    "output_tokens": 5670,
    "cost_usd": 0.012
  }
}
```

The directory itself is gitignored except for `.gitkeep` — runtime cost JSON does not get committed. Anyone running an evaluation locally produces their own cost file.

## Harness scope (Phase 131)

The harness will:

1. Load `golden_qa.json` and `dataset/manifests/metadata.json`.
2. For each tier, ask all 30 questions and record `(answer, retrieved_papers, latency, cost)`.
3. Compute retrieval precision/recall vs. `source_papers`, and dispatch the answer to an LLM judge for quality scoring.
4. Emit a per-tier results JSON plus an aggregate comparison table — the latter feeds the blog post directly.

Phase 127 deliberately stops at the schema. Implementing the harness against a stable schema in a later phase keeps Plan 02–06 unblocked.
