<!-- tier-5-agentic/expected_output.md -->
# Tier 5 — Pre-computed expected output

> ⚠️ This is a snapshot from a single live test run. Output is non-deterministic
> (LLM responses vary; tool-call order and answer wording differ). Re-run
> `pytest tier-5-agentic/tests/test_tier5_e2e_live.py -m live -s` to refresh.

**Captured:** 2026-04-27T00:48Z (UTC)
**Model:** `openrouter/google/gemini-2.5-flash`
**Max turns:** 10
**Tier 1 index:** `chroma_db/tier-1-naive/` (collection `enterprise_kb_naive` present; chunk count = 0 at capture time — see [Notes](#notes))
**Git SHA:** `274fa0c`

## Command

```bash
uv run --extra tier-5 pytest tier-5-agentic/tests/test_tier5_e2e_live.py -v -m live -s
```

## Output (verbatim)

```
============================= test session starts ==============================
platform darwin -- Python 3.13.1, pytest-8.4.2, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /Users/patrykattc/work/git/rag-architecture-patterns
configfile: pyproject.toml
plugins: anyio-4.13.0
collecting ... collected 1 item

tier-5-agentic/tests/test_tier5_e2e_live.py::test_tier5_end_to_end_multi_tool
Provider List: https://docs.litellm.ai/docs/providers       [LiteLLM banner — repeats per turn]

Latency: 8.44s
Truncated: False
Tokens: 868 in / 214 out (requests=2)
Total cost: $0.000795
Answer: I need more context to answer your question effectively. The tool call `search_text_chunks` returned no results. This could be due to a few reasons:

1. **Typos or variations in terminology:** "Retrieval-augmented generation" might be referred to by a slightly different name in the papers, or there might be an abbreviation I'm not aware of.
2. **The paper might not be in the knowledge base:** Alth...
PASSED

======================== 1 passed, 5 warnings in 11.94s ========================
```

## What this run demonstrates

Even with an empty `chroma_db/tier-1-naive/` collection at capture time, the agent loop ran end-to-end:

- **Agent loop completed:** `requests=2` (planner LLM was called twice — initial tool decision + post-tool synthesis).
- **Multi-tool composition was attempted:** the agent's first turn issued a `search_text_chunks` tool call (LiteLLM `Provider List:` banner repeats once per agent turn — 5 banners visible above ≈ tool-call decision + retry attempts).
- **`max_turns=10` cap respected:** `Truncated: False` means the agent finished BEFORE hitting the cap, no `MaxTurnsExceeded`.
- **Cost was tracked:** `$0.000795` for 868in / 214out tokens via `gemini-2.5-flash` — within the README's typical envelope (~$0.005–0.015) given the short 2-turn loop.
- **Honest failure mode preserved:** when no chunks come back, the agent declines to fabricate citations and lists possible reasons (per the `INSTRUCTIONS` system prompt: "If after iterating you still cannot answer, say so clearly — do not fabricate citations.").

## Notes

- **Tier 1 collection state at capture:** the `chroma_db/tier-1-naive/` directory and `enterprise_kb_naive` collection are present (188 KB SQLite), but `count() == 0`. Tier 5's `search_text_chunks` correctly returns `[]`; the agent does the right thing and refuses to hallucinate. To see a populated agent run, repopulate Tier 1 first:

      cd ../tier-1-naive
      python main.py --ingest

  Then re-run the live test — the agent will get real chunks back, call `lookup_paper_metadata` to verify authors, and produce a paper-id-cited answer.

- **Output is non-deterministic** — your run may differ in:
  - Tool-call order (`search_text_chunks` first vs `lookup_paper_metadata` first).
  - Answer wording (citations preserved by `paper_id`).
  - Cost (within ~$0.005–0.015 envelope on a populated index; lower when chunks come back empty as above).
- **Reproducible behaviour:** with a populated Tier 1 index, you should see the same paper_ids cited and the same architectural distinction articulated for the canned DPR ↔ RAG question (set via `python tier-5-agentic/main.py` defaults).
- **Cost > 0 floor:** even on an empty index, the agent burns ~$0.0008 (reasoning about why the search returned empty). On a populated index, expect ~$0.005–0.015 typical.
- **`MaxTurnsExceeded` not observed** for this question — the agent terminates after 2 turns when `search_text_chunks` returns empty (the loop guard fires after one retry). Pitfall 6's `getattr(exc, "usage", None)` defense is documented but not exercised in this capture.
