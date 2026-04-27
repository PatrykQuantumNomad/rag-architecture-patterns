<!-- tier-5-agentic/expected_output.md -->
# Tier 5 — Pre-computed expected output

> ⚠️ This is a snapshot from a single live test run. Output is non-deterministic
> (LLM responses vary; tool-call order and answer wording differ). Re-run
> `pytest tier-5-agentic/tests/test_tier5_e2e_live.py -m live -s` to refresh.

**Captured:** `<fill at capture time>` (UTC)
**Model:** `openrouter/google/gemini-2.5-flash`
**Max turns:** 10
**Tier 1 index:** `chroma_db/tier-1-naive/` (`<fill at capture time>` chunks, populated `<fill at capture time>`)
**Git SHA:** `<fill at capture time>`

## Command

```bash
uv run --extra tier-5 pytest tier-5-agentic/tests/test_tier5_e2e_live.py -v -m live -s
```

## Output (verbatim)

```
[ rendered shared.display table here, populated during live test ]

Latency: <fill>s
Truncated: <fill>
Tokens: <fill> in / <fill> out
Total cost: $<fill>
Answer: <fill>
```

## Notes

- Output is non-deterministic — your run may differ in:
  - Tool-call order (`search_text_chunks` first vs `lookup_paper_metadata` first)
  - Answer wording (citations preserved by `paper_id`)
  - Cost (within ~$0.005–0.015 envelope)
- Reproducible behaviour: same paper_ids cited; same architectural distinction articulated.
- Truncation is allowed but unusual for a 2-tool question; the test asserts no `MaxTurnsExceeded` for the canned probe.
