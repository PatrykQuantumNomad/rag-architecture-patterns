<!-- tier-4-multimodal/expected_output.md -->
# Tier 4 — Pre-computed expected output

> ⚠️ This is a snapshot from a single live test run. Output is non-deterministic
> (LLM responses vary). Re-run `pytest tier-4-multimodal/tests/test_tier4_e2e_live.py -m live`
> to refresh.

**Captured:** `<fill at capture time>` (UTC)
**Model:** `google/gemini-2.5-flash` (LLM + vision) via OpenRouter
**Embedding:** `openai/text-embedding-3-small` (1536d) via OpenRouter
**Subset:** 3 smallest papers + 5 first images from `dataset/manifests/figures.json`
**Git SHA:** `<fill at capture time>`
**Mode:** `hybrid`

## Command

```bash
# Live test invocation (preferred — drives ingest + query + cost capture):
uv run pytest tier-4-multimodal/tests/test_tier4_e2e_live.py -v -m live -s

# OR equivalent CLI invocation against the production WORKING_DIR:
uv run python tier-4-multimodal/main.py \
  --query "What does Figure 1 of the Attention Is All You Need paper depict?" \
  --yes
```

## Output (verbatim)

```
<rendered shared.display table here, populated during live test>

Ingest latency: <fill>s
Query latency: <fill>s
Total cost: $<fill>
Cost JSON: evaluation/results/costs/tier-4-test-<timestamp>.json
graphml size: <fill> bytes
Answer (truncated 300 chars): <fill>
```

## Notes

- Output is non-deterministic — your run may differ in wording, latency, and cost.
- Reproducible behaviour: same Cost/Latency order of magnitude; same paper cited if you happen to land on the smallest-3 subset.
- For users without MineRU/Docker: read this file as a reference for what success looks like.
- Captured by re-running the live test after refreshing `OPENROUTER_API_KEY` and `MINERU_CACHE_DIR`. The test is `@pytest.mark.live`-gated so non-interactive `pytest -q -m "not live"` runs do NOT trigger ingest cost.
