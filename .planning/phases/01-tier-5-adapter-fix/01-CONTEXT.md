# Phase 1: Tier 5 Adapter Fix - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the hard-coded `retrieved_contexts=[]` at `evaluation/harness/adapters/tier_5.py:125` with a walk over `RunResult.new_items` filtered by `ToolCallOutputItem`, extracting tool outputs into structured RAGAS-ready context strings. Verify on a 5-question smoke test before any full 30-question rerun budget is committed. Tier 4 graphml regen is Phase 2; full 5-tier rerun is Phase 7.

</domain>

<decisions>
## Implementation Decisions

### Context extraction shape

- **Granularity:** One entry per chunk/result. `search_text_chunks` output is split into N entries (one per chunk); each `lookup_paper_metadata` output is its own entry. RAGAS `context_precision@K` needs a real list with K>1 to be meaningful.
- **Entry content:** Text-only with a short provenance prefix. Format: `[paper_id=<id>] <text>` where text is the chunk text (for `search_text_chunks`) or the abstract field (for `lookup_paper_metadata`). Strip JSON braces and field names — RAGAS judges natural prose.
- **Tool scope:** Both tools feed `retrieved_contexts`. `lookup_paper_metadata` outputs are included via the abstract field only (title/authors/year are metadata, not grounding text).
- **Aggregation across iterations:** Concat all `ToolCallOutputItem` outputs in agent-iteration order, then dedup by stable id (`chunk_id` for chunks, `paper_id` for metadata lookups). Repeated tool calls don't inflate `context_precision`.

### Smoke test question selection

- **Selection method:** Deterministic question IDs. Hard-pick a specific 5 from the golden manifest and reuse them every smoke run.
- **Class mix:** 3 single-hop + 2 multi-hop. **No multimodal** in the smoke set — Tier 5 retrieves from text-only chunks (Tier 1's ChromaDB), so multimodal questions don't have fair retrieval here.
- **Stability across phases:** Same 5 IDs reused for Phase 1 (Tier 5 smoke), Phase 2 (Tier 4 smoke), and any future smoke checks. Apples-to-apples deltas; one row of provenance.
- **Location:** CLI flag `--smoke-question-ids id1,id2,...` on `evaluation/harness/run.py` with a hard-coded default constant. Default reproduces the gate; flag allows ad-hoc overrides without code edits. The actual 5 IDs are picked during planning by reading `dataset/manifests/` and selecting deterministically.

### Smoke pass/fail threshold

- **Population gate:** ≥4/5 populated `retrieved_contexts`. Allows one legitimate Pitfall 9 self-cite case while requiring strong evidence the fix landed.
- **Score gate:** Also require RAGAS `faithfulness` and `context_precision` to return non-NaN on the populated cells. Population alone doesn't guarantee usefulness — score.py runs against the smoke 5 as part of the gate.
- **Truncated cells (`error="max_turns_exceeded"`, Pitfall 8):** Excluded from the denominator, logged separately in the smoke report.
- **Zero-tool-call cells (Pitfall 9 — agent self-cited from training):** Excluded from the denominator, logged separately. The rate is surfaced in the report so we can decide later if it's acceptable.
- **Minimum measurable denominator:** If exclusions leave fewer than 3 measurable cells, the smoke is **inconclusive** — re-run with a different 5 or investigate. Otherwise apply the ≥0.8 ratio (effectively ≥4/5-equivalent) on the measurable subset.

### Fallback policy when smoke fails

- **Ordering:** Instrument first, mutate second. If the initial smoke fails, install the `debug-tier5` extra (`openinference-instrumentation-openai-agents==1.4.2`, `arize-phoenix>=6,<8`, `opentelemetry-sdk>=1.30,<2`, `opentelemetry-exporter-otlp-proto-http>=1.30,<2`), re-run smoke with `set_tracing_disabled(disabled=False)`, and observe Phoenix spans to confirm: (a) the agent is calling tools, (b) tools are returning data, (c) the adapter is seeing `ToolCallOutputItem` in `result.new_items`. Only after spans tell us *what is happening* do we attempt STACK.md mutations (simplify schema → switch model slug → bump openai-agents).
- **Exit policy:** Phase 1 ships when the smoke gate passes (≥4/5 populated AND non-NaN RAGAS on the measurable subset). If after instrumentation + the 3 STACK.md mutations the gate still fails, **mark Phase 1 incomplete, write a fallback-log doc, and escalate to the user before Phase 7**. No silent partial-ship.
- **Provenance for any degraded ship:** If the user explicitly authorizes shipping a partial fix (e.g., 3/5 measurable populated instead of 4/5), write `evaluation/results/diagnostics/tier-5-fallback-{TS}.json` recording every attempt (mutation tried, smoke result, span observations). Phase 9's frozen doc gets an explicit "Tier 5: partial-fix caveat" line citing that JSON. Honest provenance, no hidden caveats.

### Claude's Discretion

- **Per-attempt budget:** How many smoke runs / dollars / wall-clock per fallback step before calling it "tried". Claude picks based on signal quality (e.g., one smoke run is usually enough; flaky truncations may justify a re-run).
- **Specific 5 question IDs:** Selected during planning by reading `dataset/manifests/papers.json` and picking 3 single-hop + 2 multi-hop with deterministic, defensible criteria.
- **Dedup tie-breaking:** Which entry wins when the same `chunk_id` appears with different scores across iterations (probably first occurrence; not consequential for RAGAS).
- **Smoke-report shape:** Schema of the smoke-result JSON written for the gate (Claude designs to match existing harness conventions).
- **Tool-error handling:** What to do if a `ToolCallOutputItem` has an error payload rather than a result (probably skip with a logged warning).

</decisions>

<specifics>
## Specific Ideas

- The fix itself is ~5 lines per STACK.md ("a five-line code change inside the eval adapter"). Anything beyond that is scaffolding (smoke harness, fallback logging, gate evaluation) — keep the core change minimal.
- `set_tracing_disabled(disabled=True)` at `tier-5-agentic/agent.py:43` will need to become **conditional** (env-flag or arg) so OpenInference can capture spans when `debug-tier5` is active, without changing default runtime behavior.
- Pitfall 8 (Pitfall 8 of 130-RESEARCH) and Pitfall 9 (Pitfall 9 of 130-RESEARCH) are load-bearing — the fix must preserve their semantics: `MaxTurnsExceeded` still surfaces `error="max_turns_exceeded"`, and the agent's self-citation behavior in answer text is not removed (we add contexts alongside, we don't change the agent prompt).
- Smoke test runs in-process via the existing `run.amain()` path; no new entry point. The CLI flag `--smoke-question-ids` should slot into the existing argparse layout in `evaluation/harness/run.py`.
- Frozen-doc-grade provenance: any fallback log must record git SHA, timestamp, library versions (`openai-agents`, `lightrag-hku`, `raganything`, `ragas`), and the model slug used per smoke run.

</specifics>

<deferred>
## Deferred Ideas

- **OpenInference + Phoenix as a runtime dep:** They stay in the `debug-tier5` optional extra (CLEANUP-02 in REQUIREMENTS.md v1.1) — diagnostic-only, not committed to the default install.
- **Adding new tools to the Tier 5 agent** (e.g., a multimodal lookup): out of scope. The fix concerns *how the adapter consumes tool outputs*, not which tools exist.
- **Bootstrap 95% CI on the smoke gate** (METH-01): smoke uses point estimates only; full CI work is v1.1.
- **Multi-judge spot-check on the smoke 5:** Phase 8 owns the multi-judge re-score on a 5×3 subset; the smoke gate uses Gemini-only scoring.

</deferred>

---

*Phase: 01-tier-5-adapter-fix*
*Context gathered: 2026-05-04*
